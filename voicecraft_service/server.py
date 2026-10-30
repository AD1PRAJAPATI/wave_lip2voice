"""Small HTTP wrapper around VoiceCraft so the main API can call it over the network.

POST /synthesize   form fields: target_text, reference_audio (wav)   ->   audio/wav
GET  /healthz      "loading" until the models are ready, then "ready"

VoiceCraft clones a voice by continuing a short real clip, so the reference clip is
transcribed with Whisper and the first few seconds are used as the voice prompt.
"""

import io
import logging
import os
import re
import tempfile
import threading

import torch
import torchaudio
import whisper
from fastapi import FastAPI, File, Form, UploadFile
from fastapi.responses import JSONResponse, Response

from data.tokenizer import AudioTokenizer, TextTokenizer
from inference_tts_scale import inference_one_sample
from models import voicecraft

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("voicecraft-service")

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
VOICECRAFT_MODEL = os.environ.get("VOICECRAFT_MODEL", "330M_TTSEnhanced")
WHISPER_MODEL = os.environ.get("WHISPER_MODEL", "base.en")
MODELS_DIR = os.environ.get("MODELS_DIR", "/app/voicecraft/pretrained_models")
PROMPT_SECONDS = float(os.environ.get("VOICECRAFT_PROMPT_SECONDS", "3.0"))

DECODE_CONFIG = {
    "top_k": 40, "top_p": 1, "temperature": 1, "stop_repetition": 3, "kvcache": 1,
    "codec_audio_sr": 16000, "codec_sr": 50, "silence_tokens": [1388, 1898, 131], "sample_batch_size": 1,
}

models = {}
models_ready = threading.Event()


def load_models():
    """Runs in the background so the server starts answering health checks immediately."""
    try:
        models["whisper"] = whisper.load_model(WHISPER_MODEL, device=str(DEVICE))
        voicecraft_model = voicecraft.VoiceCraft.from_pretrained(f"pyp1/VoiceCraft_{VOICECRAFT_MODEL}")
        voicecraft_model.to(DEVICE).eval()
        models["voicecraft"] = voicecraft_model
        models["text_tokenizer"] = TextTokenizer(backend="espeak")
        models["audio_tokenizer"] = AudioTokenizer(signature=os.path.join(MODELS_DIR, "encodec_4cb2048_giga.th"))
        log.info("VoiceCraft ready on %s", DEVICE)
    except Exception:
        log.exception("model loading failed")
    finally:
        models_ready.set()


threading.Thread(target=load_models, daemon=True).start()
app = FastAPI()


def numbers_to_words(text):
    from num2words import num2words
    return re.sub(r"\b\d+\b", lambda m: num2words(m.group(0)), text)


def prompt_text_from(words, audio_seconds):
    """Returns (text of the first PROMPT_SECONDS of speech, the exact time it ends)."""
    end_time = min(PROMPT_SECONDS, audio_seconds)
    prompt = []
    for word in words:
        if word["end"] < end_time:
            prompt.append(word["word"])
        elif (word["start"] + word["end"]) / 2 < end_time:
            prompt.append(word["word"])
            end_time = word["end"]
            break
        else:
            break
    return "".join(prompt).strip(), end_time


@app.get("/healthz")
def healthz():
    ready = models_ready.is_set() and len(models) == 4
    return {"status": "ready" if ready else "loading", "device": str(DEVICE)}


@app.post("/synthesize")
async def synthesize(target_text: str = Form(...), reference_audio: UploadFile = File(...)):
    if not models_ready.wait(timeout=280):
        return JSONResponse({"error": "Models still loading, try again shortly."}, status_code=503)
    if len(models) < 4:
        return JSONResponse({"error": "Models failed to load, check the service logs."}, status_code=500)

    folder = tempfile.mkdtemp(prefix="vc_")
    reference_path = os.path.join(folder, "reference.wav")
    try:
        with open(reference_path, "wb") as f:
            f.write(await reference_audio.read())

        info = torchaudio.info(reference_path)
        seconds = info.num_frames / info.sample_rate
        if seconds < 1.0:
            return JSONResponse({"error": "Voice sample is too short to clone from."}, status_code=400)

        transcript = models["whisper"].transcribe(reference_path, word_timestamps=True)
        words = [w for segment in transcript.get("segments", []) for w in segment.get("words", [])]
        if not words:
            return JSONResponse({"error": "No speech found in the voice sample."}, status_code=422)

        prompt_text, prompt_end = prompt_text_from(words, seconds)
        full_text = re.sub(r"\s+", " ", f"{prompt_text} {numbers_to_words(target_text)}").strip()

        _, generated = inference_one_sample(
            models["voicecraft"], models["voicecraft"].args, models["voicecraft"].args.phn2num,
            models["text_tokenizer"], models["audio_tokenizer"], reference_path, full_text,
            DEVICE, DECODE_CONFIG, int(prompt_end * info.sample_rate),
        )

        buffer = io.BytesIO()
        torchaudio.save(buffer, generated[0].cpu(), DECODE_CONFIG["codec_audio_sr"], format="wav")
        return Response(content=buffer.getvalue(), media_type="audio/wav")
    except Exception as error:
        log.exception("synthesis failed")
        return JSONResponse({"error": f"VoiceCraft synthesis failed: {error}"}, status_code=500)
    finally:
        for name in os.listdir(folder):
            os.remove(os.path.join(folder, name))
        os.rmdir(folder)
