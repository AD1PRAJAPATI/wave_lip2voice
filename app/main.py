import os
import shutil
import time
import uuid
from contextlib import asynccontextmanager

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse
from pydantic import BaseModel

from lipreading import LipReader

from . import config, media, voice

state = {}


@asynccontextmanager
async def lifespan(app):
    for folder in ("videos", "voices", "speech"):
        os.makedirs(os.path.join(config.DATA_DIR, folder), exist_ok=True)
    state["reader"] = LipReader(config.MODEL_PATH)
    yield


app = FastAPI(title="Wave backend", lifespan=lifespan)


def data_path(folder, name):
    return os.path.join(config.DATA_DIR, folder, name)


def save_upload(upload, path):
    with open(path, "wb") as out:
        shutil.copyfileobj(upload.file, out)


class SpeakRequest(BaseModel):
    text: str
    voice_id: str = "demo"


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/voices")
def create_voice(sample: UploadFile = File(...)):
    """Voice setup screen: save a recorded or uploaded voice sample."""
    voice_id = uuid.uuid4().hex
    raw_path = data_path("voices", f"{voice_id}.raw")
    save_upload(sample, raw_path)
    try:
        media.to_wav(raw_path, data_path("voices", f"{voice_id}.wav"))
    except Exception:
        raise HTTPException(400, "Could not read that audio file.")
    finally:
        os.remove(raw_path)
    return {"voice_id": voice_id}


@app.post("/translations")
def create_translation(video: UploadFile = File(...)):
    """Record screen: read the silent video and return the possible phrases."""
    translation_id = uuid.uuid4().hex
    raw_path = data_path("videos", f"{translation_id}.raw")
    mp4_path = data_path("videos", f"{translation_id}.mp4")
    save_upload(video, raw_path)
    try:
        media.to_silent_mp4(raw_path, mp4_path)
        started = time.time()
        phrases = state["reader"].read(mp4_path, top_k=config.TOP_K)
    except Exception as error:
        raise HTTPException(422, f"Could not read lips in this video: {error}")
    finally:
        os.remove(raw_path)
    return {
        "translation_id": translation_id,
        "phrases": [{"rank": i + 1, **p} for i, p in enumerate(phrases)],
        "time_taken_ms": int((time.time() - started) * 1000),
    }


@app.post("/translations/{translation_id}/speak")
def speak_translation(translation_id: str, request: SpeakRequest):
    """Speak screen: say the chosen phrase in the user's voice. Returns a wav file."""
    sample_path = (
        config.DEMO_VOICE_PATH if request.voice_id == "demo" else data_path("voices", f"{request.voice_id}.wav")
    )
    if not os.path.exists(sample_path):
        raise HTTPException(404, "Voice not found.")
    try:
        wav_bytes = voice.speak(request.text, sample_path)
    except Exception as error:
        raise HTTPException(502, str(error))
    out_path = data_path("speech", f"{translation_id}.wav")
    with open(out_path, "wb") as out:
        out.write(wav_bytes)
    return FileResponse(out_path, media_type="audio/wav")
