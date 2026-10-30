"""Downloads the model files while the Docker image is built, so the container starts fast."""

import os

import whisper

from models import voicecraft

VOICECRAFT_MODEL = os.environ.get("VOICECRAFT_MODEL", "330M_TTSEnhanced")
WHISPER_MODEL = os.environ.get("WHISPER_MODEL", "base.en")
MODELS_DIR = os.environ.get("MODELS_DIR", "/app/voicecraft/pretrained_models")

os.makedirs(MODELS_DIR, exist_ok=True)

whisper.load_model(WHISPER_MODEL, device="cpu")
voicecraft.VoiceCraft.from_pretrained(f"pyp1/VoiceCraft_{VOICECRAFT_MODEL}")

encodec_path = os.path.join(MODELS_DIR, "encodec_4cb2048_giga.th")
if not os.path.exists(encodec_path):
    url = "https://huggingface.co/pyp1/VoiceCraft/resolve/main/encodec_4cb2048_giga.th"
    if os.system(f"wget -q {url} -O {encodec_path}") != 0:
        raise RuntimeError("Failed to download encodec_4cb2048_giga.th")
