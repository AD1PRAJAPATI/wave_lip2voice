import os

MODEL_PATH = os.environ.get("MODEL_PATH", "weights/pytorch_model.pt")
DATA_DIR = os.environ.get("DATA_DIR", "data")
VOICECRAFT_URL = os.environ.get("VOICECRAFT_URL", "http://localhost:8081").rstrip("/")
DEMO_VOICE_PATH = os.environ.get("DEMO_VOICE_PATH", "assets/demo_voice.wav")
TOP_K = int(os.environ.get("TOP_K", "5"))
