import requests

from .config import VOICECRAFT_URL


def speak(text, voice_sample_path):
    """Asks the VoiceCraft service to say `text` in the voice of the sample. Returns wav bytes."""
    with open(voice_sample_path, "rb") as sample:
        response = requests.post(
            f"{VOICECRAFT_URL}/synthesize",
            data={"target_text": text},
            files={"reference_audio": ("voice.wav", sample, "audio/wav")},
            timeout=120,
        )
    if response.status_code != 200:
        raise RuntimeError(f"VoiceCraft error {response.status_code}: {response.text[:200]}")
    return response.content
