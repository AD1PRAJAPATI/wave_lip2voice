import subprocess


def to_silent_mp4(src, dst):
    """Re-encodes any phone/browser video to a plain mp4 without audio, which is what the lipreading model reads."""
    subprocess.run(
        ["ffmpeg", "-y", "-i", src, "-c:v", "libx264", "-pix_fmt", "yuv420p",
         "-preset", "fast", "-crf", "22", "-an", dst],
        check=True, capture_output=True,
    )


def to_wav(src, dst, max_seconds=10):
    """Converts a voice sample to mono 16 kHz wav, the format VoiceCraft expects."""
    subprocess.run(
        ["ffmpeg", "-y", "-i", src, "-t", str(max_seconds), "-vn", "-ac", "1", "-ar", "16000", dst],
        check=True, capture_output=True,
    )
