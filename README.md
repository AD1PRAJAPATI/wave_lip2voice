# Wave Backend

The server behind the Wave app. It does two things:

1. **Lipreading:** you send a short silent video of a face, it sends back the 5 most likely phrases the person mouthed.
2. **Voice cloning:** you send a phrase, it says it out loud in the user's own voice.

## How to run it

You need a computer with an NVIDIA GPU, plus Docker and the NVIDIA container toolkit installed.

**Step 1.** Download this repo.
```
git clone https://github.com/AD1PRAJAPATI/wave_lip2voice.git
cd wave_lip2voice
```

**Step 2.** Download the model file `pytorch_model.pt` (about 1 GB) from this Drive folder:
https://drive.google.com/drive/folders/190usVyhLzTAm0VBv9Qj28a5amFqFrPuh?usp=sharing

Put it inside the `weights` folder, so the path is `weights/pytorch_model.pt`.

**Step 3.** (Optional) For the demo voice, put a clear 10 second voice recording at `assets/demo_voice.wav`. Skip this if the Drive folder has no such file.

**Step 4.** Start everything.
```
docker compose up --build
```
The first time takes a long time because it downloads the voice cloning models. Later starts are fast.

**Step 5.** Check it works. Open http://localhost:8080/health in your browser. You should see `{"status":"ok"}`.

**Step 6.** Try the calls. Open http://localhost:8080/docs. You can test every call there with buttons, no code needed.

Note: the voice cloning part needs about a minute to load after starting. If speaking fails right away, wait a minute and try again.

## The 3 calls the app uses

| Wave screen | Call | What you get back |
|---|---|---|
| Voice Setup | `POST /voices` | a `voice_id` |
| Camera, Review, Translating | `POST /translations` | a `translation_id` and 5 phrases |
| Spoken in your voice | `POST /translations/{translation_id}/speak` | an audio file (.wav) |

**POST /voices**
Send an audio file in the field `sample` (about 10 seconds of clear speech).
```json
{ "voice_id": "9f2c..." }
```

**POST /translations**
Send the recorded video in the field `video`.
```json
{
  "translation_id": "a41b...",
  "phrases": [
    { "rank": 1, "text": "I would like some water, please.", "confidence": 0.62 },
    { "rank": 2, "text": "Can you get the nurse for me?", "confidence": 0.21 }
  ],
  "time_taken_ms": 1840
}
```
`confidence` compares the 5 phrases with each other and adds up to 1. It is not an absolute probability.

**POST /translations/{translation_id}/speak**
Send this as JSON. You get an audio file back. Use `"demo"` as the `voice_id` for the demo voice.
```json
{ "text": "I would like some water, please.", "voice_id": "9f2c..." }
```

## What is in each folder

| Folder | What it is |
|---|---|
| `app/` | The server. `main.py` has the 3 calls above. |
| `lipreading/` | The lipreading model. Start with `pipeline.py`. |
| `voicecraft_service/` | The voice cloning service. It runs as its own container. |
| `espnet/` | Third-party building blocks for the model. Leave as is. |
| `docs/` | Backend and database design (PDF). |
| `weights/` | Where `pytorch_model.pt` goes. |
| `assets/` | Where the demo voice recording goes. |

## Settings

These are optional. The defaults work with `docker compose`.

| Name | Default | Meaning |
|---|---|---|
| `MODEL_PATH` | `weights/pytorch_model.pt` | Lipreading model file |
| `DATA_DIR` | `data` | Where uploads are saved |
| `VOICECRAFT_URL` | `http://localhost:8081` | Address of the voice cloning service |
| `DEMO_VOICE_PATH` | `assets/demo_voice.wav` | Recording used for the demo voice |
| `TOP_K` | `5` | How many phrases to return |
