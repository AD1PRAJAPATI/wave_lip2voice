# Wave Backend

The server behind the **Wave** app. It does two jobs:

1. **Lipreading:** takes a short silent video of a person's face and returns the 5 most likely phrases they mouthed.
2. **Voice cloning:** says a chosen phrase out loud in the user's own voice.

The app (mobile or web) talks to this server over HTTP. Everything else in the app (accounts, history, feedback) is built by the app team on top of this, see "What is left to build".

## How it fits together

```
 Wave app  --->  API (this repo, app/)  --->  Lipreading model (lipreading/)
                      |
                      +-------------------->  VoiceCraft service (voicecraft_service/)
```

Both models run on one GPU machine, as two Docker containers (see `docker-compose.yml`). They are separate because they need different, conflicting library versions.

## What each Wave screen calls

| Screen in the app | Call | What you get back |
|---|---|---|
| Voice Setup (record or upload samples) | `POST /voices` | `voice_id` |
| Camera, Review, Translating | `POST /translations` | `translation_id` and 5 phrases with confidence |
| Spoken in your voice | `POST /translations/{translation_id}/speak` | a `.wav` audio file |

The demo voice is `voice_id = "demo"`. It needs a short clear voice recording saved at `assets/demo_voice.wav`.

### POST /voices
Form upload, field name `sample`: any audio file (wav, mp3, m4a). About 10 seconds of clear speech works best.
```json
{ "voice_id": "9f2c..." }
```

### POST /translations
Form upload, field name `video`: the recording from the camera screen (any phone video format).
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
`confidence` compares the 5 phrases with each other (they add up to 1). It is not an absolute probability. The Choose Phrase screen shows it as a percentage.

### POST /translations/{translation_id}/speak
JSON body. Returns the audio file.
```json
{ "text": "I would like some water, please.", "voice_id": "9f2c..." }
```
`text` is the phrase after the user picked or edited it.

## Run it

You need a machine with an NVIDIA GPU, Docker and the NVIDIA container toolkit.

1. Download `pytorch_model.pt` (the lipreading model, about 1 GB) from the shared drive link and put it in `weights/`. It is not stored in git because of its size.
2. (Optional) put a demo voice recording at `assets/demo_voice.wav`.
3. Start everything:
   ```
   docker compose up --build
   ```
   The first build takes a while (VoiceCraft downloads its models). VoiceCraft needs about a minute to load after start, check `http://localhost:8080/health` for the API.
4. Open `http://localhost:8080/docs` to try every call in the browser.

## Folders

```
app/                  The server. main.py has the 3 calls above.
  config.py           Settings, read from environment variables.
  media.py            ffmpeg helpers (video to silent mp4, audio to wav).
  voice.py            Sends text + voice sample to the VoiceCraft service.
lipreading/           The lipreading model code.
  pipeline.py         Video file in, phrases out. Start reading here.
  model.py            The model and the beam search that produces 5 phrases.
  transforms.py       Frame preprocessing and the word vocabulary.
  detector/           Finds the face and crops the mouth region (MediaPipe).
  unigram5000.*       The word-piece vocabulary the model was trained with.
espnet/               Neural network building blocks the model uses (third-party code, leave as is).
voicecraft_service/   The voice cloning service (VoiceCraft + Whisper), its own Docker image.
docs/                 Backend and database design.
weights/              Put pytorch_model.pt here (not in git).
data/                 Uploaded videos, voice samples and generated speech (created at run time, not in git).
```

## Settings (environment variables)

| Name | Default | Meaning |
|---|---|---|
| `MODEL_PATH` | `weights/pytorch_model.pt` | Lipreading model file |
| `DATA_DIR` | `data` | Where uploads are saved |
| `VOICECRAFT_URL` | `http://localhost:8081` | Address of the voice cloning service (docker-compose sets this for you) |
| `DEMO_VOICE_PATH` | `assets/demo_voice.wav` | Recording used for the demo voice |
| `TOP_K` | `5` | How many phrases to return |

## What is left to build

This repo only covers the two models. The app team still needs to add, following `docs/Wave_Backend_Design.pdf`:

- **Login:** Amazon Cognito (sign up / sign in).
- **Database:** 4 tables (`users`, `voices`, `translations`, `suggestions`). Save the 5 phrases from `POST /translations`, then store what the user chose, edited and rated.
- **More calls:** `choose` (save the picked or edited phrase), `feedback` (accuracy, stars, comment), `history` (past phrases).
- **File storage:** videos and voices are saved on disk in `data/`. The design moves them to an S3 bucket and stores only the file path in the database.
- **Protect the API:** only logged-in users should be able to call it.

## Notes for developers

- The first request after start is slower while the GPU warms up.
- A video where no face is found returns error 422 with a message. Show the user "Please record again, make sure your face is visible".
- The lipreading model was trained for English phrases.
- Training and data preparation code was removed from this repo. It lives in the original research repo.
