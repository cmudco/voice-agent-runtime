# Voice Agent Runtime — Quick Start

Run the voice runtime on your machine and connect it to a local Socratic Bots backend.
For deploying to a server, see [docs/how-to/deploy.md](../how-to/deploy.md).

The runtime only handles voice. To have something to talk to, you also need DARE, the Socratic Bots
backend and the Socratic Bots frontend running. The
[Socratic Bots backend QUICKSTART](https://github.com/cmudco/socraticbots-backend/blob/dev/docs/tutorials/quickstart.md)
walks through that full stack; this guide picks up at the voice step.

## Prerequisites

| Requirement | Notes |
| --- | --- |
| Python 3.12 | Pipecat's native audio dependencies do not install cleanly on newer versions yet. The Docker image uses `python:3.12-slim`. |
| A Mistral API key | Used for speech-to-text (Voxtral realtime) and the default LLM. Get one at <https://console.mistral.ai/>. |
| Socratic Bots backend on `:8001` | It creates the sessions this runtime serves. |
| Internet access on first run | Kokoro TTS downloads its ONNX model and voices (about 340 MB) to `~/.cache/kokoro-onnx` on first use, unless you set `KOKORO_MODEL_PATH` / `KOKORO_VOICES_PATH`. |

## 1. Install

```bash
git clone https://github.com/cmudco/voice-agent-runtime.git
cd voice-agent-runtime
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pip install --no-deps -r requirements-pipecat.txt
```

**Why two steps?** `pipecat-ai==1.2.1` declares `onnxruntime~=1.24.3`, and that wheel is not
published for every platform we support. `requirements.txt` pins Pipecat's dependencies explicitly
(including a working `onnxruntime`), and `requirements-pipecat.txt` then installs Pipecat itself with
`--no-deps` so pip does not try to replace those pins. Installing Pipecat without `--no-deps` will
either fail or pull in the wrong `onnxruntime`.

## 2. Configure

```bash
cp .env.example .env
```

The runtime reads `.env` from the repository root. For local use you only need to set two values:

```bash
MISTRAL_API_KEY=your-mistral-key
DARE_INTERNAL_KEY=local-dev-internal-key   # must equal the backend's DARE_INTERNAL_KEY
```

The defaults in `.env.example` already point at a backend on `http://localhost:8001` and use the tested
production speech setup (`VOICE_RUNTIME_TTS_PROVIDER=kokoro`, `VOICE_RUNTIME_TTS_TRANSPORT=webrtc`).
Keep those two lines: the code defaults are `mistral` and `http`, an older fallback path.

Every variable is described in [docs/reference/configuration.md](../reference/configuration.md).

## 3. Run

```bash
uvicorn runtime.pipecat_app:app --host 0.0.0.0 --port 8010
```

## 4. Verify

```bash
curl http://localhost:8010/health
```

Expected:

```json
{"status":"ok","runtime":"pipecat","sttFallbackConfigured":false}
```

`sttFallbackConfigured` is `false` until you set up the optional
[Whisper backup](../how-to/whisper-backup.md). You do not need it locally.

## 5. Connect the Socratic Bots backend

In the backend's `.env`, set:

```bash
VOICE_RUNTIME_PUBLIC_URL=http://localhost:8010
DARE_INTERNAL_KEY=local-dev-internal-key   # same value as the runtime
MISTRAL_API_KEY=your-mistral-key           # the backend uses it to list Mistral voices
```

If you set `VOICE_RUNTIME_SESSION_SECRET` on one side, set the same value on the other. When it is
empty, both sides sign and check session tokens with `DARE_INTERNAL_KEY`.

Restart the backend. In the Socratic Bots frontend, open **Voice exams**, create an agent that uses the
Mistral voice provider, and start a session. The browser asks the backend for a signed session URL,
then connects straight to the runtime over WebRTC.

The runtime always allows the local frontend origins (`http://localhost:5173`, `http://localhost:5174`
and their `127.0.0.1` forms), so you do not need to change CORS settings for local development.

## 6. Run the tests

```bash
python -m unittest discover -s tests
```

The suite uses the standard library's `unittest` and needs no extra packages. See
[docs/how-to/testing.md](../how-to/testing.md) for the manual voice checklist.

## Troubleshooting

| Symptom | Cause and fix |
| --- | --- |
| `pip install` fails on `onnxruntime`, or Pipecat upgrades it | Pipecat was installed without `--no-deps`. Recreate the venv and follow step 1 exactly. |
| `pip` cannot find wheels for `kokoro-onnx`, `numba` or `aiortc` | Wrong Python version. Use 3.12. |
| Offer request returns `401` | `DARE_INTERNAL_KEY` (or `VOICE_RUNTIME_SESSION_SECRET`) differs between the runtime and the backend, or the token expired. Start a new session after fixing the keys. |
| Offer request returns `502 Failed to fetch session context` | The runtime cannot reach the backend. Check `SOCRATIC_BOTS_BACKEND_URLS` and that the backend is running on `:8001`. A `401` in the runtime log means the internal keys differ. |
| Offer request returns `409` | The backend no longer accepts this session start (stale tab, double click, or a second tab). Start a new session. |
| Backend says `VOICE_RUNTIME_PUBLIC_URL is not configured` | Set it in the backend's `.env` and restart the backend. |
| Session connects but the agent is silent | Confirm `VOICE_RUNTIME_TTS_PROVIDER=kokoro` and `VOICE_RUNTIME_TTS_TRANSPORT=webrtc`. The first session after install waits for the Kokoro model download; watch the runtime log. |
| The agent never hears you | Check browser microphone permission. Leave `VOICE_RUNTIME_AUDIO_INPUT_DEBUG=true` while debugging: the log then shows input levels, so you can tell a silent track from speech below the VAD gate (`VOICE_RUNTIME_VAD_MIN_VOLUME`). |
| Startup fails with `VOICE_RUNTIME_WHISPER_TOKEN is required when Whisper is enabled` | You set `VOICE_RUNTIME_WHISPER_URL` without a token. Set both or neither. |
| WebRTC connects on localhost but not across networks | You need a TURN server. See [docs/how-to/deploy.md](../how-to/deploy.md#webrtc-stun-and-turn). |
