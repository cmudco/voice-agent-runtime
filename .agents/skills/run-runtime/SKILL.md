---
name: run-runtime
description: Start the Socratic Bots voice runtime locally — Pipecat/FastAPI on port 8010 — including the Python 3.12 venv, the two-step Pipecat install, .env, and the health check. Use when the user wants to run, start or serve the voice runtime, test voice agents or voice exams, or asks why :8010 or a voice session is not working.
---

# Run the voice runtime

FastAPI + Pipecat service on `http://localhost:8010`. It needs a Socratic Bots backend on `:8001`
(which needs DARE on `:8000`) to have sessions to serve; `/health` works without them.

## Prerequisites

- **Python 3.12.** Pipecat's native audio stack does not install cleanly on newer versions.
- **`MISTRAL_API_KEY`** for STT and the default LLM.
- **`DARE_INTERNAL_KEY`** equal to the Socratic Bots backend's value.

## Install (once)

```bash
python3.12 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install --no-deps -r requirements-pipecat.txt
cp .env.example .env    # then set MISTRAL_API_KEY and DARE_INTERNAL_KEY
```

The `--no-deps` step is required: `pipecat-ai` declares an `onnxruntime` that is not published for
every platform, so `requirements.txt` pins Pipecat's dependencies and Pipecat is installed on top.

Keep `VOICE_RUNTIME_TTS_PROVIDER=kokoro` and `VOICE_RUNTIME_TTS_TRANSPORT=webrtc` from
`.env.example`; the code defaults (`mistral` + `http`) are a legacy fallback.

## Run

```bash
uvicorn runtime.pipecat_app:app --host 0.0.0.0 --port 8010
```

From an agent session, detach it so it outlives the session:

```bash
nohup .venv/bin/uvicorn runtime.pipecat_app:app --host 0.0.0.0 --port 8010 \
  > /tmp/voice-runtime.log 2>&1 & disown
```

## Verify

```bash
curl http://localhost:8010/health
# {"status":"ok","runtime":"pipecat","sttFallbackConfigured":false}
```

For real sessions the backend needs `VOICE_RUNTIME_PUBLIC_URL=http://localhost:8010` and the same
`DARE_INTERNAL_KEY`. The first session downloads the Kokoro model (~340 MB) to
`~/.cache/kokoro-onnx`.

## Stop

```bash
lsof -ti tcp:8010 | xargs kill
```

## Related

- `check` — tests before a PR.
- `docs/tutorials/quickstart.md` for the full walkthrough and troubleshooting; `docs/how-to/whisper-backup.md` for the
  optional STT backup worker on `:8011`.
