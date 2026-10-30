# AGENTS.md

FastAPI + Pipecat service on port 8010 that runs live voice sessions for Socratic Bots voice agents.
The browser connects over WebRTC; the runtime runs VAD → STT → LLM → TTS and reports the transcript
to the Socratic Bots backend (`cmudco/socraticbots-backend`), which owns agents, permissions,
sessions and transcripts. The runtime has no database.

There is no `rules.md` here: follow the Traps below and `.github/CONTRIBUTING.md`. Procedures live
in `.agents/skills/`: `run-runtime`, `check`.

## Commands

```bash
pip install -r requirements.txt && pip install --no-deps -r requirements-pipecat.txt   # Python 3.12 venv
uvicorn runtime.pipecat_app:app --host 0.0.0.0 --port 8010
curl http://localhost:8010/health
python -m compileall -q runtime scripts tests && python -m unittest discover -s tests
python -m runtime.preflight   # deploy gate; only passes with the Whisper backup configured
```

## Traps

- **Do not change production defaults unasked.** One runtime deployment serves several backends, so
  a default in `runtime/settings.py` changes every deployment that does not set the variable. Leave
  defaults, the always-allowed CORS list and `runtime/preflight.py` alone unless maintainers ask.
- **Install Pipecat only with `--no-deps`.** Plain `pip install pipecat-ai` replaces the pinned
  `onnxruntime`. If you bump Pipecat, re-pin its dependencies in `requirements.txt`.
- **TTS code defaults are not the tested setup.** Code defaults are `VOICE_RUNTIME_TTS_PROVIDER=mistral`
  and `VOICE_RUNTIME_TTS_TRANSPORT=http`; the tested configuration is `kokoro` + `webrtc`, set in
  `.env.example`. `webrtc` only works with `kokoro`.
- **Eagerness overrides `VOICE_RUNTIME_VAD_STOP_SECS`.** `runtime/vad.py` maps eagerness `low`/`normal`/`high`
  to 2.0/1.5/1.0 s; the env value applies only to other values.
- **Secrets are shared with the backend.** `DARE_INTERNAL_KEY` and `VOICE_RUNTIME_SESSION_SECRET` must
  match it. Token format: `base64url(json).hex(hmac_sha256)` (`runtime/session_tokens.py`).
- **Keep the backend contract backward compatible.** Context fetch is `POST` with a `GET` fallback on
  `405`; `404` means try the next backend; `409` means the start was rejected
  (`runtime/session_context.py`). Callbacks are `conversation-started/` and `conversation-ended/`
  under `/api/voice/runtime/` (`runtime/recorder.py`).
- **Never log student data by default.** No transcript text, audio or tokens in logs; new diagnostics
  that need them go behind a debug switch that defaults to off.
- **Tests use `unittest` and never hit the network.** Mock Mistral and other providers.
  `tests/smoke_whisper.py` needs a live worker and is not part of the suite.
- **No database, no migrations.** State lives in process memory for one session.
- **Env vars are read in `runtime/settings.py`.** The only exceptions are `LOCAL_SENTRY_DSN`
  (`runtime/sentry.py`) and the separate Whisper worker's `WHISPER_*` (`runtime/whisper_worker.py`).
  A new variable goes in `runtime/settings.py`, `.env.example` and `docs/reference/configuration.md`
  together.
