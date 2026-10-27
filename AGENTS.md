# AGENTS.md

Guidance for AI coding agents working in the Socratic Bots voice runtime. Codex and other tools that
follow the `AGENTS.md` convention read this file directly; Claude Code reads it through the import in
`.claude/CLAUDE.md`.

Task procedures (running the runtime, checking a change) live as skills in `.agents/skills/`. See
`.agents/README.md`.

## What this is

A FastAPI + Pipecat service on port 8010 that runs live voice sessions for Socratic Bots voice agents.
The browser connects over WebRTC; the runtime does VAD → STT → LLM → TTS and reports the transcript
to the Socratic Bots backend. It has no database. The backend (separate repo,
`cmudco/socraticbots-backend`) owns agents, permissions, session records and transcripts.

Current pipeline and session flow: `docs/explanation/architecture.md`. Every setting: `docs/reference/configuration.md`.

## Commands

```bash
# Python 3.12 venv; Pipecat must be installed with --no-deps after requirements.txt
python3.12 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
pip install --no-deps -r requirements-pipecat.txt

# Run
uvicorn runtime.pipecat_app:app --host 0.0.0.0 --port 8010
curl http://localhost:8010/health

# Check
python -m compileall -q runtime scripts tests
python -m unittest discover -s tests

# Deploy gate (only when the Whisper backup is configured)
python -m runtime.preflight
```

## Layout

| Path | What |
| --- | --- |
| `runtime/pipecat_app.py` | FastAPI app: `/health`, `POST /sessions/{id}/webrtc/offer?token=`, utterance downloads |
| `runtime/pipecat_pipeline.py` | Exam/tutoring pipeline assembly and session run loop |
| `runtime/lab_pipeline.py` | STT Lab pipeline (mic → STT only, no backend callbacks) |
| `runtime/settings.py` | The only place environment variables are read |
| `runtime/session_tokens.py` | HMAC session token validation |
| `runtime/session_context.py` | Fetches session config from the backend(s) |
| `runtime/recorder.py` | Transcript, metrics, start/end callbacks |
| `runtime/stt_resilience.py` | Mistral STT wrapper, liveness watchdog, Whisper hand-off |
| `runtime/llm_resilience.py`, `runtime/llm_providers.py` | LLM fallback chain over OpenAI-compatible endpoints |
| `runtime/processors.py` | Browser data-channel events, debug processors |
| `runtime/tts_buffer.py` | HTTP-WAV TTS fallback |
| `runtime/whisper_worker.py` | Separate-process faster-whisper STT backup |
| `tests/` | `unittest` suite; `smoke_whisper.py` needs a live worker |

## Rules and gotchas

- **Production safety.** One runtime deployment can serve several backends at once. Changing a
  default in `runtime/settings.py` changes behaviour for every deployment that does not set the
  variable. Do not change defaults, the CORS always-allowed list, or `runtime/preflight.py` without
  the maintainers asking for it.
- **Pipecat install.** Never `pip install pipecat-ai` without `--no-deps`; it would replace the pinned
  `onnxruntime`. If you bump Pipecat, re-pin its dependencies in `requirements.txt`.
- **TTS defaults.** Code defaults are `VOICE_RUNTIME_TTS_PROVIDER=mistral` and
  `VOICE_RUNTIME_TTS_TRANSPORT=http` for older deployments. The tested configuration is
  `kokoro` + `webrtc`, set in `.env.example`. `webrtc` only works with `kokoro`.
- **Eagerness overrides VAD stop time.** `runtime/vad.py` maps the agent's eagerness (`low`/`normal`/
  `high`) to 2.0/1.5/1.0 s; `VOICE_RUNTIME_VAD_STOP_SECS` applies only to other values.
- **Shared secrets.** `DARE_INTERNAL_KEY` and `VOICE_RUNTIME_SESSION_SECRET` must match the backend.
  Token format: `base64url(json).hex(hmac_sha256)`.
- **Backend contract.** Context fetch is `POST` with a `GET` fallback on `405`; `404` means "try the
  next backend"; `409` means the start was rejected. Callbacks: `conversation-started/` and
  `conversation-ended/` under `/api/voice/runtime/`. Keep changes backward compatible.
- **Student data.** Never log transcript text, audio or tokens by default. New diagnostics that do
  must sit behind a debug switch that defaults to off.
- **Tests** use `unittest` and must not hit the network. Mock providers.
- **No database, no migrations.** State lives in process memory for one session.

## Conventions

- Branch from `main` (there is no `dev`): `<author>/<feature|fix|refactor|docs>/<description>`.
- Commits: `<type>: <summary>` (`feat`, `fix`, `refactor`, `docs`, `test`, `chore`).
- New environment variables go in `runtime/settings.py`, `.env.example` and
  `docs/reference/configuration.md` together.
