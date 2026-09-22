# Enabling the self-hosted STT backup

The speech-to-text backup is **Whisper** (`faster-whisper`) — not the Wispr
dictation product. All the code is already on `main` in both
`voice-agent-runtime` and `socraticbooks-react`. Nothing needs merging.

There is a single voice runtime, hosted on the dev server, and both production
and dev traffic use it. So this is one deployment to configure, not two.

It is dormant only because its two environment variables are unset. That is the
intended default: empty settings keep the previous behaviour, so the feature
ships dark until someone turns it on.

## Current state

| | |
|---|---|
| Runtime code on `main` | yes |
| Frontend code on `main` | yes |
| Configured on the runtime host | **no** — this is the whole problem |
| `deploy.sh` starts the worker | **no** — it only restarts the runtime |

The backup runs as a **separate process** from the voice runtime. `deploy.sh`
does not start or supervise it, so it needs its own service unit.

## Four steps

**1. Install the worker into its own Python environment.** It must not share
the runtime's venv:

```
python3 -m venv .venv-whisper
.venv-whisper/bin/pip install -r requirements-whisper.txt
```

**2. Generate the shared token and write the config:**

```
./scripts/setup_whisper_backup.sh
```

This adds `VOICE_RUNTIME_WHISPER_URL` and `VOICE_RUNTIME_WHISPER_TOKEN` to
`.env`, backing the file up first. Re-running it reuses the existing token
rather than regenerating one.

**3. Start the worker on loopback** (the script prints this command):

```
set -a && . ./.env && set +a
export WHISPER_MODEL=tiny.en WHISPER_DEVICE=cpu WHISPER_COMPUTE_TYPE=int8 PYTHONPATH=.
.venv-whisper/bin/python -m uvicorn runtime.whisper_worker:app \
  --host 127.0.0.1 --port 8011 --no-access-log
```

Run it under a process supervisor. Never expose port 8011 publicly.

**4. Restart the runtime and verify:**

```
.venv/bin/python -m runtime.preflight
curl -sf http://127.0.0.1:8010/health
```

Preflight prints `Backup configured, model loaded, and authenticated inference
passed.` Health should report `"sttFallbackConfigured": true`.

`deploy.sh` already runs preflight as a gate before restarting, so a broken
backup blocks the deploy rather than silently shipping.

## Gotchas that cost real time

**The token must be exported, not inlined.** The worker reads it at import
time, so `env VOICE_RUNTIME_WHISPER_TOKEN=... python -m uvicorn ...` does not
reliably reach it. The symptom is `401 Unauthorized` even though both sides
hold an identical token, which looks like a mismatch and sends you hunting in
the wrong place. Export it into the environment first.

**Both variables are required, and a half-configuration refuses to boot.**
URL set with an empty token fails with `VOICE_RUNTIME_WHISPER_TOKEN is required
when Whisper is enabled`. This is deliberate — better a loud failure than a
backup everyone believes is running.

**Keep `VOICE_RUNTIME_STT_LIVENESS_DEADLINE_SECS` above 0** (default 4). The
watchdog is what triggers the switch; at 0 the backup can never fire.

**Why the UI shows nothing today.** The backup control renders only when the
runtime sends `sttFallbackAvailable: true`, which requires a configured
fallback. No variables means the flag is always false and the control never
appears. The blank UI is a symptom, not a separate bug.

## On the token

It is the only thing protecting an unauthenticated transcription endpoint that
processes student audio. Loopback-only binding is a deployment choice, and
choices drift — a container network or a quick `0.0.0.0` test would expose it.
The setup script generates and syncs it, so the cost is one command.

## Notes

The model is `tiny.en`. It has been used locally and is fast; accuracy is
adequate because this is a backup that runs only when the primary STT fails,
not the everyday path. `WHISPER_MODEL` can point at a local directory to avoid
a download on startup.

The worker transcribes one clip at a time and returns 503 when busy; the
runtime retries a busy worker four times, one second apart.
