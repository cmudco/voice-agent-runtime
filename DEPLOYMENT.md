# Voice Agent Runtime — Deployment Guide

> **Just want to run it locally?** See [QUICKSTART.md](QUICKSTART.md). This guide is for running the
> runtime on a server for real students.

The runtime is a single FastAPI process (`runtime.pipecat_app:app`) on port `8010`. In production you
put a reverse proxy with TLS in front of it, give WebRTC a working STUN/TURN path, and run the
optional Whisper backup worker next to it.

Hostnames below use `example.org`. Replace them with your own.

## Prerequisites

| Requirement | Notes |
| --- | --- |
| Linux host, Python 3.12 | Or Docker with the included `Dockerfile`. |
| CPU and RAM | Kokoro TTS and Silero VAD run on the CPU, once per active session. Benchmark concurrent sessions on your target machine before an exam; the project does not publish a capacity figure. |
| A Socratic Bots backend | Reachable from the runtime over HTTPS. |
| TLS hostname for the runtime | e.g. `voice.example.org`. Browsers only allow microphone access and WebRTC signaling from secure origins. |
| STUN/TURN | STUN is enough on open networks; most campus and corporate networks need TURN. |
| Mistral API key | Required. OpenAI / Anthropic / Gemini keys are optional fallbacks. |

## Environment

Start from `.env.example`. Settings that must be right in production:

```bash
MISTRAL_API_KEY=...
SOCRATIC_BOTS_BACKEND_URLS=https://api.example.org
VOICE_RUNTIME_CORS_ORIGINS=https://bots.example.org
DARE_INTERNAL_KEY=<same value as the Socratic Bots backend>
VOICE_RUNTIME_SESSION_SECRET=<optional; same value as the backend if set>

VOICE_RUNTIME_TTS_PROVIDER=kokoro
VOICE_RUNTIME_TTS_TRANSPORT=webrtc
VOICE_RUNTIME_ICE_SERVERS=stun:stun.example.org:3478,turn:turn.example.org:3478

VOICE_RUNTIME_LLM_INPUT_DEBUG=false
VOICE_RUNTIME_AUDIO_INPUT_DEBUG=false
VOICE_RUNTIME_LOG_TTS_REQUEST_TEXT=false

SENTRY_DSN=...                 # optional
SENTRY_ENVIRONMENT=production

VOICE_RUNTIME_WHISPER_URL=http://127.0.0.1:8011
VOICE_RUNTIME_WHISPER_TOKEN=<random secret shared with the worker>
```

On the Socratic Bots backend, set `VOICE_RUNTIME_PUBLIC_URL=https://voice.example.org` and the same
`DARE_INTERNAL_KEY` / `VOICE_RUNTIME_SESSION_SECRET`.

One runtime can serve several backends (for example staging and production). List each backend in
`SOCRATIC_BOTS_BACKEND_URLS` and each frontend in `VOICE_RUNTIME_CORS_ORIGINS`, comma-separated. The
runtime finds the backend that owns a session by asking each in turn, so every backend it serves must
share the same `DARE_INTERNAL_KEY` and `VOICE_RUNTIME_SESSION_SECRET`.

Full reference: [docs/configuration.md](docs/configuration.md).

## Install

```bash
sudo useradd --system --create-home --home-dir /srv/voice voice
sudo -u voice -i
git clone https://github.com/cmudco/voice-agent-runtime.git
cd voice-agent-runtime
python3.12 -m venv .venv
.venv/bin/pip install --upgrade pip
.venv/bin/pip install -r requirements.txt
.venv/bin/pip install --no-deps -r requirements-pipecat.txt
cp .env.example .env && chmod 600 .env   # then edit
```

Pre-download the Kokoro model so the first student does not wait for it: either start one session
after install, or download `kokoro-v1.0.onnx` and `voices-v1.0.bin` from the
[kokoro-onnx releases](https://github.com/thewh1teagle/kokoro-onnx/releases) and point
`KOKORO_MODEL_PATH` / `KOKORO_VOICES_PATH` at them.

## Whisper backup and the preflight gate

The self-hosted Whisper worker takes over speech recognition when Mistral's realtime STT keeps
failing. It is optional for the runtime itself: with `VOICE_RUNTIME_WHISPER_URL` and
`VOICE_RUNTIME_WHISPER_TOKEN` empty, the runtime starts and runs without a backup.

The deploy gate is stricter. `python -m runtime.preflight` **fails unless the backup is configured,
its model is loaded, and one authenticated test transcription succeeds**:

```bash
.venv/bin/python -m runtime.preflight
# Backup configured, model loaded, and authenticated inference passed.
```

The maintainers' `deploy.sh` runs it before restarting the service, so a deployment without a working
backup is blocked. If you do not want a backup, do not run preflight in your own deploy script. Setup
steps for the worker are in [docs/whisper-backup.md](docs/whisper-backup.md).

## systemd

`/etc/systemd/system/voice-agent-runtime.service`:

```ini
[Unit]
Description=Socratic Bots voice runtime
After=network-online.target
Wants=network-online.target

[Service]
User=voice
WorkingDirectory=/srv/voice/voice-agent-runtime
ExecStart=/srv/voice/voice-agent-runtime/.venv/bin/uvicorn runtime.pipecat_app:app --host 127.0.0.1 --port 8010
Restart=on-failure
RestartSec=3

[Install]
WantedBy=multi-user.target
```

The runtime loads `.env` itself, so the unit needs no `EnvironmentFile`. Run a single Uvicorn worker:
sessions live in process memory, and the HTTP-WAV fallback serves utterance files from the process that
created them.

```bash
sudo systemctl daemon-reload
sudo systemctl enable --now voice-agent-runtime
curl -sf http://127.0.0.1:8010/health
```

The Whisper worker gets its own unit; see [docs/whisper-backup.md](docs/whisper-backup.md#run-it-under-systemd).

## Reverse proxy (nginx)

The proxy carries signaling and the HTTP-WAV fallback only. WebRTC media does not pass through it.

```nginx
server {
    listen 443 ssl http2;
    server_name voice.example.org;

    ssl_certificate     /etc/letsencrypt/live/voice.example.org/fullchain.pem;
    ssl_certificate_key /etc/letsencrypt/live/voice.example.org/privkey.pem;

    location / {
        proxy_pass http://127.0.0.1:8010;
        proxy_set_header Host $host;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
        proxy_read_timeout 60s;
    }
}
```

CORS is handled by the runtime (`VOICE_RUNTIME_CORS_ORIGINS`); do not add CORS headers in nginx.

## WebRTC, STUN and TURN

After the offer/answer exchange over HTTPS, the browser and the runtime negotiate a direct media path
with ICE. `VOICE_RUNTIME_ICE_SERVERS` is used by the runtime; the backend passes its own
`VOICE_RUNTIME_ICE_SERVERS` to the browser, so keep the two lists the same.

- The default, `stun:stun.l.google.com:19302`, works when the runtime host has a public IP and
  students are on permissive networks.
- Networks that block UDP or use symmetric NAT need a TURN relay. Run one (for example
  [coturn](https://github.com/coturn/coturn)) on a public host and add it to both lists, including a
  `turns:` entry on 443 for networks that only allow TLS.
- The runtime host must allow inbound UDP on the ephemeral port range aiortc uses, or all media will
  go through TURN.

Test from the networks your students actually use before an exam.

## Docker

```bash
docker build -t voice-agent-runtime .
docker run -d --name voice-agent-runtime --env-file .env -p 127.0.0.1:8010:8010 voice-agent-runtime
```

WebRTC media from a container needs either host networking (`--network host`) or a TURN server,
because the container's private address is not reachable by browsers. The image does not include the
Whisper worker.

## Upgrades

```bash
git pull
.venv/bin/pip install -r requirements.txt
.venv/bin/pip install --no-deps -r requirements-pipecat.txt
.venv/bin/python -m runtime.preflight     # if you run the Whisper backup
sudo systemctl restart voice-agent-runtime
curl -sf http://127.0.0.1:8010/health
```

Restarting ends any live sessions. Schedule restarts outside exam windows.

`deploy.sh` and `.github/workflows/deploy-staging.yml` are the maintainers' own deployment scripts.
They assume a specific host layout, systemd unit name and SSH setup, and the workflow only runs on
the upstream repository. Use them as a reference and adapt them to your environment.

## Production checklist

- [ ] `.env` is not world-readable and is not committed.
- [ ] `DARE_INTERNAL_KEY` (and `VOICE_RUNTIME_SESSION_SECRET`, if used) are long random values shared
      only with the backend.
- [ ] `VOICE_RUNTIME_TTS_PROVIDER=kokoro` and `VOICE_RUNTIME_TTS_TRANSPORT=webrtc` are set explicitly.
- [ ] Debug switches (`VOICE_RUNTIME_LLM_INPUT_DEBUG`, `VOICE_RUNTIME_LOG_TTS_REQUEST_TEXT`) are off.
      `VOICE_RUNTIME_LLM_INPUT_DEBUG` writes student answers to `/tmp`.
- [ ] TURN is configured and tested from student networks.
- [ ] The Whisper worker listens on loopback only and `runtime.preflight` passes.
- [ ] `/health` is monitored.
- [ ] Concurrent-session capacity was measured on this host.
