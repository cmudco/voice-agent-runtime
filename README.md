# Voice Agent Runtime

Standalone voice runtime for Socratic Bots Mistral voice agents, built on
**Pipecat**. The runtime uses WebRTC for browser microphone/control transport,
server-side Silero VAD for turn detection, Mistral Voxtral for STT/LLM/TTS, and
HTTP WAV playback for audible agent speech.

The Django backend issues short-lived signed session tokens and stores exam
state. This service owns the live browser audio/control transport and reports
final transcripts back to Django at
`/api/voice/runtime/conversation-ended/`.

## Environment

Copy `.env.example` to `.env` and fill the secrets.

```bash
MISTRAL_API_KEY=...
SOCRATIC_BOTS_BACKEND_URL=http://localhost:8001
DARE_INTERNAL_KEY=...
VOICE_RUNTIME_SESSION_SECRET=
VOICE_RUNTIME_ICE_SERVERS=stun:stun.l.google.com:19302
VOICE_RUNTIME_CORS_ORIGINS=http://localhost:5174,http://127.0.0.1:5174
VOICE_RUNTIME_LOG_TTS_REQUEST_TEXT=false
MISTRAL_LLM_MODEL=mistral-large-latest
MISTRAL_LLM_MAX_TOKENS=180
MISTRAL_STT_MODEL=voxtral-mini-latest
MISTRAL_STT_TARGET_DELAY_MS=600
MISTRAL_TTS_MODEL=voxtral-mini-tts-2603
MISTRAL_TTS_VOICE=en_paul_neutral
VOICE_RUNTIME_VAD_CONFIDENCE=0.7
VOICE_RUNTIME_VAD_START_SECS=0.2
VOICE_RUNTIME_VAD_STOP_SECS=1.2
VOICE_RUNTIME_VAD_MIN_VOLUME=0.6
```

For production WebRTC across NAT/firewalls, configure proper STUN/TURN
infrastructure through `VOICE_RUNTIME_ICE_SERVERS`.

For V1, `DARE_INTERNAL_KEY` is intentionally reused as the shared
internal-service secret between Django and this runtime. `VOICE_RUNTIME_SESSION_SECRET`
is optional and defaults to `DARE_INTERNAL_KEY`; split it later only if key
rotation or isolation requirements justify the extra env.

## Local Run

Use Python 3.12 for the runtime. Pipecat's native audio dependency stack does
not currently resolve cleanly on newer Python versions in our setup.

```bash
cd voice-agent-runtime
python3.12 -m venv venv
source venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pip install --no-deps -r requirements-pipecat.txt
uvicorn runtime.pipecat_app:app --host 0.0.0.0 --port 8010
```

`VOICE_RUNTIME_PUBLIC_URL` on the Django side must point at this runtime, for
example `http://localhost:8010`. Django returns `webrtcUrl`, which points at:

```text
POST /sessions/<session_id>/webrtc/offer?token=<signed-token>
```

Health check: `GET /health`.

## Deployment Shape

For Socratic staging, the expected shape is:

```bash
SOCRATIC_BOTS_BACKEND_URL=https://socraticbots-dev-api.hss.cmu.edu
VOICE_RUNTIME_CORS_ORIGINS=https://socraticbots-dev.hss.cmu.edu
DARE_INTERNAL_KEY=<same value as Django>
VOICE_RUNTIME_SESSION_SECRET=
```

The Django backend must set:

```bash
VOICE_RUNTIME_PUBLIC_URL=https://socraticbots-dev-voice-api.hss.cmu.edu
```

Nginx or the platform load balancer should route that hostname to this runtime
service on port `8010`.

## Pipeline

```text
Browser mic
  -> WebRTC media track
  -> SmallWebRTCTransport
  -> MistralBatchSTT
  -> user aggregator / Silero VAD
  -> Mistral LLM
  -> BufferedMistralTTSService
  -> runtime WAV artifact
  -> tts_utterance_ready data-channel event
  -> HTTP fetch
  -> HTMLAudioElement playback
```

Barge-in is handled through Pipecat's interruption flow. Latency metrics are
captured through Pipecat observers and sent back to Django in the runtime
completion callback.
