# Voice stabilization — local changes, not deployed

## Behavior

- Interrupting a response cancels it without flushing a partial word as a completed reply.
- If that interruption produces no recognized words, retry the original reply once,
  after 2.5 seconds without a speaking signal. Any recognized word (including an
  interim), a newer request, or session termination cancels the pending retry.
- A retry regenerates from the original request. It can repeat a sentence; this
  is not sample-exact audio continuation. Genuine continuing speech must win.
- Speaking state ends on output-transport playback completion, not LLM completion.
- Explicit end/cancel requests receive a correlated acknowledgement before the
  client closes (bounded wait: 1.5 seconds). Older runtimes without acknowledgement
  still receive the end request, but the new UI reports that completion is unconfirmed.
- A transient WebRTC disconnection gets 8 seconds to recover before teardown.
- After two failed STT rebuild/replays, an enabled Whisper backup receives the
  retained turn and stays active for the rest of that session. Late Mistral
  transcripts are suppressed. Each subsequent VAD-ended turn is transcribed
  by the backup without blocking the audio frame loop.
- The UI reports reconnection/backup processing. Failed backup requests remain
  visible errors; recovered turns have health counters, not a terminal error.

## Self-hosted backup setup (not deployed)

The backup is **Whisper**, using faster-whisper, not the Wispr dictation product.
Install `requirements-whisper.txt` in a separate Python environment. Start one
worker process on loopback, with a private shared token supplied through the
environment (never commit it):

```
WHISPER_MODEL=small WHISPER_DEVICE=cpu WHISPER_COMPUTE_TYPE=int8 \
  python -m uvicorn runtime.whisper_worker:app --host 127.0.0.1 --port 8011 --no-access-log
```

Set `VOICE_RUNTIME_WHISPER_TOKEN` for both worker and runtime, and set
`VOICE_RUNTIME_WHISPER_URL=http://127.0.0.1:8011` in the runtime. Keep
`VOICE_RUNTIME_STT_LIVENESS_DEADLINE_SECS` positive. Empty backup settings retain
the previous behavior for a staged rollout. `/health` becomes ready after model
loading; preload the model before enabling fallback. `WHISPER_MODEL` can be a
local model directory to avoid a download on startup.

The worker processes one request at a time with four CPU threads by default
(`WHISPER_CPU_THREADS`). Overload returns 503 immediately. Runtime requests have
a 30-second total deadline. The worker holds its admission slot even if an HTTP
caller disconnects during native inference. Run under a process supervisor and
resource limits; do not share the voice event loop or start one model per session.
RAM alone does not establish concurrent transcription capacity: benchmark the
chosen model on the target machine before deployment. No live load test was run.

Only PCM16 mono 16kHz is accepted. Audio is kept in memory, not logged or written
to disk. A turn is bounded to 120 seconds with two queued batches per session.
Overflow, empty recognition, timeout and overload explicitly ask for repetition;
truncated answers are never silently submitted as complete. Speech predating VAD
by more than the one-second pre-roll cannot be recovered by this buffer. This is
an STT fallback, not a fix for a microphone that sends no audio or failed WebRTC.

## Diagnostics

The existing Django callback retains `metrics.sessionHealth`, including connection
success, user/agent transcript counts, retry counters, output-queue events, and
whether a user final occurred after the last agent audio was queued. These are
not proof that the browser actually played or the user heard the audio.

Every 10 seconds during the running pipeline, content-free samples retain minimum
available system memory, last free swap, peak runtime RSS, and maximum observed
event-loop scheduling delay. Linux memory fields are omitted on other platforms.
This does not capture synchronous model initialization before pipeline startup.

Exit reasons preserve explicit `manual_end`, `time_limit`, `route_change`, or a
best-effort `page_hidden_exit` report. A page-exit report cannot distinguish tab
close from refresh/navigation. Missing reports remain `unknown_disconnect`, or
`connection_not_established` when no connection succeeded. No reason is inferred
from conversation duration. Unknown client-supplied strings are not retained.

## Verification before deployment

Run `python -m unittest discover -s tests -v` with installed runtime dependencies.
Use Python 3.12 for the local runtime, per the project's run-voice instructions.

Manual voice acceptance cases:

1. Normal question/reply: agent-speaking indication lasts through outgoing audio.
2. Noise during a reply, then silence: one automatic retry, without fake user text.
3. Actual interruption: new words suppress retry; agent responds to the new turn.
4. Continuous monologue with brief pauses: no forced replay over the user.
5. Interrupt after first token: no one-word completed reply flushed to TTS.
6. End during pending retry: no later audio; end reason is preserved.
7. Navigate/refresh/close: report available evidence only, never invent an action.
8. Network drop and return within 8s: do not tear down prematurely.
9. Unrecoverable disconnect: surface connection loss and record cancellation.

## Verified locally

- Runtime regression suite: recovery, late transcript suppression, no reconnect
  after switching, active speech protection, HTTP validation, bounded buffers,
  interruption recovery and exit reasons.
- `PYTHONPATH=. python tests/smoke_whisper.py <synthetic-mono-16k-wav>` with the
  worker running and shared token in the environment: actual Pipecat pipeline,
  two stubbed failed Mistral recoveries, real local HTTP inference, exactly one
  recovered transcript, and clean pipeline shutdown. A 7.7-second synthetic
  answer was recovered in roughly 2.1 seconds using tiny.en on the local Mac.
  This is a smoke test, not a production model quality or concurrency benchmark.
- Frontend voice hook regression tests and production build.

No server capacity change, deployment, or broad Sentry suppression is included.
Real-browser microphone/WebRTC acceptance and production-sized concurrency
testing remain deployment gates; synthetic tests do not establish those.
