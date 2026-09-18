# Voice stabilization review handover

Please review and improve this implementation without touching production or
overwriting unrelated local work. Read the repository instructions first.

Repositories and branches:
- `voice-agent-runtime`: `farhat/feat/context-elapsed-timestamps`, base `origin/main`.
- `socraticbooks-react`: `farhat/feat/retire-elevenlabs`, base `origin/dev`.
- Preserve the user's unrelated frontend `VoiceAgentDetail.tsx` changes and
  `.hotfix-chat-socket-errors/` directory. Do not push, deploy, or rewrite unrelated PRs.

Objective: prevent a healthy WebRTC session from silently losing speech recognition,
and communicate the true current stage without making the user diagnose providers.
Keep Mistral primary; use one bounded reconnect, then isolated self-hosted Whisper.
Whisper can be selected directly in the STT Lab and manually mid-session. Keep the
existing noise-only interruption retry and explicit end-session acknowledgement.

Read `STABILIZATION.md` for contracts, setup, privacy and deployment limitations.
Primary review surfaces: `runtime/stt_resilience.py`, `speech_policy.py`,
`voice_stage.py`, `lab_pipeline.py`, `whisper_worker.py`, and frontend
`usePipecatWebRTCSession`, `useVoiceFeedback`, `BackupSpeechControl`, `VoiceInterface`,
and `SttLab`. Local logs are produced by `scripts/local_logs.py` for each backend.

Challenge these specific risks with tests:
1. Continuous intelligible speech, silence, repeated interims and VAD false positives.
   No transcript is not by itself proof of a provider bug.
2. Reconnect timeout, manual/automatic switch races, late Mistral transcripts,
   disconnect during switching, and end during in-flight native Whisper inference.
3. Fast successive backup turns: stable lab IDs and no overwriting the next live
   audio buffer. Overflow must not silently submit a truncated answer.
4. Stage truth: text generation is not audible speech; selection is not recovery;
   a server event is not proof of browser playback. Preserve stale-event protection.
5. Noise interruption resumes once; actual new words win; no fabricated user message.
6. Completed/cancelled reflects explicit evidence, not inferred intent from duration.
7. Manual button accessibility, pending/failure copy, and older-runtime compatibility.
8. Local Sentry never reuses production DSN; local logs must not retain speech or tokens.

Run Python 3.12 `python -m unittest discover -s tests -v`, frontend
`npm run test:voice`, lint the changed files and `npm run build`. Run the synthetic
`tests/smoke_whisper.py` with `auto`, `manual`, and `direct` modes against the local
worker. Never substitute real student recordings for the synthetic fixture.

Still required before deployment: real browser/microphone acceptance, forced
provider-failure trials, selected production Whisper-model quality/latency and
concurrency benchmarks, and appropriate worker resource limits. More RAM alone
does not prove sufficient CPU capacity. Do not describe the system as flawless.

Please return concrete findings with severity, add regression tests for fixes,
and summarize remaining risks. Ask before any material infrastructure change.
