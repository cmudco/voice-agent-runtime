# Voice testing

Use a test assignment. Use a student account when checking attempt limits.

1. Deny the microphone. Allow it and retry. No attempt should be used.
2. Start, then cancel while connecting. No completion screen should appear.
3. Double-click Start. Open a second tab. Only one session should start.
4. Refresh or leave during startup. Old callbacks must not affect the next session.
5. Turn Wi-Fi off briefly, then on. Check the connection status and recovery.
6. Switch to backup recognition. Speak, then pause. Your words should appear.
7. Make a brief noise while the agent speaks. If no words are detected, it should resume.
8. End normally. Completion should appear only after the backend confirms it.
9. Review history. The ending reason should match what happened, or say unknown.

If something fails, send the session ID, time, browser, and what you did.
Never send passwords or tokens. Run failure tests locally, not on live exams.

## Before deployment

- Deploy the backend first, with startup claims disabled.
- Configure the backup worker URL and token. Run `python -m runtime.preflight`.
- Deploy the runtime. Test both manual and automatic backup switching.
- Enable `VOICE_RUNTIME_STARTUP_CLAIMS_ENABLED` only after every runtime uses POST claims.
- Run `python manage.py expire_voice_startups` every minute.
- Deploy the frontend after the backend's `session_status` endpoint is available.
- Test concurrent starts on PostgreSQL and real microphone/audio in a browser.
- Check Sentry delivery and backup capacity before calling the release verified.

Offline cleanup is not instant: a claimed startup has a 90-second lease.
The cleanup job may take another minute. Active exams must never be reclaimed.
Disable startup claims before rolling back to an older runtime.
