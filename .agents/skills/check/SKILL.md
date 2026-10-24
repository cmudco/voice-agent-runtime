---
name: check
description: Run the voice runtime's pre-PR checks — compile check and the unittest suite — and say which manual voice checks a change needs. Use before opening a pull request, after changing anything under runtime/, or when the user asks to test, verify or check the voice runtime.
---

# Check a voice runtime change

Run from the repository root with the Python 3.12 virtualenv active (see `run-runtime`).

```bash
python -m compileall -q runtime scripts tests
python -m unittest discover -s tests
```

The suite uses the standard library's `unittest` (no pytest needed), makes no network calls and
takes about ten seconds. Both commands must pass.

## Also check

- **New or changed environment variable?** It must appear in `runtime/settings.py`, `.env.example`
  and `docs/configuration.md`.
- **Changed a default?** Defaults apply to every deployment that does not set the variable. Flag it
  in the PR description.
- **Pipeline, STT/LLM resilience, browser events?** Unit tests do not cover real microphones or
  networks. Name the relevant items from the manual checklist in `docs/testing.md` and say whether
  they were run.
- **Whisper backup code?** `tests/smoke_whisper.py` exercises it against a live worker; see
  `docs/testing.md`.
- **Logging?** Nothing should log transcript text, audio or tokens unless behind a debug switch that
  defaults to off.

Report results plainly: which commands ran, pass/fail counts, and what was not verified.
