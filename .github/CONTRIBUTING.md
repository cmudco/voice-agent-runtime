# Contributing

Thanks for helping with the voice runtime, the part of Socratic Bots that runs live voice sessions.
This guide covers issues, pull requests and the conventions this repository follows.

## Filing issues

- **Before you file.** Search existing issues, and check that the problem reproduces on the latest
  `main`.
- **Bugs.** Include your OS, Python version and browser, the steps to reproduce, expected and actual
  behaviour, and the relevant runtime log lines. For voice problems, also give the session ID, the
  time, the browser, and whether the Whisper backup was configured.
- **Redact secrets and student data.** Runtime logs can contain transcript text when debug switches
  are on.
- **Features.** Describe the problem for students or instructors, who is affected, and any
  constraints. Leave the implementation open.
- **Security issues.** Do not open a public issue. Follow [SECURITY.md](SECURITY.md).

## Pull requests

### Branching

This repository has no `dev` branch. Fork it, branch from the latest `main`, and target `main`.

```bash
git checkout main
git pull --ff-only origin main
git checkout -b your-name/fix/short-description
```

Name branches `<author>/<feature|fix|refactor|docs>/<short-description>`.

### Before you open a PR

1. Run the checks with the Python 3.12 virtualenv from
   [docs/tutorials/quickstart.md](../docs/tutorials/quickstart.md) active (or run the `check` skill):

   ```bash
   python -m compileall -q runtime scripts tests
   python -m unittest discover -s tests
   ```

2. For changes to the pipeline, STT/LLM resilience or browser events, run the relevant parts of the
   manual checklist in [docs/how-to/testing.md](../docs/how-to/testing.md) with a real browser and
   microphone, and say what you ran in the PR.
3. If you add or change an environment variable, update `runtime/settings.py`, `.env.example` and
   [docs/reference/configuration.md](../docs/reference/configuration.md) in the same PR.

This repository has no `rules.md`. The conventions below and the Traps in [AGENTS.md](../AGENTS.md)
are its coding rules.

### Conventions

- **Python 3.12.** Do not use features that break on it.
- **Dependencies.** Add runtime packages to `requirements.txt` with a pin. Pipecat stays in
  `requirements-pipecat.txt` and is installed with `--no-deps`; if you bump Pipecat, re-check its
  declared dependencies and pin what it needs in `requirements.txt`. Whisper worker packages go in
  `requirements-whisper.txt` only.
- **Settings.** Read environment variables in `runtime/settings.py`, with a default and a short
  comment for non-obvious values. Changing a default changes every deployment that relies on it, so
  call it out in the PR.
- **Student data.** Never log transcript text, audio or tokens by default. Anything that does belongs
  behind an explicit debug switch that defaults to off.
- **Backend contract.** The session token format, the context endpoint and the start/end callbacks
  are shared with the Socratic Bots backend. Keep changes backward compatible, or document the deploy
  order.
- **Tests.** Use `unittest`. Mock Mistral and other providers; tests must not make network calls.

### PR description

- **What.** A one-paragraph summary.
- **Why.** The motivation; link the issue (`Closes #123`).
- **How.** The approach, and any non-obvious decision.
- **Testing.** Automated tests added, and manual voice checks run.
- **Rollout notes.** Changes that need a matching Socratic Bots backend or frontend change, or a
  specific deploy order.

### Review and merge

- At least one approving review from a maintainer.
- Squash-merge by default; the squashed message describes the change.

### Commit messages

`<type>: <summary>`, where type is `feat`, `fix`, `refactor`, `docs`, `test` or `chore`. Keep the
summary under 70 characters and put the reasoning in the body.

```text
fix: replay the buffered turn after an STT rebuild
feat: add Gemini to the LLM fallback chain
docs: document the Whisper preflight gate
```

## Using AI coding agents

[AGENTS.md](../AGENTS.md) holds the commands and traps; Codex reads it directly and Claude Code
imports it through `.claude/CLAUDE.md`. Task procedures live in `.agents/skills/` (`run-runtime`,
`check`), symlinked into `.claude/skills/`. Invoke one by name (`/run-runtime` in Claude Code,
`$run-runtime` in Codex). See [`.agents/README.md`](../.agents/README.md) for the
layout and the Windows symlink caveat. If you change how the runtime is run or tested, update the
matching skill in the same PR.

## Getting help

Open an issue (with a design sketch before a large change) or a draft PR with your question in the
description. You can also email the project team at vks@andrew.cmu.edu.
