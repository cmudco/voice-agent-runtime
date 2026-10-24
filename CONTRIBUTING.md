# Contributing to the Voice Agent Runtime

Thanks for your interest in contributing. This guide covers issues, pull requests and the conventions
this repository follows.

## Filing issues

Before opening an issue:

1. Search existing issues to avoid duplicates.
2. Check that the problem reproduces on the latest `main`.
3. Gather your environment (OS, Python version, browser), the steps to reproduce, expected and actual
   behaviour, and the relevant runtime log lines.

For voice problems, also include the session ID, the time, the browser, and whether the Whisper backup
was configured. **Redact secrets and student data.** Runtime logs can contain transcript text when
debug switches are on.

### Feature requests

Describe the problem for students or instructors you want to solve, who is affected, and any
constraints. Leave the implementation open.

### Security issues

Do not open public issues for vulnerabilities. See [SECURITY.md](SECURITY.md).

---

## Pull requests

### Before opening a PR

1. Fork the repository, sync the latest `main`, and branch from it. This repository has no `dev`
   branch; pull requests target `main`.

   ```bash
   git checkout main
   git pull --ff-only origin main
   git checkout -b your-name/fix/short-description
   ```

   Naming: `<author>/<feature|fix|refactor|docs>/<short-description>`.

2. Run the checks (with the Python 3.12 virtualenv from [QUICKSTART.md](QUICKSTART.md) active):

   ```bash
   python -m compileall -q runtime scripts tests
   python -m unittest discover -s tests
   ```

3. For changes to the pipeline, STT/LLM resilience or browser events, run the relevant parts of the
   manual checklist in [docs/testing.md](docs/testing.md) with a real browser and microphone, and say
   what you ran in the PR.

4. If you add or change an environment variable, update `.env.example` and
   [docs/configuration.md](docs/configuration.md) in the same PR.

### What a PR should contain

- **What** — a one-paragraph summary.
- **Why** — the motivation; link the issue (`Closes #123`).
- **How** — the approach, and any non-obvious decision.
- **Testing** — automated tests added, and manual voice checks run.
- **Rollout notes** — changes that need a matching Socratic Bots backend or frontend change, or a
  specific deploy order.

### Review and merge

- At least one approving review from a maintainer.
- Squash-merge by default; the squashed message should describe the change.

---

## Conventions

- **Python 3.12.** Do not use features that break on it.
- **Dependencies.** Add runtime packages to `requirements.txt` with a pin. Pipecat itself stays in
  `requirements-pipecat.txt` and is installed with `--no-deps`; if you bump Pipecat, re-check its
  declared dependencies and pin what it needs in `requirements.txt`. Whisper worker packages go in
  `requirements-whisper.txt` only.
- **Settings.** Read environment variables only in `runtime/settings.py`, with a default and a short
  comment explaining non-obvious values. Changing a default changes every deployment that relies on
  it, so call it out in the PR.
- **Student data.** Never log transcript text, audio or tokens by default. Anything that does belongs
  behind an explicit debug switch that defaults to off.
- **Backend contract.** The session token format, the context endpoint and the start/end callbacks are
  shared with the Socratic Bots backend. Keep changes backward compatible, or document the deploy
  order.
- **Tests.** Use `unittest`. Mock Mistral and other providers; tests must not make network calls.

### Commit messages

`<type>: <summary>` where type is `feat`, `fix`, `refactor`, `docs`, `test` or `chore`. Keep the
summary under 70 characters and use the body for the reasoning.

---

## Using AI coding agents

This repository ships shared configuration for AI coding agents.

| File | What it is |
| --- | --- |
| `AGENTS.md` | Project instructions: architecture, conventions, gotchas. Read by Codex and other tools that follow the `AGENTS.md` convention. |
| `CLAUDE.md` | Imports `AGENTS.md`, plus Claude Code-specific notes. |
| `.agents/skills/<name>/SKILL.md` | Task procedures: `run-runtime`, `check`. |
| `.claude/skills/<name>` | Symlinks to the above, because Claude Code scans a different path. |

Invoke a skill by name (`/run-runtime` in Claude Code, `$run-runtime` in Codex) or describe the task
and let the agent pick the skill. To change a skill, edit the file under `.agents/skills/`. See
[`.agents/README.md`](.agents/README.md) for the layout and the Windows symlink caveat.

Treat these files like code: if you change how the runtime is run or tested, update the matching
skill in the same PR.

---

## Getting help

Open a draft PR and ask in the description, or open an issue with a design sketch before starting a
large change.
