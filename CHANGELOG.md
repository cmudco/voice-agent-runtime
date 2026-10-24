# Changelog

All notable changes to the voice agent runtime are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/). The project will use
semantic versioning once public releases begin.

## [Unreleased]

### Added

- AGPL-3.0-only `LICENSE`.
- Open-source documentation: rewritten `README.md`, `QUICKSTART.md`, `DEPLOYMENT.md`,
  `CONTRIBUTING.md`, `SECURITY.md`, and `docs/` (configuration, architecture, Whisper backup,
  testing).
- Shared AI coding agent configuration: `AGENTS.md`, `CLAUDE.md`, and `run-runtime` / `check` skills
  under `.agents/skills/`.

### Changed

- `.env.example` documents `VOICE_RUNTIME_AUDIO_INPUT_DEBUG`, `VOICE_RUNTIME_STT_LIVENESS_DEADLINE_SECS`
  and the Whisper worker settings.
- The staging deploy workflow runs only on the upstream repository, not on forks.
- The HTTP-WAV architecture note moved to `docs/history/`. Stabilization, Whisper setup and
  reliability-testing notes were folded into `docs/`.

### Removed

- An internal review handover note.
