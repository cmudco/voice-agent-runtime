# Voice Agent Runtime

> Real-time voice runtime for **Socratic Bots** voice agents, built on Pipecat, FastAPI and WebRTC.

[![License: AGPL v3](https://img.shields.io/badge/License-AGPL_v3-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.12-3776AB.svg)](https://www.python.org/)

## Table of Contents

- [What is Socratic Bots?](#what-is-socratic-bots)
- [Overview](#overview)
- [Features](#features)
- [Quick Start](#quick-start)
- [Documentation](#documentation)
- [Tech Stack](#tech-stack)
- [What is included in this repo?](#what-is-included-in-this-repo)
- [What is not included in this repo?](#what-is-not-included-in-this-repo)
- [Stay in touch](#stay-in-touch)
- [Acknowledgements](#acknowledgements)
- [Contributors](#contributors)
- [License](#license)

## What is Socratic Bots?

Socratic Bots (formerly SocraticBooks) lets educators build AI tutors that teach by asking questions
instead of handing out answers. Instructors set up bots with their own course material, prompts and
access rules. Students chat with them in the browser, or talk to them out loud in voice sessions such
as oral exams.

Socratic Bots runs on top of [DARE](https://github.com/cmudco/dare-backend), the Dietrich Analysis
Research Education platform. Accounts, LLM chat, file processing, retrieval and billing all live in
DARE. Socratic Bots adds the teaching layer: bots, bot groups, access codes, notes and voice agents.

> **Requires the Socratic Bots backend, which requires DARE.** This runtime is not a standalone
> product. It serves voice sessions that the
> [Socratic Bots backend](https://github.com/cmudco/socraticbots-backend) creates and signs, and the
> backend in turn needs a running [DARE backend](https://github.com/cmudco/dare-backend). DARE runs
> without Socratic Bots; the reverse is not true.

## Overview

This repository is the **voice runtime**. It owns the live audio of a voice session: the browser
connects to it over WebRTC, it listens, transcribes, asks an LLM what the agent should say, speaks the
reply, and reports the transcript back to the Socratic Bots backend when the session ends. It has no
database; agents, sessions and transcripts are stored by the backend.

```mermaid
flowchart LR
    fe["Socratic Bots frontend<br/>:5174"]
    sb["Socratic Bots backend<br/>:8001"]
    dare["DARE backend<br/>:8000"]
    rt["Voice runtime<br/>:8010"]
    mistral["Mistral API<br/>Voxtral STT + LLM"]
    whisper["Whisper worker<br/>:8011 (optional)"]

    fe -->|REST: signed session URL| sb
    fe <-->|WebRTC audio + data channel| rt
    rt -->|REST: session context,<br/>start/end callbacks| sb
    sb -->|REST: accounts, files| dare
    rt -->|HTTPS| mistral
    rt -.->|HTTP, loopback| whisper
```

The Socratic Bots backend decides who may start a session and issues a short-lived signed token. The
browser presents that token when it opens the WebRTC connection. The runtime then fetches the session
configuration (prompt, voice, language, model) from the backend server-to-server, runs the
conversation, and posts the final transcript and session health metrics back.

[ElevenLabs](https://elevenlabs.io/) voice agents are a separate, alternative provider. They are
handled by the Socratic Bots backend and frontend directly and do not use this runtime.

## Features

- **WebRTC transport** for microphone audio, agent audio and control events (Pipecat
  `SmallWebRTCTransport`).
- **Server-side turn detection** with Silero VAD, tunable through environment variables.
- **Streaming speech-to-text** with Mistral Voxtral realtime.
- **STT recovery.** A liveness watchdog rebuilds a stalled STT connection and replays the buffered
  turn. An optional self-hosted [Whisper](docs/whisper-backup.md) worker takes over if Mistral keeps
  failing.
- **LLM fallback chain.** The agent's model is tried first; on an error or a slow first token the next
  model in `VOICE_RUNTIME_LLM_FALLBACKS` answers the turn. Mistral, OpenAI, Anthropic and Gemini are
  supported through their OpenAI-compatible endpoints.
- **Text-to-speech** with local Kokoro (ONNX, no API key) streamed on the WebRTC track, or Mistral
  Voxtral TTS played as buffered WAVs over HTTP.
- **Barge-in and interruption recovery.** Students can interrupt the agent; an interruption that
  produced no words resumes the agent's reply once.
- **STT Lab mode**: a microphone-to-transcript pipeline with no LLM or TTS, for testing recognition.
- **Session health metrics** (latency, retries, fallbacks, exit reason) returned to the backend, with
  no audio stored.

## Quick Start

Python 3.12, a Mistral API key, and a running Socratic Bots backend.

```bash
git clone https://github.com/cmudco/voice-agent-runtime.git
cd voice-agent-runtime
python3.12 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
pip install --no-deps -r requirements-pipecat.txt
cp .env.example .env    # set MISTRAL_API_KEY and DARE_INTERNAL_KEY
uvicorn runtime.pipecat_app:app --host 0.0.0.0 --port 8010
curl http://localhost:8010/health
```

Then set `VOICE_RUNTIME_PUBLIC_URL=http://localhost:8010` in the Socratic Bots backend. The full
walkthrough, including why the install is two steps, is in [QUICKSTART.md](QUICKSTART.md).

## Documentation

| Kind | Doc | What's in it |
| --- | --- | --- |
| Tutorial | [QUICKSTART.md](QUICKSTART.md) | Run the runtime locally and connect it to the Socratic Bots backend |
| How-to | [DEPLOYMENT.md](DEPLOYMENT.md) | Production: systemd, reverse proxy, TURN, Whisper backup, deploy gate |
| How-to | [docs/whisper-backup.md](docs/whisper-backup.md) | Setting up the self-hosted Whisper STT backup |
| How-to | [docs/testing.md](docs/testing.md) | Automated tests and the manual voice acceptance checklist |
| Reference | [docs/configuration.md](docs/configuration.md) | Every environment variable the runtime and Whisper worker read |
| Explanation | [docs/architecture.md](docs/architecture.md) | Pipeline stages, session lifecycle, tokens, backend callbacks |
| Explanation | [docs/history/http-wav-architecture.md](docs/history/http-wav-architecture.md) | Historical design note from the HTTP-WAV era |
| Project | [CONTRIBUTING.md](CONTRIBUTING.md) | Issues, pull requests, checks, AI coding agents |
| Project | [AGENTS.md](AGENTS.md) | Guide for AI coding agents working in this repo |
| Project | [SECURITY.md](SECURITY.md) | Reporting vulnerabilities and hardening notes |
| Project | [CHANGELOG.md](CHANGELOG.md) | Notable changes |

## Tech Stack

- **Python 3.12**, **FastAPI**, **Uvicorn**
- **Pipecat** 1.2 (`SmallWebRTCTransport`, aiortc)
- **Silero VAD** (local)
- **Mistral** Voxtral realtime STT, Voxtral TTS and LLMs (`mistralai`, `openai` client against
  Mistral's OpenAI-compatible API)
- **Kokoro** TTS via `kokoro-onnx` and ONNX Runtime (local)
- **faster-whisper** for the optional STT backup worker
- **Sentry** for error reporting (optional)

## What is included in this repo?

- `runtime/` — the FastAPI app, Pipecat pipelines, STT/LLM resilience, Whisper worker
- `tests/` — unit tests and a Whisper smoke test
- `scripts/` — Whisper backup setup and a metadata-only local log capture wrapper
- `requirements*.txt`, `.env.example`, `Dockerfile`
- `deploy.sh` and `.github/workflows/deploy-staging.yml` — the maintainers' own deploy scripts,
  provided as a reference

## What is not included in this repo?

- [socraticbots-backend](https://github.com/cmudco/socraticbots-backend) — creates voice sessions,
  signs tokens, stores agents and transcripts (required)
- [socraticbots-frontend](https://github.com/cmudco/socraticbots-frontend) — the web app students and
  instructors use
- [dare-backend](https://github.com/cmudco/dare-backend) — accounts, LLM access and files for the
  whole platform (required by the Socratic Bots backend)
- Model weights. Kokoro and Whisper models are downloaded on first use.

## Stay in touch

- **Found a bug or have a feature request?** Open an issue on this repository.
- **Questions or general inquiries?** Email the project team at vks@andrew.cmu.edu.

## Acknowledgements

Socratic Bots is developed at the Dietrich College of Humanities and Social Sciences, Carnegie Mellon
University, and is built on the [DARE](https://github.com/cmudco/dare-backend) platform. The runtime
is built on [Pipecat](https://github.com/pipecat-ai/pipecat).

"DARE" (Dietrich Analysis Research Education Platform) and "SocraticBots" are trademarks of Carnegie
Mellon University. For trademark, licensing, or general inquiries, contact the project team at
vks@andrew.cmu.edu.

This software calls third-party APIs (Mistral, and optionally OpenAI, Anthropic and Google Gemini).
Use of those APIs is subject to each provider's own terms of service.

## Contributors

Socratic Bots is built and maintained by the team at the Dietrich College of Humanities and Social
Sciences, Carnegie Mellon University.

**Creators**

<table>
  <tr>
    <td align="center" valign="top" width="33.33%"><img src="https://ui-avatars.com/api/?name=Sayeed+Choudhury&size=100&background=162B4B&color=fff" width="100" height="100" style="border-radius:50%" alt="Sayeed Choudhury" /><br /><sub><b>Sayeed Choudhury</b></sub><br />Creator</td>
    <td align="center" valign="top" width="33.33%"><img src="https://ui-avatars.com/api/?name=Vincent+Sha&size=100&background=162B4B&color=fff" width="100" height="100" style="border-radius:50%" alt="Vincent Sha" /><br /><sub><b>Vincent Sha</b></sub><br />Creator</td>
    <td align="center" valign="top" width="33.33%"><img src="https://ui-avatars.com/api/?name=George+Cann&size=100&background=162B4B&color=fff" width="100" height="100" style="border-radius:50%" alt="George Cann" /><br /><sub><b>George Cann</b></sub><br />Creator</td>
  </tr>
</table>

**Team**

<table>
  <tr>
    <td align="center" valign="top" width="33.33%"><img src="https://ui-avatars.com/api/?name=Carl+Skipper&size=100&background=162B4B&color=fff" width="100" height="100" style="border-radius:50%" alt="Carl Skipper" /><br /><sub><b>Carl Skipper</b></sub><br />Contributor</td>
    <td align="center" valign="top" width="33.33%"><img src="https://ui-avatars.com/api/?name=Muhammad+Abdurrehman&size=100&background=162B4B&color=fff" width="100" height="100" style="border-radius:50%" alt="Muhammad Abdurrehman" /><br /><sub><b>Muhammad Abdurrehman</b></sub><br />Team Lead</td>
    <td align="center" valign="top" width="33.33%"><img src="https://ui-avatars.com/api/?name=Brian+Wingenroth&size=100&background=162B4B&color=fff" width="100" height="100" style="border-radius:50%" alt="Brian Wingenroth" /><br /><sub><b>Brian Wingenroth</b></sub><br />Developer</td>
  </tr>
  <tr>
    <td align="center" valign="top" width="33.33%"><img src="https://ui-avatars.com/api/?name=Farhat+Abbas&size=100&background=162B4B&color=fff" width="100" height="100" style="border-radius:50%" alt="Farhat Abbas" /><br /><sub><b>Farhat Abbas</b></sub><br />Developer</td>
    <td align="center" valign="top" width="33.33%"><img src="https://ui-avatars.com/api/?name=Hariss+M&size=100&background=162B4B&color=fff" width="100" height="100" style="border-radius:50%" alt="Hariss M." /><br /><sub><b>Hariss M.</b></sub><br />Developer</td>
    <td align="center" valign="top" width="33.33%"><img src="https://ui-avatars.com/api/?name=Eira+Khan&size=100&background=162B4B&color=fff" width="100" height="100" style="border-radius:50%" alt="Eira Khan" /><br /><sub><b>Eira Khan</b></sub><br />QA</td>
  </tr>
</table>

## License

This project is licensed under the [GNU Affero General Public License v3.0](LICENSE) (AGPL-3.0-only).
© 2024–2026 Carnegie Mellon University, Dietrich College of Humanities and Social Sciences.

See the [LICENSE](LICENSE) file for the full license text, or visit
<https://www.gnu.org/licenses/agpl-3.0.en.html>.
