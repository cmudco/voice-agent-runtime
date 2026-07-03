# Voice model test harnesses

Isolated, single-model tests for the voice overhaul — run each model on its own
(the way you'd talk to it), before touching the live pipeline.

## Setup (one time)
```bash
~/.local/bin/python3.11 -m venv .venv-harness
.venv-harness/bin/pip install "mistralai[realtime]"           # STT harness
.venv-harness/bin/pip install onnxruntime numpy soundfile kokoro-onnx sounddevice  # TTS harness + mic
```
`MISTRAL_API_KEY` is read from `../.env` automatically.

## Stage 1 — STT (top priority): streaming Voxtral realtime
```bash
.venv-harness/bin/python tools/stt_harness.py --mic            # talk, watch live transcript
.venv-harness/bin/python tools/stt_harness.py --file clip.wav  # test a recording
.venv-harness/bin/python tools/stt_harness.py --mic --delay-ms 800
```
This is the streaming path the deployed runtime does NOT use (it falls back to
batch). Verified working with the current key — the old "403 billing-gated"
note in batch_stt.py is stale. Note: streaming has no language lock (batch does),
so watch for any language drift on unclear audio.

## Stage 2 — TTS: Kokoro (local, replaces Mistral TTS)
```bash
.venv-harness/bin/python tools/tts_harness.py --engine kokoro
.venv-harness/bin/python tools/tts_harness.py --engine kokoro --list-voices
afplay tts_out/kokoro_01.wav
```
Listen for: per-sentence pauses gone, no gibberish/other-language bursts on
short lines ("Thank you.", "Noted."), and TTFB well under 300 ms.
