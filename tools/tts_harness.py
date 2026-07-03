#!/usr/bin/env python3
"""Isolated TTS harness — compare candidate TTS engines by ear + latency.

Purpose (Stage 2 of the voice overhaul test plan): let a human listen to each
TTS engine on the exact phrases that failed in QA, and see time-to-first-audio
(TTFB) and whether audio streams smoothly vs. pausing per sentence.

Runs fully local on Apple Silicon (Kokoro via onnxruntime, CPU). No mic needed.

Usage:
    # synth the built-in QA phrase set with Kokoro, save WAVs, print TTFB
    python tools/tts_harness.py --engine kokoro

    # one custom line
    python tools/tts_harness.py --engine kokoro --text "Thank you. Please continue."

    # list Kokoro voices
    python tools/tts_harness.py --engine kokoro --list-voices

Output WAVs land in ./tts_out/ so you can play them back and A/B by ear.
The QA failure modes to listen for:
  * per-sentence pauses (the Mistral runtime buffered each sentence)
  * bursts of gibberish / another language at the start of a turn
  * robotic / wrong prosody on short acknowledgements ("Thank you.", "Noted.")
"""
from __future__ import annotations

import argparse
import os
import pathlib
import time
import urllib.request
import wave

OUT = pathlib.Path(__file__).resolve().parent.parent / "tts_out"
CACHE = pathlib.Path.home() / ".cache" / "kokoro-harness"

# The exact phrases QA reported problems with, plus a real exam-length question.
QA_PHRASES = [
    "Thank you.",
    "Understood.",
    "Noted. Please continue.",
    "Yes, I'm ready.",
    "If you were advising the company before launch, what concerns would you "
    "raise about how customers might react to the data collection?",
]

KOKORO_MODEL = "https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0/kokoro-v1.0.onnx"
KOKORO_VOICES = "https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0/voices-v1.0.bin"


def _download(url: str, dest: pathlib.Path) -> None:
    if dest.exists() and dest.stat().st_size > 0:
        return
    dest.parent.mkdir(parents=True, exist_ok=True)
    print(f"  downloading {dest.name} (first run only)…")
    urllib.request.urlretrieve(url, dest)


def _write_wav(path: pathlib.Path, samples, sample_rate: int) -> None:
    import numpy as np

    path.parent.mkdir(parents=True, exist_ok=True)
    pcm = (np.clip(samples, -1.0, 1.0) * 32767).astype("<i2").tobytes()
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sample_rate)
        w.writeframes(pcm)


def run_kokoro(texts: list[str], voice: str, list_voices: bool) -> None:
    from kokoro_onnx import Kokoro

    model_path = CACHE / "kokoro-v1.0.onnx"
    voices_path = CACHE / "voices-v1.0.bin"
    _download(KOKORO_MODEL, model_path)
    _download(KOKORO_VOICES, voices_path)
    kokoro = Kokoro(str(model_path), str(voices_path))

    if list_voices:
        names = sorted(kokoro.get_voices())
        print("Kokoro voices:\n  " + "\n  ".join(names))
        print("\n(English 'af_'/'am_' = US female/male, 'bf_'/'bm_' = British)")
        return

    OUT.mkdir(parents=True, exist_ok=True)
    print(f"Kokoro voice={voice}  ->  {OUT}\n")
    for i, text in enumerate(texts, 1):
        # Streaming API: measure time to the FIRST audio chunk (TTFB) — the
        # metric that decides whether it feels responsive.
        t0 = time.perf_counter()
        first_chunk_at = None
        chunks = []
        sr = 24000
        stream = kokoro.create_stream(text, voice=voice, speed=1.0, lang="en-us")
        import asyncio

        async def _drain():
            nonlocal first_chunk_at, sr
            async for samples, sample_rate in stream:
                if first_chunk_at is None:
                    first_chunk_at = time.perf_counter()
                sr = sample_rate
                chunks.append(samples)

        asyncio.run(_drain())
        total = time.perf_counter() - t0
        ttfb = (first_chunk_at - t0) if first_chunk_at else total

        import numpy as np

        audio = np.concatenate(chunks) if chunks else np.zeros(1)
        dur = len(audio) / sr
        out = OUT / f"kokoro_{i:02d}.wav"
        _write_wav(out, audio, sr)
        print(f"  [{i}] TTFB={ttfb*1000:6.0f}ms  total={total*1000:6.0f}ms  "
              f"audio={dur:4.1f}s  rtf={total/max(dur,0.01):.2f}  -> {out.name}")
        print(f"       “{text[:70]}{'…' if len(text) > 70 else ''}”")
    print("\nPlay them:  afplay tts_out/kokoro_01.wav   (or open the folder)")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--engine", choices=["kokoro"], default="kokoro",
                    help="TTS engine (kokoro = local, Apple-Silicon friendly)")
    ap.add_argument("--voice", default="af_heart", help="Kokoro voice (see --list-voices)")
    ap.add_argument("--text", default=None, help="single line to synth (default: QA phrase set)")
    ap.add_argument("--list-voices", action="store_true")
    args = ap.parse_args()

    texts = [args.text] if args.text else QA_PHRASES
    if args.engine == "kokoro":
        run_kokoro(texts, args.voice, args.list_voices)


if __name__ == "__main__":
    main()
