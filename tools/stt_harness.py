#!/usr/bin/env python3
"""Isolated STT harness — test Mistral realtime (streaming) transcription.

Purpose (Stage 1 of the voice overhaul test plan, the top priority): verify the
"correct real-time streaming" STT path — Voxtral realtime via the official
mistralai SDK — on real speech, the way you'd talk to it in an exam. This is
the streaming path the deployed runtime does NOT use (it falls back to batch;
the old "realtime is 403 billing-gated" note is stale — realtime is GA and the
SDK opens the WebSocket for you).

Two modes:
  --mic            speak into your mic; transcript prints live as you talk
  --file clip.wav  feed a recording (any format; ffmpeg normalizes to 16k mono)

Examples:
    # live mic test — talk normally, watch the transcript
    python tools/stt_harness.py --mic

    # regression-test the exact QA phrases from a recording
    python tools/stt_harness.py --file myclip.wav

    # tune latency vs accuracy (240..2400 ms; lower = snappier, higher = accurate)
    python tools/stt_harness.py --mic --delay-ms 800

Key finding this harness lets you confirm by ear/eye:
  * streaming realtime has NO documented language lock (batch does) — watch
    whether it ever drifts to another language on unclear audio. If it does,
    that's the tradeoff vs. the batch path that pins language="en".
"""
from __future__ import annotations

import argparse
import asyncio
import os
import pathlib
import subprocess
import sys
import time

# Load MISTRAL_API_KEY from the runtime .env without extra deps.
_ENV = pathlib.Path(__file__).resolve().parent.parent / ".env"
API_KEY = os.environ.get("MISTRAL_API_KEY", "")
if not API_KEY and _ENV.exists():
    for line in _ENV.read_text().splitlines():
        if line.startswith("MISTRAL_API_KEY="):
            API_KEY = line.split("=", 1)[1].strip().strip('"').strip("'")
if not API_KEY:
    sys.exit("MISTRAL_API_KEY not set (checked env and .env)")

from mistralai.client import Mistral
from mistralai.client.models import AudioFormat

SAMPLE_RATE = 16000
MODEL = os.environ.get("STT_HARNESS_MODEL", "voxtral-mini-transcribe-realtime-2602")
CHUNK_MS = 40
CHUNK_BYTES = SAMPLE_RATE * 2 * CHUNK_MS // 1000  # s16le mono


def _wav_to_pcm16(path: str) -> bytes:
    """Normalize any audio file to raw 16 kHz mono s16le via ffmpeg."""
    out = subprocess.run(
        ["ffmpeg", "-v", "quiet", "-i", path, "-ac", "1", "-ar", str(SAMPLE_RATE),
         "-f", "s16le", "-"],
        capture_output=True,
    )
    if out.returncode != 0:
        sys.exit(f"ffmpeg failed on {path}")
    return out.stdout


async def _file_stream(pcm: bytes):
    """Yield PCM chunks paced at real time so streaming behaves realistically."""
    for i in range(0, len(pcm), CHUNK_BYTES):
        yield pcm[i:i + CHUNK_BYTES]
        await asyncio.sleep(CHUNK_MS / 1000)


async def _mic_stream():
    """Yield PCM chunks from the default input device (needs sounddevice)."""
    try:
        import sounddevice as sd
    except ImportError:
        sys.exit("mic mode needs sounddevice:  .venv-harness/bin/pip install sounddevice")
    q: asyncio.Queue = asyncio.Queue()
    loop = asyncio.get_running_loop()

    def cb(indata, frames, tinfo, status):
        loop.call_soon_threadsafe(q.put_nowait, bytes(indata))

    with sd.RawInputStream(samplerate=SAMPLE_RATE, channels=1, dtype="int16",
                           blocksize=CHUNK_BYTES // 2, callback=cb):
        print("🎤 listening — speak now (Ctrl-C to stop)…\n")
        while True:
            yield await q.get()


async def run(audio_stream, delay_ms: int) -> None:
    client = Mistral(api_key=API_KEY)
    fmt = AudioFormat(encoding="pcm_s16le", sample_rate=SAMPLE_RATE)
    t0 = time.perf_counter()
    first_at = None
    final_text = []
    print(f"model={MODEL}  target_delay={delay_ms}ms\n{'-'*60}")
    async for event in client.audio.realtime.transcribe_stream(
        audio_stream=audio_stream,
        model=MODEL,
        audio_format=fmt,
        target_streaming_delay_ms=delay_ms,
    ):
        name = type(event).__name__
        text = getattr(event, "text", None)
        if name == "TranscriptionStreamTextDelta" and text:
            if first_at is None:
                first_at = time.perf_counter()
            final_text.append(text)
            sys.stdout.write(text)
            sys.stdout.flush()
        elif "Language" in name:
            print(f"\n[detected language: {getattr(event, 'language', '?')}]")
        elif "Error" in name:
            print(f"\n[ERROR] {event}")
        elif name == "TranscriptionStreamDone":
            break
    print(f"\n{'-'*60}")
    ttfb = (first_at - t0) if first_at else -1
    print(f"first-token latency: {ttfb*1000:.0f}ms" if ttfb >= 0 else "no transcript produced")
    print("FINAL:", "".join(final_text).strip() or "(empty)")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--mic", action="store_true", help="stream from the default microphone")
    g.add_argument("--file", help="audio file to transcribe (any format)")
    ap.add_argument("--delay-ms", type=int, default=1000, help="target_streaming_delay_ms (240..2400)")
    args = ap.parse_args()

    if args.mic:
        stream = _mic_stream()
    else:
        stream = _file_stream(_wav_to_pcm16(args.file))
    try:
        asyncio.run(run(stream, args.delay_ms))
    except KeyboardInterrupt:
        print("\nstopped.")


if __name__ == "__main__":
    main()
