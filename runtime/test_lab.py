"""Voice Lab — isolated STT/TTS test endpoints for R&D.

Dev-only (gated by VOICE_RUNTIME_ENABLE_LAB). Lets the Socratic Books "Voice
Lab" page exercise each layer on its own — synthesize arbitrary text and hear
it, or transcribe mic audio and read it — so we can judge quality and latency
independently before combining them in the exam pipeline. Nothing here touches
the exam runtime.
"""
from __future__ import annotations

import asyncio
import io
import json
import subprocess
import time
import wave

import httpx
import numpy as np
from fastapi import APIRouter, File, Form, HTTPException, UploadFile, WebSocket
from starlette.websockets import WebSocketDisconnect
from fastapi.responses import JSONResponse, Response
from loguru import logger

from runtime.settings import get_settings

settings = get_settings()
router = APIRouter(prefix="/lab", tags=["voice-lab"])

_MISTRAL_TRANSCRIBE_URL = "https://api.mistral.ai/v1/audio/transcriptions"
_LAB_STT_SAMPLE_RATE = 16000
_LAB_STT_CHUNK_MS = 40
_LAB_STT_CHUNK_BYTES = _LAB_STT_SAMPLE_RATE * 2 * _LAB_STT_CHUNK_MS // 1000

# Lazily loaded so importing the router (and booting the runtime) stays cheap.
_kokoro = None


def _get_kokoro():
    global _kokoro
    if _kokoro is None:
        from kokoro_onnx import Kokoro

        model = settings.kokoro_model_path or None
        voices = settings.kokoro_voices_path or None
        if not model or not voices:
            raise HTTPException(
                503, "KOKORO_MODEL_PATH / KOKORO_VOICES_PATH not configured"
            )
        _kokoro = Kokoro(model, voices)
    return _kokoro


def _to_wav(samples: np.ndarray, sample_rate: int) -> bytes:
    pcm = (np.clip(samples, -1.0, 1.0) * 32767).astype("<i2").tobytes()
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sample_rate)
        w.writeframes(pcm)
    return buf.getvalue()


async def _stream_pcm_realtime(pcm: bytes):
    for i in range(0, len(pcm), _LAB_STT_CHUNK_BYTES):
        yield pcm[i : i + _LAB_STT_CHUNK_BYTES]
        await asyncio.sleep(_LAB_STT_CHUNK_MS / 1000)


async def _queue_pcm_stream(audio_queue: "asyncio.Queue[bytes | None]"):
    while True:
        chunk = await audio_queue.get()
        if chunk is None:
            break
        yield chunk


@router.websocket("/stt/stream")
async def stt_stream(websocket: WebSocket) -> None:
    """Live browser-mic PCM16 stream -> Mistral realtime -> partial/final text.

    The browser sends 16 kHz mono signed-16-bit PCM chunks as binary websocket
    frames while the user is speaking, then a JSON/text {"type":"stop"} frame.
    """
    await websocket.accept()

    from mistralai.client import Mistral
    from mistralai.client.models import (
        AudioFormat,
        RealtimeTranscriptionError,
        TranscriptionStreamDone,
        TranscriptionStreamTextDelta,
    )

    audio_queue: asyncio.Queue[bytes | None] = asyncio.Queue()
    started_at = time.perf_counter()
    first_token_at: float | None = None
    accumulated_text = ""

    async def receive_audio() -> None:
        try:
            while True:
                message = await websocket.receive()
                if message.get("type") == "websocket.disconnect":
                    break
                if message.get("bytes"):
                    await audio_queue.put(message["bytes"])
                    continue
                text = message.get("text")
                if text:
                    try:
                        payload = json.loads(text)
                    except json.JSONDecodeError:
                        payload = {"type": text}
                    if payload.get("type") == "stop":
                        break
        except WebSocketDisconnect:
            pass
        finally:
            await audio_queue.put(None)

    receive_task = asyncio.create_task(receive_audio())
    try:
        client = Mistral(api_key=settings.mistral_api_key)
        audio_format = AudioFormat(encoding="pcm_s16le", sample_rate=_LAB_STT_SAMPLE_RATE)
        async for event in client.audio.realtime.transcribe_stream(
            audio_stream=_queue_pcm_stream(audio_queue),
            model=settings.mistral_stt_model,
            audio_format=audio_format,
            target_streaming_delay_ms=settings.mistral_stt_target_delay_ms,
        ):
            if isinstance(event, TranscriptionStreamTextDelta) and event.text:
                if first_token_at is None:
                    first_token_at = time.perf_counter()
                accumulated_text += event.text
                await websocket.send_json(
                    {
                        "type": "partial",
                        "text": accumulated_text,
                        "firstTokenLatencySeconds": round(first_token_at - started_at, 3),
                    }
                )
            elif isinstance(event, TranscriptionStreamDone):
                final_text = (event.text or accumulated_text).strip()
                await websocket.send_json(
                    {
                        "type": "final",
                        "text": final_text,
                        "provider": "mistral-realtime",
                        "latencySeconds": round(time.perf_counter() - started_at, 3),
                        "firstTokenLatencySeconds": (
                            round(first_token_at - started_at, 3)
                            if first_token_at is not None
                            else None
                        ),
                        "model": settings.mistral_stt_model,
                        "targetDelayMs": settings.mistral_stt_target_delay_ms,
                    }
                )
                logger.info(
                    "[lab] live STT provider=mistral-realtime first_token={} latency={:.2f}s -> {!r}",
                    round(first_token_at - started_at, 3) if first_token_at else None,
                    time.perf_counter() - started_at,
                    final_text[:80],
                )
                break
            elif isinstance(event, RealtimeTranscriptionError):
                message = event.error.message if event.error else "unknown realtime STT error"
                await websocket.send_json({"type": "error", "error": message})
                break
    except Exception as exc:  # noqa: BLE001 - report errors to the lab UI
        logger.warning("[lab] live STT websocket failed: {}", exc)
        try:
            await websocket.send_json({"type": "error", "error": str(exc)})
        except Exception:
            pass
    finally:
        receive_task.cancel()
        try:
            await receive_task
        except asyncio.CancelledError:
            pass
        try:
            await websocket.close()
        except Exception:
            pass


@router.get("/tts/voices")
def tts_voices() -> dict:
    """List available TTS voices per engine (Kokoro for now)."""
    k = _get_kokoro()
    voices = sorted(k.get_voices())
    return {"engines": {"kokoro": voices}}


@router.post("/tts")
async def tts(
    text: str = Form(...),
    engine: str = Form("kokoro"),
    voice: str = Form("af_heart"),
    speed: float = Form(1.0),
) -> Response:
    """Synthesize `text` and return WAV audio + timing headers.

    ``X-Synth-Seconds`` = wall time to generate; ``X-Audio-Seconds`` = audio
    length; ``X-RTF`` = generate/audio (lower is faster than real time).
    """
    if not text.strip():
        raise HTTPException(400, "text is required")
    if engine != "kokoro":
        raise HTTPException(400, f"unknown engine {engine!r}")

    k = _get_kokoro()
    t0 = time.perf_counter()
    samples, sample_rate = k.create(text, voice=voice, speed=speed, lang="en-us")
    synth = time.perf_counter() - t0
    audio_secs = len(samples) / sample_rate
    logger.info(
        "[lab] TTS engine={} voice={} chars={} synth={:.2f}s audio={:.2f}s",
        engine, voice, len(text), synth, audio_secs,
    )
    return Response(
        content=_to_wav(samples, sample_rate),
        media_type="audio/wav",
        headers={
            "X-Synth-Seconds": f"{synth:.3f}",
            "X-Audio-Seconds": f"{audio_secs:.3f}",
            "X-RTF": f"{synth / max(audio_secs, 0.01):.3f}",
            "Access-Control-Expose-Headers": "X-Synth-Seconds,X-Audio-Seconds,X-RTF",
        },
    )


@router.post("/stt")
async def stt(
    audio: UploadFile = File(...),
    provider: str = Form("voxtral-batch"),
    language: str = Form("en"),
) -> JSONResponse:
    """Transcribe uploaded audio (any browser format) and return text + latency.

    Batch Voxtral for now; streaming providers (Deepgram) get their own
    streaming endpoint once a key is configured.
    """
    raw = await audio.read()
    if not raw:
        raise HTTPException(400, "empty audio")
    # Normalize whatever the browser recorded (webm/opus, mp4, ...) to the
    # format needed by each provider.
    conv = subprocess.run(
        ["ffmpeg", "-v", "quiet", "-i", "pipe:0", "-ac", "1", "-ar", "16000",
         "-f", "wav", "pipe:1"],
        input=raw, capture_output=True,
    )
    if conv.returncode != 0 or not conv.stdout:
        raise HTTPException(400, "could not decode audio (ffmpeg failed)")
    wav = conv.stdout

    if provider not in {"voxtral-batch", "mistral-realtime"}:
        raise HTTPException(400, f"provider {provider!r} not wired yet")

    if provider == "mistral-realtime":
        pcm_conv = subprocess.run(
            [
                "ffmpeg",
                "-v",
                "quiet",
                "-i",
                "pipe:0",
                "-ac",
                "1",
                "-ar",
                str(_LAB_STT_SAMPLE_RATE),
                "-f",
                "s16le",
                "pipe:1",
            ],
            input=raw,
            capture_output=True,
        )
        if pcm_conv.returncode != 0 or not pcm_conv.stdout:
            raise HTTPException(400, "could not decode audio (ffmpeg failed)")

        from mistralai.client import Mistral
        from mistralai.client.models import (
            AudioFormat,
            RealtimeTranscriptionError,
            TranscriptionStreamDone,
            TranscriptionStreamTextDelta,
        )

        t0 = time.perf_counter()
        first_token_at = None
        final_text = ""
        client = Mistral(api_key=settings.mistral_api_key)
        audio_format = AudioFormat(encoding="pcm_s16le", sample_rate=_LAB_STT_SAMPLE_RATE)
        async for event in client.audio.realtime.transcribe_stream(
            audio_stream=_stream_pcm_realtime(pcm_conv.stdout),
            model=settings.mistral_stt_model,
            audio_format=audio_format,
            target_streaming_delay_ms=settings.mistral_stt_target_delay_ms,
        ):
            if isinstance(event, TranscriptionStreamTextDelta) and event.text:
                if first_token_at is None:
                    first_token_at = time.perf_counter()
            elif isinstance(event, TranscriptionStreamDone):
                final_text = (event.text or "").strip()
                break
            elif isinstance(event, RealtimeTranscriptionError):
                message = event.error.message if event.error else "unknown realtime STT error"
                raise HTTPException(502, f"Mistral realtime STT error: {message}")

        latency = time.perf_counter() - t0
        first_token_latency = (
            round(first_token_at - t0, 3) if first_token_at is not None else None
        )
        logger.info(
            "[lab] STT provider={} first_token={} latency={:.2f}s -> {!r}",
            provider,
            first_token_latency,
            latency,
            final_text[:80],
        )
        return JSONResponse(
            {
                "text": final_text,
                "provider": provider,
                "latencySeconds": round(latency, 3),
                "firstTokenLatencySeconds": first_token_latency,
                "model": settings.mistral_stt_model,
                "targetDelayMs": settings.mistral_stt_target_delay_ms,
            }
        )

    t0 = time.perf_counter()
    data = {"model": settings.mistral_stt_model}
    if language:
        data["language"] = language
    async with httpx.AsyncClient(timeout=30.0) as client:
        resp = await client.post(
            _MISTRAL_TRANSCRIBE_URL,
            data=data,
            files={"file": ("audio.wav", wav, "audio/wav")},
            headers={"Authorization": f"Bearer {settings.mistral_api_key}"},
        )
    latency = time.perf_counter() - t0
    if resp.is_error:
        raise HTTPException(502, f"STT provider error {resp.status_code}: {resp.text[:200]}")
    text = (resp.json().get("text") or "").strip()
    logger.info("[lab] STT provider={} latency={:.2f}s -> {!r}", provider, latency, text[:80])
    return JSONResponse({"text": text, "provider": provider, "latencySeconds": round(latency, 3)})
