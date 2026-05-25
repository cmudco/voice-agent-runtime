"""Batch Mistral STT processor (drop-in for the realtime WebSocket STT).

The realtime transcription WebSocket (``wss://api.mistral.ai/v1/audio/
transcriptions/realtime``) is billing-gated and currently returns HTTP 403
on the free Experiment tier. The standard batch endpoint
(``POST /v1/audio/transcriptions``) works on the same key without
billing - that is what this processor uses.

Behavior:
- Buffer raw PCM s16le mono audio between ``VADUserStartedSpeakingFrame``
  and ``VADUserStoppedSpeakingFrame`` from the user aggregator's Silero
  VAD (same signal the realtime service flushed on).
- On VAD stop, POST the buffered audio as a WAV file and push one
  ``TranscriptionFrame`` downstream with the result.
- Buffer is bounded; very short utterances are dropped to avoid spamming
  the API with silence.
"""

from __future__ import annotations

import io
import wave

import httpx
from loguru import logger

from pipecat.frames.frames import (
    Frame,
    InputAudioRawFrame,
    StartFrame,
    TranscriptionFrame,
    VADUserStartedSpeakingFrame,
    VADUserStoppedSpeakingFrame,
)
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor
from pipecat.transcriptions.language import Language
from pipecat.utils.time import time_now_iso8601

# 16 kHz mono s16le -> 32_000 bytes/sec.
_MIN_SPEECH_BYTES = 32_000 // 5      # ~200 ms - drop turns shorter than this
_MAX_SPEECH_BYTES = 32_000 * 60      # ~60 s - safety cap

_MISTRAL_TRANSCRIBE_URL = "https://api.mistral.ai/v1/audio/transcriptions"


class MistralBatchSTT(FrameProcessor):
    """Pipecat STT processor backed by Mistral's batch transcription API."""

    def __init__(
        self,
        *,
        api_key: str,
        model: str,
        language: str | None = None,
        sample_rate: int = 16000,
    ):
        super().__init__()
        self._api_key = api_key
        self._model = model
        self._language_code = language  # e.g. "en"; for the form field
        self._language_enum = (
            Language.EN if language and language.lower().startswith("en") else None
        )
        self._sample_rate = sample_rate
        self._buffer = bytearray()
        self._speaking = False

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)

        if isinstance(frame, StartFrame):
            self._sample_rate = frame.audio_in_sample_rate or self._sample_rate

        elif isinstance(frame, VADUserStartedSpeakingFrame):
            self._buffer = bytearray()
            self._speaking = True

        elif isinstance(frame, VADUserStoppedSpeakingFrame):
            audio = bytes(self._buffer)
            self._buffer = bytearray()
            self._speaking = False
            if len(audio) >= _MIN_SPEECH_BYTES:
                self.create_task(self._transcribe_and_emit(audio), name="batch_stt_post")

        elif isinstance(frame, InputAudioRawFrame) and self._speaking:
            self._buffer.extend(frame.audio)
            if len(self._buffer) > _MAX_SPEECH_BYTES:
                # Keep only the most recent window to bound memory.
                del self._buffer[: -_MAX_SPEECH_BYTES]

        await self.push_frame(frame, direction)

    async def _transcribe_and_emit(self, pcm: bytes) -> None:
        try:
            text = await self._post_mistral(pcm)
        except Exception as exc:  # noqa: BLE001 - STT failure must not crash the turn
            logger.warning("Mistral batch STT failed: {}", exc)
            return
        text = (text or "").strip()
        if not text:
            return
        await self.push_frame(
            TranscriptionFrame(
                text,
                "",
                time_now_iso8601(),
                language=self._language_enum,
            ),
            FrameDirection.DOWNSTREAM,
        )

    async def _post_mistral(self, pcm: bytes) -> str:
        wav = _pcm_to_wav(pcm, self._sample_rate)
        data: dict[str, str] = {"model": self._model}
        if self._language_code:
            data["language"] = self._language_code
        files = {"file": ("speech.wav", wav, "audio/wav")}
        headers = {"Authorization": f"Bearer {self._api_key}"}
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.post(
                _MISTRAL_TRANSCRIBE_URL,
                data=data,
                files=files,
                headers=headers,
            )
            response.raise_for_status()
        return response.json().get("text", "")


def _pcm_to_wav(pcm: bytes, sample_rate: int) -> bytes:
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)  # int16
        w.setframerate(sample_rate)
        w.writeframes(pcm)
    return buf.getvalue()
