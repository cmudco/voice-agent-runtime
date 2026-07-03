"""Buffered Mistral TTS and runtime WAV artifact writing."""

from pathlib import Path
import wave

from loguru import logger
import numpy as np
import soxr

from pipecat.frames.frames import OutputTransportMessageUrgentFrame, TTSAudioRawFrame
from pipecat.services.kokoro.tts import KokoroTTSService
from pipecat.services.mistral.tts import MistralTTSService


RUNTIME_AUDIO_DIR = Path("/tmp/voice-runtime-audio")


class _BufferedTTSMixin:
    """Pre-render the full utterance, then yield one consolidated audio frame.

    Shared by the Mistral and Kokoro buffered services. The base TTS service's
    ``run_tts`` streams ``TTSAudioRawFrame`` chunks; this mixin collects them,
    resamples to the transport rate, writes source + transport WAVs for the
    frontend's HTTP ``<audio>`` playback, and emits a single steady frame. That
    keeps the audio path identical regardless of engine, so swapping the engine
    needs no frontend change.
    """

    def _init_buffer(self, *, session_id: str | None, output_sample_rate: int | None) -> None:
        self._utterance_counter = 0
        self._output_sample_rate = output_sample_rate or self.sample_rate
        self._audio_dir: Path | None = None
        if session_id:
            self._audio_dir = RUNTIME_AUDIO_DIR / session_id
            try:
                self._audio_dir.mkdir(parents=True, exist_ok=True)
            except OSError as exc:  # noqa: BLE001
                logger.warning("Could not create runtime audio dir: {}", exc)
                self._audio_dir = None

    async def run_tts(self, text: str, context_id: str):
        parts: list[bytes] = []
        sample_rate = self.sample_rate
        async for frame in super().run_tts(text, context_id):
            if isinstance(frame, TTSAudioRawFrame):
                parts.append(bytes(frame.audio))
                continue
            yield frame
        if parts:
            joined = b"".join(parts)
            output_audio = _resample_s16le_mono(
                joined, from_rate=sample_rate, to_rate=self._output_sample_rate
            )
            output_audio = _pad_s16le_to_10ms(output_audio, self._output_sample_rate)
            utterance_files = self._dump_runtime_wavs(
                joined, sample_rate, output_audio, self._output_sample_rate
            )
            logger.info(
                "TTS utterance buffered: source={}Hz {:.0f}ms -> webrtc={}Hz {:.0f}ms",
                sample_rate,
                len(joined) / 2 / sample_rate * 1000,
                self._output_sample_rate,
                len(output_audio) / 2 / self._output_sample_rate * 1000,
            )
            if utterance_files:
                yield OutputTransportMessageUrgentFrame(
                    message={
                        "type": "tts_utterance_ready",
                        **utterance_files,
                        "sourceSampleRate": sample_rate,
                        "webrtcSampleRate": self._output_sample_rate,
                    }
                )
            yield TTSAudioRawFrame(
                output_audio, self._output_sample_rate, 1, context_id=context_id
            )

    def _dump_runtime_wavs(
        self,
        source_pcm: bytes,
        source_sample_rate: int,
        output_pcm: bytes,
        output_sample_rate: int,
    ) -> dict[str, str] | None:
        """Write source and transport-ready WAVs for runtime playback."""
        if not self._audio_dir or not source_pcm:
            return None
        self._utterance_counter += 1
        prefix = f"utterance-{self._utterance_counter:03d}"
        source_name = f"{prefix}-source-{source_sample_rate // 1000}k.wav"
        webrtc_name = f"{prefix}-webrtc-{output_sample_rate // 1000}k.wav"
        self._write_runtime_wav(self._audio_dir / source_name, source_pcm, source_sample_rate)
        self._write_runtime_wav(self._audio_dir / webrtc_name, output_pcm, output_sample_rate)
        return {"sourceWavName": source_name, "webrtcWavName": webrtc_name}

    def _write_runtime_wav(self, path: Path, pcm_bytes: bytes, sample_rate: int) -> None:
        try:
            with wave.open(str(path), "wb") as wav:
                wav.setnchannels(1)
                wav.setsampwidth(2)  # int16
                wav.setframerate(sample_rate)
                wav.writeframes(pcm_bytes)
        except Exception as exc:  # noqa: BLE001
            logger.warning("Could not write runtime WAV {}: {}", path, exc)


class BufferedMistralTTSService(_BufferedTTSMixin, MistralTTSService):
    """Buffered Mistral Voxtral TTS.

    Mistral's Voxtral TTS streams audio chunks with irregular cadence — we
    routinely see 200–900 ms gaps mid-utterance, which exceeds Pipecat's
    500 ms upstream-jitter buffer and causes audible stutter on WebRTC. The
    mixin pre-renders the whole utterance so the transport emits it at a steady
    cadence (zero mid-stream gaps) and writes WAVs for HTTP ``<audio>`` playback.

    Trade-off: first audio is delayed by the full TTS generation time
    (~1–1.5 s) instead of ~90 ms. For 5–15 s exam turns that "thinking pause"
    reads as natural; mid-sentence jitter does not.
    """

    def __init__(
        self,
        *args,
        session_id: str | None = None,
        output_sample_rate: int | None = None,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self._init_buffer(session_id=session_id, output_sample_rate=output_sample_rate)


class BufferedKokoroTTSService(_BufferedTTSMixin, KokoroTTSService):
    """Buffered local Kokoro TTS — drop-in replacement for the Mistral engine.

    Kokoro is English-pinned (no wrong-language artifacts) and renders ~3x
    faster than real time on CPU, so the per-utterance "thinking pause" is much
    shorter than Mistral's. It reuses the exact same WAV-playback path via the
    mixin, so switching engines needs no frontend change — set
    ``VOICE_RUNTIME_TTS_PROVIDER=kokoro``.

    Follow-up (needs a coordinated frontend change): stream Kokoro over WebRTC
    to remove buffering entirely and drop first-audio latency to ~130 ms.
    """

    def __init__(
        self,
        *args,
        session_id: str | None = None,
        output_sample_rate: int | None = None,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self._init_buffer(session_id=session_id, output_sample_rate=output_sample_rate)


def _resample_s16le_mono(pcm_bytes: bytes, *, from_rate: int, to_rate: int) -> bytes:
    if from_rate == to_rate:
        return pcm_bytes
    samples = np.frombuffer(pcm_bytes, dtype=np.int16).astype(np.float32)
    resampled = soxr.resample(samples, from_rate, to_rate, quality="HQ")
    resampled = np.clip(np.rint(resampled), -32768, 32767).astype(np.int16)
    return resampled.tobytes()


def _pad_s16le_to_10ms(pcm_bytes: bytes, sample_rate: int) -> bytes:
    bytes_per_10ms = (sample_rate // 100) * 2
    remainder = len(pcm_bytes) % bytes_per_10ms
    if remainder == 0:
        return pcm_bytes
    return pcm_bytes + bytes(bytes_per_10ms - remainder)
