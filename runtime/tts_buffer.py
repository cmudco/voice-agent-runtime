"""Buffered Mistral TTS and runtime WAV artifact writing."""

from pathlib import Path
import wave

from loguru import logger
import numpy as np
import soxr

from pipecat.frames.frames import OutputTransportMessageUrgentFrame, TTSAudioRawFrame
from pipecat.services.mistral.tts import MistralTTSService


RUNTIME_AUDIO_DIR = Path("/tmp/voice-runtime-audio")


class BufferedMistralTTSService(MistralTTSService):
    """Pre-renders the full utterance before yielding any audio.

    Mistral's Voxtral TTS streams audio chunks with irregular cadence — we
    routinely see 200–900 ms gaps mid-utterance, which exceeds Pipecat's
    500 ms upstream-jitter buffer and causes audible stutter on WebRTC.

    This subclass collects every audio chunk for a single ``run_tts`` call,
    concatenates them, and yields one consolidated ``TTSAudioRawFrame``.
    The transport's output pacer then emits the audio at a perfectly steady
    cadence over WebRTC — zero mid-stream gaps possible because nothing
    leaves until the whole utterance is in hand.

    Trade-off: agent's first audio is delayed by the full TTS generation
    time (~1–1.5 s typical) instead of ~90 ms. For a Socratic exam where
    turns are 5–15 s long, that "thinking pause" feels natural; mid-
    sentence jitter does not.

    Reversible: drop this subclass and switch back to ``MistralTTSService``
    the moment Mistral's streaming cadence improves.

    Playback side effect (when ``session_id`` is provided): each pre-rendered
    utterance is also written to ``RUNTIME_AUDIO_DIR / <session_id> /
    utterance-<n>.wav`` so the frontend can fetch and play the stable source
    PCM through a plain ``<audio>`` element.
    """

    def __init__(
        self,
        *args,
        session_id: str | None = None,
        output_sample_rate: int | None = None,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
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
            # Pass non-audio frames (errors, etc.) through immediately
            # so the pipeline can react to them.
            yield frame
        if parts:
            joined = b"".join(parts)
            output_audio = _resample_s16le_mono(
                joined,
                from_rate=sample_rate,
                to_rate=self._output_sample_rate,
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
                output_audio,
                self._output_sample_rate,
                1,
                context_id=context_id,
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
