"""Small Pipecat processors used by the voice runtime pipeline."""

from dataclasses import replace
import json
import re
import time
from pathlib import Path

import numpy as np
from pipecat.audio.utils import calculate_audio_volume
from pipecat.frames.frames import (
    AggregatedTextFrame,
    BotStartedSpeakingFrame,
    BotStoppedSpeakingFrame,
    CancelFrame,
    EndFrame,
    ErrorFrame,
    Frame,
    InputAudioRawFrame,
    InputTransportMessageFrame,
    InterruptionFrame,
    InterimTranscriptionFrame,
    LLMContextFrame,
    LLMFullResponseEndFrame,
    LLMFullResponseStartFrame,
    OutputAudioRawFrame,
    OutputTransportMessageUrgentFrame,
    TranscriptionFrame,
    TTSStartedFrame,
    TTSStoppedFrame,
    TTSTextFrame,
    UserStartedSpeakingFrame,
    UserStoppedSpeakingFrame,
)
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor
from loguru import logger

from runtime.recorder import SessionRecorder
from runtime.text_utils import _clean_transcript_text, _normalize_transcript_text
from runtime.sentry import capture_browser_error


RUNTIME_DEBUG_DIR = Path("/tmp/voice-runtime-audio")


class AgentTurnCoordinator:
    """Keeps browser transcript turn IDs stable from TTS preview through final."""

    def __init__(self):
        self._current_turn_id = 0
        self._preview_turn_id: int | None = None

    def start_preview_turn(self) -> int:
        self._current_turn_id += 1
        self._preview_turn_id = self._current_turn_id
        return self._current_turn_id

    def start_audio_turn(self) -> int:
        if self._preview_turn_id is not None:
            turn_id = self._preview_turn_id
            self._preview_turn_id = None
            return turn_id
        self._current_turn_id += 1
        return self._current_turn_id

    def clear_preview_turn(self) -> None:
        self._preview_turn_id = None


class AudioInputDebugProcessor(FrameProcessor):
    """Logs inbound mic-audio levels so dead-mic sessions are diagnosable.

    "Voice not picked up" sessions have two very different causes the logs
    could not previously distinguish: the browser delivering a silent audio
    track (mic/permission/unmute problem on the client) versus real speech
    arriving below the Silero VAD volume gate (``min_volume`` tuning problem
    in the runtime). This taps the raw audio straight off the input transport
    and periodically logs peak/RMS plus the same EBU-R128 volume measure the
    VAD gate compares against ``min_volume``. Levels only — no speech content
    is captured. Reading a session's lines:

      no "inbound audio" lines at all  -> no mic frames reached the runtime
      peak~0.000, vad_volume max ~0    -> track connected but digitally silent
      vad_volume max < min_volume      -> speech present but gated by VAD
      vad_volume max >= min_volume     -> input healthy; look downstream (STT)
    """

    SUMMARY_INTERVAL_SECS = 5.0
    # VAD-comparable volume is computed over sub-chunks about this long, so a
    # short utterance inside an otherwise quiet window still shows its true
    # loudness instead of being averaged away.
    VOLUME_CHUNK_SECS = 0.25

    def __init__(self, session_id: str, *, min_volume: float, confidence: float):
        super().__init__()
        self._session_id = session_id
        self._min_volume = min_volume
        self._confidence = confidence
        self._started_at = time.monotonic()
        self._first_frame_logged = False
        self._window_frames = 0
        self._window_audio = bytearray()
        self._sample_rate = 0

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)
        if isinstance(frame, InputAudioRawFrame):
            try:
                self._observe(frame)
            except Exception as exc:  # noqa: BLE001 - telemetry must not break the call
                logger.warning(
                    "Pipecat session {} inbound audio debug failed: {}",
                    self._session_id,
                    exc,
                )
        await self.push_frame(frame, direction)

    def _observe(self, frame: InputAudioRawFrame) -> None:
        self._sample_rate = frame.sample_rate or self._sample_rate
        if not self._first_frame_logged:
            self._first_frame_logged = True
            logger.info(
                "Pipecat session {} inbound mic audio flowing: first frame {:.2f}s "
                "after pipeline start (sample_rate={} channels={})",
                self._session_id,
                time.monotonic() - self._started_at,
                frame.sample_rate,
                frame.num_channels,
            )
        self._window_frames += 1
        self._window_audio.extend(frame.audio)
        # Windows are measured in audio time, not wall-clock: every summary
        # line covers the same amount of actual microphone signal.
        window_secs = len(self._window_audio) / 2 / self._sample_rate
        if window_secs >= self.SUMMARY_INTERVAL_SECS:
            self._log_window(window_secs)
            self._window_frames = 0
            self._window_audio = bytearray()

    def _log_window(self, elapsed: float) -> None:
        samples = np.frombuffer(bytes(self._window_audio), dtype=np.int16)
        if samples.size == 0 or not self._sample_rate:
            return
        peak = float(np.abs(samples.astype(np.int32)).max()) / 32768.0
        rms = float(np.sqrt(np.mean(np.square(samples.astype(np.float64))))) / 32768.0
        chunk_bytes = max(2, int(self._sample_rate * self.VOLUME_CHUNK_SECS) * 2)
        volumes = [
            calculate_audio_volume(chunk, self._sample_rate)
            for offset in range(0, len(self._window_audio), chunk_bytes)
            if len(chunk := bytes(self._window_audio[offset : offset + chunk_bytes]))
            >= chunk_bytes
        ]
        vol_max = max(volumes, default=0.0)
        above_gate = sum(1 for volume in volumes if volume >= self._min_volume)
        logger.info(
            "Pipecat session {} inbound audio last {:.1f}s: frames={} peak={:.3f} "
            "rms={:.4f} vad_volume max={:.2f} chunks_above_gate={}/{} "
            "(gate: min_volume={} confidence={})",
            self._session_id,
            elapsed,
            self._window_frames,
            peak,
            rms,
            vol_max,
            above_gate,
            len(volumes),
            self._min_volume,
            self._confidence,
        )


class BrowserEventProcessor(FrameProcessor):
    """Repackages pipeline events into browser messages.

    Pipecat's output transport only hands audio, ``InterruptionFrame`` and
    ``OutputTransportMessage`` frames to the serializer. Speaking/listening
    state and transcripts are plain frames that would otherwise never reach
    the browser, so this processor (placed right before ``transport.output()``)
    mirrors them as ``OutputTransportMessageUrgentFrame`` and records the
    transcript for the Django callback. Every original frame is forwarded
    untouched.
    """

    def __init__(
        self,
        recorder: SessionRecorder,
        turn_coordinator: AgentTurnCoordinator,
    ):
        super().__init__()
        self._recorder = recorder
        self._turn_coordinator = turn_coordinator
        self._agent_parts: list[str] = []
        self._agent_preview_parts: list[str] = []
        self._agent_preview_context_id: str | None = None
        self._agent_context_id: str | None = None
        self._agent_turn_id = 0
        self._agent_audio_started = False
        self._agent_expects_llm_end = False

    async def _emit(self, message: dict) -> None:
        await self.push_frame(
            OutputTransportMessageUrgentFrame(message=message),
            FrameDirection.DOWNSTREAM,
        )

    async def _emit_agent_preview(self, parts: list[str]) -> None:
        agent_text = _normalize_transcript_text(parts)
        if not agent_text:
            return
        logger.info(
            "[transcript] PREVIEW turn={} ctx={} text={!r}",
            self._agent_turn_id,
            self._agent_preview_context_id,
            agent_text[:90],
        )
        await self._emit(
            {
                "type": "transcript_update",
                "role": "agent",
                "message": agent_text,
                "content": agent_text,
                "turnId": self._agent_turn_id,
                "isFinal": False,
            }
        )

    async def _finalize_agent_turn(self) -> None:
        agent_text = _normalize_transcript_text(self._agent_parts)
        if agent_text:
            logger.info(
                "Pipecat session {} agent transcript FINAL turn={} chars={} text={!r}",
                self._recorder.session_id,
                self._agent_turn_id,
                len(agent_text),
                agent_text[:90],
            )
            entry = self._recorder.add_transcript("agent", agent_text)
            await self._emit(
                {
                    "type": "transcript_update",
                    "role": "agent",
                    "message": entry["message"],
                    "content": entry["message"],
                    "timeInCallSecs": entry["timeInCallSecs"],
                    "turnId": self._agent_turn_id,
                    "isFinal": True,
                }
            )
        # Text generation ending is not audio playback ending. The output
        # transport reports BotStoppedSpeakingFrame after its audio drains.
        self._agent_parts = []
        self._agent_preview_parts = []
        self._agent_preview_context_id = None
        self._agent_context_id = None
        self._agent_audio_started = False
        self._agent_expects_llm_end = False

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)

        if isinstance(frame, LLMFullResponseStartFrame):
            self._agent_expects_llm_end = True

        elif isinstance(frame, BotStartedSpeakingFrame):
            await self._emit({"type": "speaking", "isSpeaking": True})

        elif isinstance(frame, BotStoppedSpeakingFrame):
            await self._emit({"type": "speaking", "isSpeaking": False})
            self._recorder.health_event("agent_output_stopped")

        elif isinstance(frame, LLMFullResponseEndFrame):
            await self._finalize_agent_turn()

        elif isinstance(frame, InterruptionFrame):
            # The browser needs an explicit data-channel interrupt signal on
            # barge-in: it clears the pending-reply UI state, and on the HTTP
            # WAV fallback transport it also stops queued/playing utterances
            # (the WebRTC audio track is cut by the transport itself).
            await self._emit({"type": "interrupt"})
            await self._emit({"type": "speaking", "isSpeaking": False})
            await self._emit({"type": "processing", "isProcessing": False})
            self._agent_audio_started = False
            self._agent_parts.clear()
            self._agent_preview_parts.clear()
            self._agent_preview_context_id = None
            self._agent_context_id = None
            self._agent_expects_llm_end = False
            self._turn_coordinator.clear_preview_turn()

        elif isinstance(frame, UserStartedSpeakingFrame):
            logger.info(
                "Pipecat session {} VAD user_started_speaking",
                self._recorder.session_id,
            )
            await self._emit({"type": "interrupt"})
            await self._emit({"type": "listening", "isListening": True})

        elif isinstance(frame, UserStoppedSpeakingFrame):
            logger.info(
                "Pipecat session {} VAD user_stopped_speaking",
                self._recorder.session_id,
            )
            await self._emit({"type": "listening", "isListening": False})

        elif isinstance(frame, TTSStartedFrame):
            is_new_context = frame.context_id != self._agent_context_id
            if is_new_context:
                self._agent_context_id = frame.context_id
                self._agent_turn_id = self._turn_coordinator.start_audio_turn()
                self._agent_parts = []
                self._agent_audio_started = False
                self._agent_preview_parts = []
                self._agent_preview_context_id = frame.context_id
            logger.info(
                "Pipecat session {} TTS_STARTED turn={} ctx={} new_context={}",
                self._recorder.session_id,
                self._agent_turn_id,
                frame.context_id,
                is_new_context,
            )

        elif isinstance(frame, OutputAudioRawFrame) and not self._agent_audio_started:
            self._agent_audio_started = True
            self._recorder.mark_agent_audio_queued()
            logger.info(
                "Pipecat session {} AGENT_AUDIO_STARTED turn={}",
                self._recorder.session_id,
                self._agent_turn_id,
            )
            await self._emit_agent_preview(
                self._agent_parts if self._agent_parts else self._agent_preview_parts
            )
            self._recorder.mark_runtime_started()
            await self._emit({"type": "processing", "isProcessing": False})
            await self._emit({"type": "speaking", "isSpeaking": True})
            await self._emit({"type": "agent_audio_started", "turnId": self._agent_turn_id})

        elif isinstance(frame, TTSTextFrame):
            if self._agent_turn_id == 0:
                self._agent_turn_id = self._turn_coordinator.start_audio_turn()
            self._agent_parts.append(frame.text)
            if self._agent_audio_started:
                await self._emit_agent_preview(self._agent_parts)

        elif isinstance(frame, AggregatedTextFrame):
            if frame.context_id != self._agent_preview_context_id:
                self._agent_preview_parts = []
                self._agent_preview_context_id = frame.context_id
                self._turn_coordinator.start_preview_turn()
            self._agent_preview_parts.append(frame.text)

        elif isinstance(frame, TTSStoppedFrame):
            if not self._agent_expects_llm_end:
                await self._finalize_agent_turn()

        elif isinstance(frame, ErrorFrame):
            capture_browser_error(frame.error, fatal=frame.fatal)
            await self._emit(
                {
                    "type": "error",
                    "error": frame.error,
                    "isFatal": frame.fatal,
                }
            )

        # Note: user TranscriptionFrame is handled by UserTranscriptProcessor
        # (placed right after STT) because the user-context aggregator
        # consumes it before it could ever reach this processor.

        await self.push_frame(frame, direction)


class UserTranscriptProcessor(FrameProcessor):
    """Mirrors the student's final transcription to the browser.

    Placed immediately after STT and before the user-context aggregator,
    because that aggregator consumes ``TranscriptionFrame`` to build LLM
    context and does not forward it downstream - so it would otherwise
    never reach ``BrowserEventProcessor`` and the student's words would be
    missing from the live transcript. The emitted message frame travels
    downstream to the output transport and is serialized there.
    """

    def __init__(self, recorder: SessionRecorder, liveness_watchdog=None):
        super().__init__()
        self._recorder = recorder
        # SttLivenessWatchdog sits upstream of STT so it never sees the
        # interim/final frames itself; this processor reports them to it.
        self._liveness_watchdog = liveness_watchdog
        self._user_turn_id = 0
        self._active_user_turn_id: int | None = None
        self._interim_logged_turn_id: int | None = None

    def _ensure_user_turn_id(self) -> int:
        if self._active_user_turn_id is None:
            self._user_turn_id += 1
            self._active_user_turn_id = self._user_turn_id
        return self._active_user_turn_id

    async def _emit_user_transcript(self, text: str, *, is_final: bool) -> None:
        normalized = _normalize_transcript_text([text])
        if not normalized:
            return
        turn_id = self._ensure_user_turn_id()
        if not is_final and turn_id != self._interim_logged_turn_id:
            # First interim of the turn proves STT is hearing the student even
            # when the turn later fails to finalize into a transcript.
            self._interim_logged_turn_id = turn_id
            logger.info(
                "Pipecat session {} user transcript first interim turn={} text={!r}",
                self._recorder.session_id,
                turn_id,
                normalized[:60],
            )
        entry = (
            self._recorder.add_transcript("user", normalized)
            if is_final
            else {"message": normalized, "timeInCallSecs": self._recorder.duration_seconds()}
        )
        await self.push_frame(
            OutputTransportMessageUrgentFrame(
                message={
                    "type": "transcript_update",
                    "role": "user",
                    "message": entry["message"],
                    "content": entry["message"],
                    "timeInCallSecs": entry["timeInCallSecs"],
                    "turnId": turn_id,
                    "isFinal": is_final,
                }
            ),
            FrameDirection.DOWNSTREAM,
        )
        if is_final:
            logger.info(
                "Pipecat session {} user transcript FINAL turn={} chars={} text={!r}",
                self._recorder.session_id,
                turn_id,
                len(normalized),
                normalized[:90],
            )
            await self.push_frame(
                OutputTransportMessageUrgentFrame(
                    message={"type": "processing", "isProcessing": True}
                ),
                FrameDirection.DOWNSTREAM,
            )
            self._active_user_turn_id = None

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)

        if isinstance(frame, UserStartedSpeakingFrame):
            self._ensure_user_turn_id()

        elif isinstance(frame, InterimTranscriptionFrame):
            # Any STT response — even empty — proves the socket is alive, so
            # report it before the text checks (see SttLivenessWatchdog).
            if self._liveness_watchdog is not None:
                self._liveness_watchdog.notify_transcript(is_final=False)
            text = _clean_transcript_text(frame.text or "")
            if text:
                frame = replace(frame, text=text)
                await self._emit_user_transcript(text, is_final=False)

        elif isinstance(frame, TranscriptionFrame):
            if self._liveness_watchdog is not None:
                self._liveness_watchdog.notify_transcript(is_final=True)
            text = _clean_transcript_text(frame.text or "")
            if text:
                language = getattr(frame, "language", None)
                if language:
                    logger.info(
                        "Pipecat session {} STT detected_language={} final_text_preview={!r}",
                        self._recorder.session_id,
                        language,
                        text[:90],
                    )
                frame = replace(frame, text=text)
                await self._emit_user_transcript(text, is_final=True)

        await self.push_frame(frame, direction)


class ClientControlProcessor(FrameProcessor):
    """Handles control messages sent over WebRTC data channels."""

    def __init__(self, recorder: SessionRecorder):
        super().__init__()
        self._recorder = recorder
        self._client_ended = False

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)

        # Once the client has ended the session, stop forwarding mic audio:
        # the transport keeps delivering frames until the peer connection
        # actually closes, and downstream services (STT) would treat that
        # dead-air as a live session.
        if self._client_ended and isinstance(frame, InputAudioRawFrame):
            return

        if isinstance(frame, InputTransportMessageFrame):
            message = frame.message if isinstance(frame.message, dict) else {}
            message_type = message.get("type")
            if message_type == "client_exit":
                self._recorder.record_exit(message.get("reason", "client_cancelled"))
                return
            if message_type == "end":
                self._client_ended = True
                self._recorder.record_exit(message.get("reason", "client_ended"), completed=True)
                await self.push_frame(OutputTransportMessageUrgentFrame(message={
                    "type": "session_end_ack", "requestId": message.get("requestId"),
                }), FrameDirection.DOWNSTREAM)
                await self.push_frame(
                    EndFrame(reason=message.get("reason", "client_ended")),
                    direction,
                )
                return
            if message_type == "cancel":
                self._client_ended = True
                self._recorder.record_exit(message.get("reason", "client_cancelled"))
                await self.push_frame(OutputTransportMessageUrgentFrame(message={
                    "type": "session_end_ack", "requestId": message.get("requestId"),
                }), FrameDirection.DOWNSTREAM)
                await self.push_frame(
                    CancelFrame(reason=message.get("reason", "client_cancelled")),
                    direction,
                )
                return

        await self.push_frame(frame, direction)


class LLMInputDebugProcessor(FrameProcessor):
    """Writes the exact LLM chat context sent to the examiner model.

    Debug aid only — dumps every request's messages (including student
    answers) as JSON under ``/tmp/voice-runtime-audio/<session>/``. Only wired
    into the pipeline when ``VOICE_RUNTIME_LLM_INPUT_DEBUG`` is enabled; keep
    it off in production.
    """

    def __init__(self, session_id: str):
        super().__init__()
        self._session_id = session_id
        self._request_index = 0
        self._debug_dir = RUNTIME_DEBUG_DIR / session_id
        try:
            self._debug_dir.mkdir(parents=True, exist_ok=True)
        except OSError as exc:
            logger.warning("Could not create LLM debug dir for {}: {}", session_id, exc)

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)

        if isinstance(frame, LLMContextFrame):
            self._request_index += 1
            messages = frame.context.get_messages()
            path = self._debug_dir / f"llm-input-{self._request_index:03d}.json"
            try:
                payload = {
                    "sessionId": self._session_id,
                    "requestIndex": self._request_index,
                    "messageCount": len(messages),
                    "messages": messages,
                }
                path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=str))
                tail = [
                    {
                        "role": message.get("role"),
                        "content": _preview_message_content(message.get("content")),
                    }
                    for message in messages[-5:]
                ]
                logger.info(
                    "Pipecat session {} LLM input #{} wrote {} messages to {}; tail={}",
                    self._session_id,
                    self._request_index,
                    len(messages),
                    path,
                    tail,
                )
            except Exception as exc:  # noqa: BLE001 - debug logging must not break the call
                logger.warning(
                    "Could not write LLM input debug for session {} request #{}: {}",
                    self._session_id,
                    self._request_index,
                    exc,
                )

        await self.push_frame(frame, direction)


def _preview_message_content(content) -> str:
    if isinstance(content, str):
        text = content
    else:
        text = json.dumps(content, ensure_ascii=False, default=str)
    text = " ".join(text.split())
    if len(text) <= 180:
        return text
    return f"{text[:177]}..."


ELAPSED_STAMP_RE = re.compile(r"^\[\d+:\d{2}\] \((?:user|agent)\) ")


def format_elapsed(seconds: int) -> str:
    return f"{seconds // 60}:{seconds % 60:02d}"


class ElapsedTimestampProcessor(FrameProcessor):
    """Prefix every context message with the elapsed exam time before each LLM call.

    The aggregators store raw text, so the brain had no idea how long the exam
    had been running and guessed. User messages are stamped with the time of the
    request they trigger (the moment the student stopped talking, matching the
    transcript's timeInCallSecs). Agent messages enter the context only after
    they were spoken, so they are stamped lazily on the next request with the
    time the request that produced them was made.
    """

    def __init__(self, recorder: SessionRecorder):
        super().__init__()
        self._recorder = recorder
        self._agent_stamp = 0

    def _stamp(self, messages: list[dict]) -> list[dict]:
        now = self._recorder.duration_seconds()
        for message in messages:
            role = message.get("role")
            content = message.get("content")
            if role not in ("user", "assistant") or not isinstance(content, str):
                continue
            if ELAPSED_STAMP_RE.match(content):
                continue
            label = "user" if role == "user" else "agent"
            secs = now if role == "user" else self._agent_stamp
            message["content"] = f"[{format_elapsed(secs)}] ({label}) {content}"
        self._agent_stamp = now
        return messages

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)
        if isinstance(frame, LLMContextFrame):
            frame.context.transform_messages(self._stamp)
        await self.push_frame(frame, direction)
