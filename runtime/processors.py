"""Small Pipecat processors used by the voice runtime pipeline."""

from dataclasses import replace
import json
from pathlib import Path

from pipecat.frames.frames import (
    AggregatedTextFrame,
    CancelFrame,
    EndFrame,
    ErrorFrame,
    Frame,
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
        await self._emit({"type": "speaking", "isSpeaking": False})
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
            await self._emit({"type": "interrupt"})
            await self._emit({"type": "listening", "isListening": True})

        elif isinstance(frame, UserStoppedSpeakingFrame):
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

    def __init__(self, recorder: SessionRecorder):
        super().__init__()
        self._recorder = recorder
        self._user_turn_id = 0
        self._active_user_turn_id: int | None = None

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
            text = _clean_transcript_text(frame.text or "")
            if text:
                frame = replace(frame, text=text)
                await self._emit_user_transcript(text, is_final=False)

        elif isinstance(frame, TranscriptionFrame):
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

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)

        if isinstance(frame, InputTransportMessageFrame):
            message = frame.message if isinstance(frame.message, dict) else {}
            message_type = message.get("type")
            if message_type == "end":
                self._recorder.mark_client_ended()
                await self.push_frame(
                    EndFrame(reason=message.get("reason", "client_ended")),
                    direction,
                )
                return
            if message_type == "cancel":
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
