"""Student-controlled turns: the Talk button decides when an answer starts and ends."""

import asyncio

from loguru import logger
from pipecat.frames.frames import (
    BotStartedSpeakingFrame,
    CancelFrame,
    EndFrame,
    Frame,
    InputAudioRawFrame,
    InputTransportMessageFrame,
    OutputTransportMessageUrgentFrame,
    UserStartedSpeakingFrame,
    UserStoppedSpeakingFrame,
    VADUserStartedSpeakingFrame,
    VADUserStoppedSpeakingFrame,
)
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor

# Audio still in flight when the student presses stop; cutting at the press clips the last word.
STOP_TAIL_SECS = 0.3
# VAD keeps running on captured audio; mid-answer pauses close a segment, not the turn.
PTT_VAD_STOP_SECS = 0.8
# Ending the turn before the last segment's transcript lands would drop those words.
LAST_FINAL_WAIT_SECS = 1.5
# Long enough for TurnRecovery to resume a reply the press interrupted without any words.
NO_SPEECH_NOTICE_SECS = 3.0


class PushToTalkProcessor(FrameProcessor):
    """Gates the mic to the student's turn and ends the turn on their press.

    Audio outside a turn becomes silence, not dropped frames, so noise can't reach VAD
    or STT while the socket keeps steady timing. VAD still runs on captured audio, so
    the liveness watchdog only arms for real speech and an empty press can't look like
    an STT failure.
    """

    def __init__(self, *, max_hold_secs: float, empty_turn_secs: float):
        super().__init__()
        self._max_hold_secs = max_hold_secs
        self._empty_turn_secs = empty_turn_secs
        self._capturing = False
        self._in_turn = False
        self._speech_seen = False
        self._heard_final = False
        self._closed = False
        self._vad_idle = asyncio.Event()
        self._vad_idle.set()
        self._final_arrived = asyncio.Event()
        self._max_hold_task: asyncio.Task | None = None
        self._finish_task: asyncio.Task | None = None
        self._notice_task: asyncio.Task | None = None

    @property
    def capturing(self) -> bool:
        return self._capturing

    def notify_final_transcript(self) -> None:
        self._heard_final = True
        self._final_arrived.set()
        self._cancel_timer("_notice_task")

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)

        if isinstance(frame, InputTransportMessageFrame) and isinstance(frame.message, dict):
            message_type = frame.message.get("type")
            if message_type == "ptt_start":
                await self._start()
                return
            if message_type == "ptt_stop":
                await self._stop()
                return
        elif isinstance(frame, InputAudioRawFrame):
            if not self._in_turn:
                frame.audio = bytes(len(frame.audio))
        elif isinstance(frame, VADUserStartedSpeakingFrame):
            self._speech_seen = True
            self._vad_idle.clear()
        elif isinstance(frame, VADUserStoppedSpeakingFrame):
            self._final_arrived.clear()
            self._vad_idle.set()
        elif isinstance(frame, BotStartedSpeakingFrame):
            self._cancel_timer("_notice_task")
        elif isinstance(frame, (EndFrame, CancelFrame)):
            self._closed = True
            self._capturing = False
            for timer in ("_max_hold_task", "_finish_task", "_notice_task"):
                self._cancel_timer(timer)

        await self.push_frame(frame, direction)

    async def _start(self) -> None:
        if self._closed or self._capturing:
            return
        self._capturing = True
        self._in_turn = True
        self._cancel_timer("_notice_task")
        if self._finish_task is not None:
            # Pressed again before the turn closed: still the same answer.
            self._cancel_timer("_finish_task")
        else:
            self._speech_seen = False
            self._heard_final = False
            await self.broadcast_interruption()
            await self.push_frame(UserStartedSpeakingFrame())
        await self._emit({"type": "ptt_state", "capturing": True})
        self._max_hold_task = self.create_task(self._max_hold_timer())

    async def _stop(self, *, auto: bool = False) -> None:
        if self._closed or not self._capturing:
            return
        self._capturing = False
        if not auto:
            self._cancel_timer("_max_hold_task")
        await self._emit({"type": "ptt_state", "capturing": False})
        if auto:
            await self._emit({"type": "ptt_auto_stopped"})
        self._finish_task = self.create_task(self._finish_turn())

    async def _finish_turn(self) -> None:
        await asyncio.sleep(STOP_TAIL_SECS)
        self._in_turn = False
        if self._speech_seen:
            await self._wait_for(self._vad_idle, PTT_VAD_STOP_SECS + 1.0)
        else:
            # VAD can miss a quiet voice, so still ask STT for whatever it heard.
            self._final_arrived.clear()
            await self.push_frame(VADUserStoppedSpeakingFrame())
        await self._wait_for(self._final_arrived, LAST_FINAL_WAIT_SECS)
        self._finish_task = None
        await self.push_frame(UserStoppedSpeakingFrame())
        if not self._heard_final:
            wait = self._empty_turn_secs if self._speech_seen else NO_SPEECH_NOTICE_SECS
            self._notice_task = self.create_task(self._no_speech_notice(wait))

    async def _max_hold_timer(self) -> None:
        await asyncio.sleep(self._max_hold_secs)
        self._max_hold_task = None
        logger.info("Push-to-talk turn reached the {}s cap; ending it", self._max_hold_secs)
        await self._stop(auto=True)

    async def _no_speech_notice(self, wait: float) -> None:
        await asyncio.sleep(wait)
        self._notice_task = None
        await self._emit({"type": "ptt_no_speech"})

    async def _emit(self, message: dict) -> None:
        await self.push_frame(OutputTransportMessageUrgentFrame(message=message))

    @staticmethod
    async def _wait_for(event: asyncio.Event, timeout: float) -> None:
        try:
            await asyncio.wait_for(event.wait(), timeout)
        except TimeoutError:
            pass

    def _cancel_timer(self, name: str) -> None:
        task = getattr(self, name)
        if task is not None:
            task.cancel()
            setattr(self, name, None)
