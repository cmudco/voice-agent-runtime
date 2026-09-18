"""Bounded recovery of a reply interrupted without any recognized user words.

No audio or transcript is persisted here. A retry regenerates the interrupted
reply from its original context; it may repeat part of a sentence, rather than
pretending we know exactly which samples the browser heard.
"""

import asyncio
from copy import deepcopy

from pipecat.frames.frames import (
    BotStartedSpeakingFrame, BotStoppedSpeakingFrame, CancelFrame, EndFrame,
    InterruptionFrame, InterimTranscriptionFrame, LLMContextFrame,
    OutputTransportMessageUrgentFrame, TranscriptionFrame, TTSSpeakFrame,
    UserStartedSpeakingFrame, UserStoppedSpeakingFrame,
)
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor
from pipecat.processors.aggregators.llm_context import LLMContext


class TurnRecovery:
    """Shared by input/context taps. One automatic retry per original reply."""

    def __init__(self, recorder, silence_seconds=2.5):
        self.recorder = recorder
        self.silence_seconds = silence_seconds
        self.pending = False
        self.interrupted = False
        self.speaking = False
        self.heard_words = False
        self.retried = False
        self.closed = False
        self.messages = None
        self.greeting = None
        self.task = None
        self.send = None

    def cancel_timer(self):
        if self.task is not None:
            self.task.cancel()
            self.task = None

    def request(self, context):
        self.cancel_timer()
        self.messages = deepcopy(context.get_messages())
        self.greeting = None
        self.pending = True
        self.interrupted = self.heard_words = self.retried = False

    def words(self):
        self.heard_words = True
        self.interrupted = False
        self.cancel_timer()

    def interrupt(self):
        if self.closed or not self.pending or self.heard_words or self.retried:
            return
        if not self.interrupted:
            self.recorder.health_event("empty_interruption_candidate")
        self.interrupted = True
        self.schedule()

    def schedule(self):
        self.cancel_timer()
        if self.interrupted and not self.speaking and not self.closed:
            self.task = asyncio.create_task(self.resume())

    async def resume(self):
        try:
            await asyncio.sleep(self.silence_seconds)
            if self.closed or self.speaking or self.heard_words or not self.interrupted:
                return
            self.retried = True
            self.interrupted = False
            self.recorder.health_event("empty_interruption_retry")
            await self.send(OutputTransportMessageUrgentFrame(message={
                "type": "turn_recovery", "state": "resuming",
            }), FrameDirection.DOWNSTREAM)
            if self.greeting is not None:
                frame = TTSSpeakFrame(self.greeting)
            elif self.messages is not None:
                frame = LLMContextFrame(LLMContext(messages=deepcopy(self.messages)))
            else:
                return
            await self.send(frame, FrameDirection.DOWNSTREAM)
        except asyncio.CancelledError:
            pass

    def close(self):
        self.closed = True
        self.cancel_timer()


class RecoveryInputProcessor(FrameProcessor):
    def __init__(self, recovery):
        super().__init__()
        self.recovery = recovery

    async def process_frame(self, frame, direction):
        await super().process_frame(frame, direction)
        r = self.recovery
        if isinstance(frame, (InterimTranscriptionFrame, TranscriptionFrame)):
            # A single real word (including "stop") is sufficient. Do not
            # guess whether the user's words are relevant to the assignment.
            if any(char.isalnum() for char in (frame.text or "")):
                r.words()
        elif isinstance(frame, UserStartedSpeakingFrame):
            r.speaking = True
            r.cancel_timer()
        elif isinstance(frame, UserStoppedSpeakingFrame):
            r.speaking = False
            r.schedule()
        elif isinstance(frame, (EndFrame, CancelFrame)):
            r.close()
        await self.push_frame(frame, direction)


class RecoveryContextProcessor(FrameProcessor):
    def __init__(self, recovery):
        super().__init__()
        self.recovery = recovery
        recovery.send = self.push_frame

    async def process_frame(self, frame, direction):
        await super().process_frame(frame, direction)
        r = self.recovery
        if isinstance(frame, LLMContextFrame) and direction == FrameDirection.DOWNSTREAM:
            r.request(frame.context)
        elif isinstance(frame, TTSSpeakFrame) and direction == FrameDirection.DOWNSTREAM:
            r.greeting = frame.text
            r.pending = True
            r.heard_words = r.retried = False
        elif isinstance(frame, InterruptionFrame):
            r.interrupt()
            if r.interrupted:
                await self.push_frame(OutputTransportMessageUrgentFrame(message={
                    "type": "turn_recovery", "state": "paused",
                }), FrameDirection.DOWNSTREAM)
        elif isinstance(frame, BotStartedSpeakingFrame):
            r.pending = True
        elif isinstance(frame, BotStoppedSpeakingFrame) and not r.interrupted:
            r.pending = False
        elif isinstance(frame, (EndFrame, CancelFrame)):
            r.close()
        await self.push_frame(frame, direction)
