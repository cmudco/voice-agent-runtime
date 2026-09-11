"""Content-free user-visible stage events, shared across pipeline taps."""

import asyncio
from enum import StrEnum

from pipecat.frames.frames import (
    BotStartedSpeakingFrame, BotStoppedSpeakingFrame, CancelFrame, EndFrame,
    InterruptionFrame, LLMContextFrame, LLMFullResponseEndFrame,
    LLMTextFrame, TTSSpeakFrame, VADUserStartedSpeakingFrame,
    VADUserStoppedSpeakingFrame,
)
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor


class VoiceStage(StrEnum):
    LISTENING = "listening"
    TRANSCRIBING = "transcribing"
    THINKING = "thinking"
    PREPARING_VOICE = "preparing_voice"
    SPEAKING = "speaking"
    RECOVERING = "recovering"


class VoiceStageTracker:
    def __init__(self, emit):
        self.emit = emit
        self.stage = None
        self.sequence = 0
        self.speaking = False
        self.recovering = False
        self.response_pending = False
        self.closed = False
        self._lock = asyncio.Lock()

    async def set(self, stage: VoiceStage):
        async with self._lock:
            if self.closed:
                return
            if self.recovering:
                stage = VoiceStage.RECOVERING
            elif self.speaking:
                stage = VoiceStage.SPEAKING
            if stage == self.stage:
                return
            self.stage = stage
            self.sequence += 1
            await self.emit({"type": "voice_stage", "voiceStage": stage.value,
                             "sequence": self.sequence})

    async def recognition_event(self, event: str):
        if event in {"stalled", "fallback_started"}:
            self.recovering = True
            await self.set(VoiceStage.RECOVERING)
        elif event == "fallback_processing":
            # Once the backup is the provider, transcribing a turn is the
            # normal path, not a recovery in progress.
            self.recovering = False
            await self.set(VoiceStage.TRANSCRIBING)
        elif event in {"recovered", "fallback_recovered", "fallback_failed", "fallback_reverted", "failed"}:
            self.recovering = False
            await self.set(VoiceStage.THINKING if self.response_pending else VoiceStage.LISTENING)


class VoiceStageTap(FrameProcessor):
    """Input sees upstream VAD; response sees text and transport playback events."""

    def __init__(self, tracker: VoiceStageTracker, *, input_side=False):
        super().__init__()
        self.tracker = tracker
        self.input_side = input_side

    async def process_frame(self, frame, direction):
        await super().process_frame(frame, direction)
        state = self.tracker
        if isinstance(frame, (EndFrame, CancelFrame)):
            state.closed = True
        elif self.input_side:
            if isinstance(frame, VADUserStartedSpeakingFrame):
                state.speaking = False
                await state.set(VoiceStage.LISTENING)
            elif isinstance(frame, VADUserStoppedSpeakingFrame):
                await state.set(VoiceStage.TRANSCRIBING)
        elif isinstance(frame, LLMContextFrame) and direction == FrameDirection.DOWNSTREAM:
            state.response_pending = True
            await state.set(VoiceStage.THINKING)
        elif isinstance(frame, (LLMTextFrame, TTSSpeakFrame)) and direction == FrameDirection.DOWNSTREAM:
            await state.set(VoiceStage.PREPARING_VOICE)
        elif isinstance(frame, LLMFullResponseEndFrame):
            state.response_pending = False
        elif isinstance(frame, BotStartedSpeakingFrame):
            state.speaking = True
            await state.set(VoiceStage.SPEAKING)
        elif isinstance(frame, BotStoppedSpeakingFrame):
            state.speaking = False
            await state.set(VoiceStage.PREPARING_VOICE if state.response_pending else VoiceStage.LISTENING)
        elif isinstance(frame, InterruptionFrame):
            state.speaking = state.response_pending = False
            await state.set(VoiceStage.LISTENING)
        await self.push_frame(frame, direction)
