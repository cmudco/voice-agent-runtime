"""Keep accidentally echoed transcript headers out of TTS and saved replies."""
import re
from dataclasses import replace

from pipecat.frames.frames import (
    LLMTextFrame, LLMFullResponseStartFrame, LLMFullResponseEndFrame,
    InterruptionFrame, CancelFrame, EndFrame,
)
from pipecat.processors.frame_processor import FrameProcessor

HEADER = re.compile(r"^\s*\[\d+:\d{2}\]\s*\((?:agent|assistant)\)\s*", re.IGNORECASE)


class SpeechTextGuard(FrameProcessor):
    """Buffers only a possible leading header, including split streaming tokens."""
    def __init__(self):
        super().__init__()
        self._pending = ""
        self._leading = True

    def feed(self, text: str, *, final=False) -> str:
        if not self._leading:
            return text
        self._pending += text
        candidate = self._pending.lstrip()
        if not candidate and not final:
            return ""
        match = HEADER.match(self._pending)
        if match:
            value = self._pending[match.end():]
        elif candidate.startswith("[") and ")" not in candidate and len(candidate) < 64 and not final:
            return ""
        else:
            value = self._pending
        self._pending = ""
        self._leading = False
        return value

    async def process_frame(self, frame, direction):
        await super().process_frame(frame, direction)
        if isinstance(frame, (LLMFullResponseStartFrame, InterruptionFrame, CancelFrame, EndFrame)):
            self._pending = ""
            self._leading = True
        elif isinstance(frame, LLMTextFrame):
            text = self.feed(frame.text)
            if not text:
                return
            frame = replace(frame, text=text)
        elif isinstance(frame, LLMFullResponseEndFrame) and self._pending:
            text = self.feed("", final=True)
            if text.strip():
                await self.push_frame(LLMTextFrame(text), direction)
        await self.push_frame(frame, direction)
