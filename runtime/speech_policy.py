"""Pure speech-progress policy; no provider, transport, or wall-clock dependencies."""

from dataclasses import dataclass


@dataclass(frozen=True)
class SpeechRecoveryPolicy:
    progress_timeout: float = 12.0
    minimum_speech: float = 2.0
    reconnect_timeout: float = 12.0
    reconnect_attempts: int = 1

    def __post_init__(self):
        if min(self.progress_timeout, self.minimum_speech, self.reconnect_timeout) <= 0:
            raise ValueError("Speech recovery deadlines must be positive")
        if self.reconnect_attempts != 1:
            raise ValueError("Exactly one reconnect is supported before backup")


class SpeechProgress:
    def __init__(self, policy: SpeechRecoveryPolicy):
        self.policy = policy
        self.last_progress: float | None = None
        self.last_audio: float | None = None
        self.speech_seconds = 0.0
        self.last_text = ""

    def begin(self, now: float, *, fresh: bool = False) -> None:
        if fresh or self.last_progress is None:
            self.last_progress = now
            self.speech_seconds = 0.0
            self.last_text = ""

    def audio(self, now: float, seconds: float, speaking: bool) -> None:
        self.last_audio = now
        if speaking:
            self.speech_seconds += seconds

    def transcript(self, now: float, text: str) -> bool:
        # Repeated identical interim results are not new progress.
        if text.strip() and text != self.last_text:
            self.last_text = text
            self.last_progress = now
            self.speech_seconds = 0.0
            return True
        return False

    def stalled(self, now: float) -> bool:
        return (
            self.last_progress is not None and self.last_audio is not None
            and now - self.last_progress >= self.policy.progress_timeout
            and now - self.last_audio < 2.0
            and self.speech_seconds >= self.policy.minimum_speech
        )
