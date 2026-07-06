"""Voice activity detection settings for Pipecat user turns."""

from pipecat.audio.vad.silero import SileroVADAnalyzer
from pipecat.audio.vad.vad_analyzer import VADParams

from runtime.audio_constants import BROWSER_INPUT_SAMPLE_RATE
from runtime.settings import get_settings


settings = get_settings()


# Per-agent eagerness translates to VAD ``stop_secs``: how long of silence
# before treating the student as "done speaking" so the agent can respond.
# Smaller = snappier; larger = more patient. Oral-exam answers pause to think,
# so the values err toward patience — at lower values a natural mid-sentence
# pause ended the turn and the STT only saw a fragment. Semantic turn
# detection (e.g. Smart Turn) would be the principled replacement for these
# fixed timeouts if turn-taking needs further tuning.
EAGERNESS_TO_STOP_SECS = {
    "low": 2.0,
    "normal": 1.5,
    "high": 1.0,
}


def _build_vad_analyzer(eagerness: str = "normal") -> SileroVADAnalyzer:
    """Build VAD with stop_secs tuned by the agent's eagerness setting."""
    stop_secs = EAGERNESS_TO_STOP_SECS.get(eagerness, settings.vad_stop_secs)
    return SileroVADAnalyzer(
        sample_rate=BROWSER_INPUT_SAMPLE_RATE,
        params=VADParams(
            confidence=settings.vad_confidence,
            start_secs=settings.vad_start_secs,
            stop_secs=stop_secs,
            min_volume=settings.vad_min_volume,
        ),
    )

