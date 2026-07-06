"""Voice activity detection settings for Pipecat user turns."""

from pipecat.audio.vad.silero import SileroVADAnalyzer
from pipecat.audio.vad.vad_analyzer import VADParams

from runtime.audio_constants import BROWSER_INPUT_SAMPLE_RATE
from runtime.settings import get_settings


settings = get_settings()


# Per-agent eagerness translates to VAD ``stop_secs``: how long of silence
# before treating the student as "done speaking" so the agent can respond.
# Smaller = snappier; larger = more patient (good for thoughtful answers).
#
# Raised across the board: at the old values a natural mid-sentence pause ended
# the turn, so the batch STT only got a fragment ("You work, right? the") — which
# read as both truncation AND poor transcription. Oral-exam answers pause to
# think, so we err toward patience. (A proper fix is semantic turn detection —
# Smart Turn v3 — which needs the streaming-STT turn rework.)
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

