"""Examiner voices an agent can pick; mirrors KOKORO_VOICES in the SB backend."""

from pipecat.transcriptions.language import Language

KOKORO_VOICE_IDS = frozenset({
    "af_heart", "af_bella", "af_nicole", "af_aoede", "af_kore", "af_sarah", "af_alloy", "af_nova",
    "bf_emma", "bf_isabella",
    "am_fenrir", "am_michael", "am_puck",
    "bm_george", "bm_fable",
})


def resolve_kokoro_voice(voice_id: str, default: str) -> str:
    # voice_id can hold a legacy Mistral or ElevenLabs id, which would crash Kokoro.
    return voice_id if voice_id in KOKORO_VOICE_IDS else default


def kokoro_language(voice: str) -> Language:
    return Language.EN_GB if voice.startswith("b") else Language.EN
