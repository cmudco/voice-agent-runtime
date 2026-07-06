import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

RUNTIME_DIR = Path(__file__).resolve().parents[1]

# This repository is deployed as an independent runtime service. Keep the
# runtime .env local to this repo instead of depending on the Django backend
# checkout.
load_dotenv(RUNTIME_DIR / ".env")


@dataclass(frozen=True)
class RuntimeSettings:
    mistral_api_key: str
    socratic_bots_backend_urls: list[str]
    dare_internal_key: str
    voice_runtime_session_secret: str
    mistral_llm_model: str
    mistral_llm_max_tokens: int
    mistral_stt_model: str
    mistral_stt_target_delay_ms: int
    mistral_tts_model: str
    mistral_tts_voice: str
    # "mistral" (Voxtral TTS, current default) or "kokoro" (local, English-pinned,
    # ~3x-realtime — kills the wrong-language artifacts and shortens the
    # per-sentence pause). Kokoro reuses the same WAV-playback path, so switching
    # needs no frontend change; flip VOICE_RUNTIME_TTS_PROVIDER=kokoro to test.
    tts_provider: str
    kokoro_voice: str
    kokoro_model_path: str
    kokoro_voices_path: str
    # "http" (default): pre-render each utterance, play WAVs over HTTP. Audio and
    # the transcript travel on separate paths, so they can drift out of sync.
    # "webrtc": stream Kokoro audio over the WebRTC transport in-order with the
    # transcript events — fixes A/V sync and removes the buffering pause. Only
    # valid with tts_provider=kokoro (local, no network jitter that broke the
    # old Mistral-over-WebRTC path).
    tts_transport: str
    vad_confidence: float
    vad_start_secs: float
    vad_stop_secs: float
    vad_min_volume: float
    user_turn_stop_timeout: float
    voice_runtime_ice_servers: list[str]
    voice_runtime_cors_origins: list[str]
    log_tts_request_text: bool


# Origins that must always be allowed, regardless of deployment env config.
# These are the Socratic Books frontends that embed the voice runtime; baking
# them in keeps CORS working even if a deployment forgets to set (or sets an
# incomplete) VOICE_RUNTIME_CORS_ORIGINS. Local Vite ports are included for dev.
_ALWAYS_ALLOWED_CORS_ORIGINS = [
    "http://localhost:5173",
    "http://127.0.0.1:5173",
    "http://localhost:5174",
    "http://127.0.0.1:5174",
    "https://socraticbots-dev.hss.cmu.edu",
    "https://socraticbots.hss.cmu.edu",
]


def _resolve_backend_urls() -> list[str]:
    """Resolve the Socratic Books backend URLs the runtime can talk to.

    A single runtime instance serves more than one environment (e.g. staging
    and prod), so ``SOCRATIC_BOTS_BACKEND_URLS`` accepts a comma-separated list.
    The legacy singular ``SOCRATIC_BOTS_BACKEND_URL`` is still honored as a
    fallback so existing deployments keep working. Each browser session belongs
    to exactly one backend; the runtime discovers which by probing these URLs in
    order for the session context (see ``fetch_session_context``).
    """
    raw = os.environ.get("SOCRATIC_BOTS_BACKEND_URLS") or os.environ.get(
        "SOCRATIC_BOTS_BACKEND_URL", "http://localhost:8001"
    )
    urls: list[str] = []
    for url in raw.split(","):
        cleaned = url.strip().rstrip("/")
        if cleaned and cleaned not in urls:
            urls.append(cleaned)
    return urls


def _resolve_cors_origins() -> list[str]:
    """Merge the env-provided CORS origins with the always-allowed defaults.

    Order is preserved and duplicates are dropped so the runtime always accepts
    the Socratic Books frontends without requiring perfect env configuration.
    """
    env_origins = [
        origin.strip()
        for origin in os.environ.get("VOICE_RUNTIME_CORS_ORIGINS", "").split(",")
        if origin.strip()
    ]
    resolved: list[str] = []
    for origin in [*_ALWAYS_ALLOWED_CORS_ORIGINS, *env_origins]:
        if origin not in resolved:
            resolved.append(origin)
    return resolved


def get_settings() -> RuntimeSettings:
    internal_key = os.environ.get("DARE_INTERNAL_KEY", "")
    return RuntimeSettings(
        mistral_api_key=os.environ.get("MISTRAL_API_KEY", ""),
        socratic_bots_backend_urls=_resolve_backend_urls(),
        dare_internal_key=internal_key,
        voice_runtime_session_secret=os.environ.get("VOICE_RUNTIME_SESSION_SECRET")
        or internal_key,
        mistral_llm_model=os.environ.get("MISTRAL_LLM_MODEL", "mistral-large-latest"),
        mistral_llm_max_tokens=int(os.environ.get("MISTRAL_LLM_MAX_TOKENS", "180")),
        mistral_stt_model=os.environ.get(
            "MISTRAL_STT_MODEL",
            "voxtral-mini-transcribe-realtime-2602",
        ),
        mistral_stt_target_delay_ms=int(
            os.environ.get("MISTRAL_STT_TARGET_DELAY_MS", "600")
        ),
        mistral_tts_model=os.environ.get("MISTRAL_TTS_MODEL", "voxtral-mini-tts-2603"),
        mistral_tts_voice=os.environ.get("MISTRAL_TTS_VOICE", "en_paul_neutral"),
        tts_provider=os.environ.get("VOICE_RUNTIME_TTS_PROVIDER", "mistral").lower(),
        kokoro_voice=os.environ.get("KOKORO_VOICE", "af_heart"),
        # Empty -> Kokoro auto-downloads its ONNX model/voices on first use.
        kokoro_model_path=os.environ.get("KOKORO_MODEL_PATH", ""),
        kokoro_voices_path=os.environ.get("KOKORO_VOICES_PATH", ""),
        tts_transport=os.environ.get("VOICE_RUNTIME_TTS_TRANSPORT", "http").lower(),
        # confidence + min_volume were 0.7 / 0.6 — too strict, which forced
        # students to speak loudly and close to the mic (QA finding). Silero's
        # min_volume is a raw energy gate applied ON TOP of the model, so it was
        # the main culprit for dropping soft speech and feeding the STT clipped
        # fragments (a source of "Yes, I'm ready" -> "Thank you"-style errors).
        # Lowered to catch normal conversational voice at a normal distance.
        vad_confidence=float(os.environ.get("VOICE_RUNTIME_VAD_CONFIDENCE", "0.5")),
        vad_start_secs=float(os.environ.get("VOICE_RUNTIME_VAD_START_SECS", "0.2")),
        vad_stop_secs=float(os.environ.get("VOICE_RUNTIME_VAD_STOP_SECS", "1.2")),
        vad_min_volume=float(os.environ.get("VOICE_RUNTIME_VAD_MIN_VOLUME", "0.35")),
        # Pipecat's semantic turn detector is intentionally patient by default
        # (5s). Keep VAD stable for natural pauses, but cap the fallback wait so
        # a missed "done speaking" signal does not leave the UI blinking/listening.
        user_turn_stop_timeout=float(
            os.environ.get("VOICE_RUNTIME_USER_TURN_STOP_TIMEOUT", "2.0")
        ),
        voice_runtime_ice_servers=[
            server.strip()
            for server in os.environ.get(
                "VOICE_RUNTIME_ICE_SERVERS",
                "stun:stun.l.google.com:19302",
            ).split(",")
            if server.strip()
        ],
        voice_runtime_cors_origins=_resolve_cors_origins(),
        log_tts_request_text=os.environ.get(
            "VOICE_RUNTIME_LOG_TTS_REQUEST_TEXT",
            "false",
        ).lower() in {"1", "true", "yes"},
    )
