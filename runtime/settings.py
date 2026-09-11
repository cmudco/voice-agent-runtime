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
    # Sentry is a no-op when the DSN is empty (local dev).
    sentry_dsn: str
    sentry_environment: str
    sentry_traces_sample_rate: float
    socratic_bots_backend_urls: list[str]
    dare_internal_key: str
    voice_runtime_session_secret: str
    mistral_llm_model: str
    mistral_llm_max_tokens: int
    # Brain fallback (see llm_resilience). Keys for non-Mistral providers are
    # optional: a fallback model whose provider has no key is skipped.
    openai_api_key: str
    anthropic_api_key: str
    gemini_api_key: str
    llm_fallback_models: list[str]
    llm_ttfb_deadline_secs: float
    llm_response_timeout_secs: float
    mistral_stt_model: str
    mistral_stt_target_delay_ms: int
    mistral_tts_model: str
    mistral_tts_voice: str
    # TTS engine: "kokoro" (local, English-pinned, ~3x realtime) is the tested
    # production choice; "mistral" (Voxtral TTS) is kept as a fallback. The env
    # default stays "mistral" for backward compatibility with older deployments,
    # so production must set VOICE_RUNTIME_TTS_PROVIDER=kokoro explicitly.
    tts_provider: str
    kokoro_voice: str
    kokoro_model_path: str
    kokoro_voices_path: str
    # TTS transport: "webrtc" streams Kokoro audio over the WebRTC track,
    # in-order with the transcript events — this is the tested production
    # configuration (A/V stays in sync, no per-utterance buffering pause).
    # "http" (env default, kept for backward compatibility) pre-renders each
    # utterance and plays WAVs over HTTP; it works with either TTS engine and
    # tolerates provider-side streaming jitter (the reason Mistral TTS cannot
    # go over WebRTC directly). "webrtc" is only valid with tts_provider=kokoro.
    tts_transport: str
    vad_confidence: float
    vad_start_secs: float
    vad_stop_secs: float
    vad_min_volume: float
    user_turn_stop_timeout: float
    voice_runtime_ice_servers: list[str]
    voice_runtime_cors_origins: list[str]
    log_tts_request_text: bool
    # Debug only: dump each LLM request's full message context to /tmp (see
    # LLMInputDebugProcessor). Contains student answers — keep off in production.
    llm_input_debug: bool
    # Inbound mic-audio level telemetry (see AudioInputDebugProcessor). Logs
    # levels only — never speech content — so it defaults on: it is the only
    # way to tell a silent browser track from speech lost below the VAD gate.
    audio_input_debug: bool
    # STT liveness deadline (see SttLivenessWatchdog): seconds allowed between
    # VAD closing a user turn and the first transcript before the Mistral
    # socket is presumed dead and rebuilt with a replay. Healthy sessions
    # respond well under 2s; <= 0 disables the watchdog.
    stt_liveness_deadline_secs: float
    whisper_url: str
    whisper_token: str
    stt_progress_timeout_secs: float
    stt_reconnect_timeout_secs: float


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
    whisper_url = os.environ.get("VOICE_RUNTIME_WHISPER_URL", "").strip()
    whisper_token = os.environ.get("VOICE_RUNTIME_WHISPER_TOKEN", "")
    if whisper_url and not whisper_token:
        raise ValueError("VOICE_RUNTIME_WHISPER_TOKEN is required when Whisper is enabled")
    if whisper_url and float(os.environ.get("VOICE_RUNTIME_STT_LIVENESS_DEADLINE_SECS", "4")) <= 0:
        raise ValueError("Whisper fallback requires a positive STT liveness deadline")
    return RuntimeSettings(
        mistral_api_key=os.environ.get("MISTRAL_API_KEY", ""),
        sentry_dsn=os.environ.get("SENTRY_DSN", ""),
        sentry_environment=os.environ.get("SENTRY_ENVIRONMENT", "development"),
        sentry_traces_sample_rate=float(
            os.environ.get("SENTRY_TRACES_SAMPLE_RATE", "0.1")
        ),
        socratic_bots_backend_urls=_resolve_backend_urls(),
        dare_internal_key=internal_key,
        voice_runtime_session_secret=os.environ.get("VOICE_RUNTIME_SESSION_SECRET")
        or internal_key,
        mistral_llm_model=os.environ.get("MISTRAL_LLM_MODEL", "mistral-large-latest"),
        mistral_llm_max_tokens=int(os.environ.get("MISTRAL_LLM_MAX_TOKENS", "180")),
        openai_api_key=os.environ.get("OPENAI_API_KEY", ""),
        anthropic_api_key=os.environ.get("ANTHROPIC_API_KEY", ""),
        gemini_api_key=os.environ.get("GEMINI_API_KEY", ""),
        llm_fallback_models=[
            m.strip()
            for m in os.environ.get(
                "VOICE_RUNTIME_LLM_FALLBACKS",
                "gpt-5.6-luna,claude-haiku-4-5-20251001,mistral-large-latest",
            ).split(",")
            if m.strip()
        ],
        llm_ttfb_deadline_secs=float(
            os.environ.get("VOICE_RUNTIME_LLM_TTFB_DEADLINE_SECS", "6.0")
        ),
        llm_response_timeout_secs=float(
            os.environ.get("VOICE_RUNTIME_LLM_RESPONSE_TIMEOUT_SECS", "30.0")
        ),
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
        llm_input_debug=os.environ.get(
            "VOICE_RUNTIME_LLM_INPUT_DEBUG",
            "false",
        ).lower() in {"1", "true", "yes"},
        audio_input_debug=os.environ.get(
            "VOICE_RUNTIME_AUDIO_INPUT_DEBUG",
            "true",
        ).lower() in {"1", "true", "yes"},
        stt_liveness_deadline_secs=float(
            os.environ.get("VOICE_RUNTIME_STT_LIVENESS_DEADLINE_SECS", "4.0")
        ),
        whisper_url=whisper_url,
        whisper_token=whisper_token,
        stt_progress_timeout_secs=float(os.environ.get("VOICE_RUNTIME_STT_PROGRESS_TIMEOUT_SECS", "12")),
        stt_reconnect_timeout_secs=float(os.environ.get("VOICE_RUNTIME_STT_RECONNECT_TIMEOUT_SECS", "12")),
    )
