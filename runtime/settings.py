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
    socratic_bots_backend_url: str
    dare_internal_key: str
    voice_runtime_session_secret: str
    mistral_llm_model: str
    mistral_llm_max_tokens: int
    mistral_stt_model: str
    mistral_stt_target_delay_ms: int
    mistral_tts_model: str
    mistral_tts_voice: str
    vad_confidence: float
    vad_start_secs: float
    vad_stop_secs: float
    vad_min_volume: float
    voice_runtime_ice_servers: list[str]
    voice_runtime_cors_origins: list[str]
    log_tts_request_text: bool


def get_settings() -> RuntimeSettings:
    internal_key = os.environ.get("DARE_INTERNAL_KEY", "")
    return RuntimeSettings(
        mistral_api_key=os.environ.get("MISTRAL_API_KEY", ""),
        socratic_bots_backend_url=os.environ.get(
            "SOCRATIC_BOTS_BACKEND_URL",
            "http://localhost:8001",
        ),
        dare_internal_key=internal_key,
        voice_runtime_session_secret=os.environ.get("VOICE_RUNTIME_SESSION_SECRET")
        or internal_key,
        mistral_llm_model=os.environ.get("MISTRAL_LLM_MODEL", "mistral-large-latest"),
        mistral_llm_max_tokens=int(os.environ.get("MISTRAL_LLM_MAX_TOKENS", "180")),
        mistral_stt_model=os.environ.get(
            "MISTRAL_STT_MODEL",
            "voxtral-mini-latest",
        ),
        mistral_stt_target_delay_ms=int(
            os.environ.get("MISTRAL_STT_TARGET_DELAY_MS", "600")
        ),
        mistral_tts_model=os.environ.get("MISTRAL_TTS_MODEL", "voxtral-mini-tts-2603"),
        mistral_tts_voice=os.environ.get("MISTRAL_TTS_VOICE", "en_paul_neutral"),
        vad_confidence=float(os.environ.get("VOICE_RUNTIME_VAD_CONFIDENCE", "0.7")),
        vad_start_secs=float(os.environ.get("VOICE_RUNTIME_VAD_START_SECS", "0.2")),
        vad_stop_secs=float(os.environ.get("VOICE_RUNTIME_VAD_STOP_SECS", "1.2")),
        vad_min_volume=float(os.environ.get("VOICE_RUNTIME_VAD_MIN_VOLUME", "0.6")),
        voice_runtime_ice_servers=[
            server.strip()
            for server in os.environ.get(
                "VOICE_RUNTIME_ICE_SERVERS",
                "stun:stun.l.google.com:19302",
            ).split(",")
            if server.strip()
        ],
        voice_runtime_cors_origins=[
            origin.strip()
            for origin in os.environ.get(
                "VOICE_RUNTIME_CORS_ORIGINS",
                # Default includes both common Vite ports on localhost and 127.0.0.1.
                "http://localhost:5173,http://127.0.0.1:5173,"
                "http://localhost:5174,http://127.0.0.1:5174",
            ).split(",")
            if origin.strip()
        ],
        log_tts_request_text=os.environ.get(
            "VOICE_RUNTIME_LOG_TTS_REQUEST_TEXT",
            "false",
        ).lower() in {"1", "true", "yes"},
    )
