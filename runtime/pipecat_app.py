"""FastAPI entrypoint for the Pipecat WebRTC voice runtime.

Django builds the browser transport URL from ``VOICE_RUNTIME_PUBLIC_URL``
and the React voice hook posts a browser SDP offer with the signed session
token as a query parameter:

    POST https://<host>/sessions/<session_id>/webrtc/offer?token=<signed-token>

Start locally with:

    ../../venv/bin/uvicorn runtime.pipecat_app:app --host 0.0.0.0 --port 8010
"""

import asyncio
import logging
import os
from typing import Any

from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import FileResponse
from pipecat.transports.smallwebrtc.request_handler import (
    SmallWebRTCRequest,
    SmallWebRTCRequestHandler,
)
from pydantic import BaseModel, Field
from starlette.middleware.cors import CORSMiddleware

from runtime.pipecat_pipeline import run_pipecat_webrtc_session
from runtime.session_context import SessionContextError, fetch_session_context
from runtime.session_tokens import SessionTokenError, validate_session_token
from runtime.settings import get_settings
from runtime.tts_buffer import RUNTIME_AUDIO_DIR

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Socratic Books Pipecat Voice Runtime")
settings = get_settings()
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.voice_runtime_cors_origins,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)

# Dev-only isolated STT/TTS test endpoints (see runtime/test_lab.py). Off by
# default; enable locally with VOICE_RUNTIME_ENABLE_LAB=true.
if os.environ.get("VOICE_RUNTIME_ENABLE_LAB", "").lower() in {"1", "true", "yes"}:
    from runtime.test_lab import router as lab_router

    app.include_router(lab_router)
    logger.info("Voice Lab test endpoints enabled at /lab/*")


class WebRTCOfferRequest(BaseModel):
    """Browser SDP offer for a Pipecat SmallWebRTC session."""

    sdp: str
    type: str
    pc_id: str | None = Field(default=None, alias="pcId")
    restart_pc: bool | None = Field(default=None, alias="restartPc")
    request_data: dict[str, Any] | None = Field(default=None, alias="requestData")


def _validate_claims(session_id: str, token: str):
    if not token:
        raise HTTPException(status_code=401, detail="Missing session token")

    try:
        claims = validate_session_token(token, settings.voice_runtime_session_secret)
    except SessionTokenError as exc:
        logger.warning("Pipecat session token rejected: %s", exc)
        raise HTTPException(status_code=401, detail="Invalid session token") from exc

    if claims.session_id != session_id:
        logger.warning("Pipecat session id mismatch for %s", session_id)
        raise HTTPException(status_code=401, detail="Session id mismatch")

    return claims


@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok", "runtime": "pipecat"}


@app.post("/sessions/{session_id}/webrtc/offer")
async def create_webrtc_session(
    session_id: str,
    offer: WebRTCOfferRequest,
    token: str = Query(default=""),
) -> dict[str, str]:
    """Authenticate, accept a browser WebRTC offer, and start the Pipecat task."""
    claims = _validate_claims(session_id, token)

    try:
        context = await fetch_session_context(session_id)
    except SessionContextError as exc:
        logger.warning("Pipecat WebRTC context fetch failed for %s: %s", session_id, exc)
        raise HTTPException(status_code=502, detail="Failed to fetch session context") from exc

    handler = SmallWebRTCRequestHandler(ice_servers=settings.voice_runtime_ice_servers)

    async def _start_session(webrtc_connection):
        logger.info("Started Pipecat WebRTC voice session %s", session_id)
        asyncio.create_task(run_pipecat_webrtc_session(webrtc_connection, claims, context))

    request = SmallWebRTCRequest.from_dict(
        {
            "sdp": offer.sdp,
            "type": offer.type,
            "pc_id": offer.pc_id,
            "restart_pc": offer.restart_pc,
            "request_data": offer.request_data,
        }
    )
    answer = await handler.handle_web_request(request, _start_session)
    if not answer:
        raise HTTPException(status_code=500, detail="Failed to create WebRTC answer")

    return {
        "sdp": answer["sdp"],
        "type": answer["type"],
        "pcId": answer["pc_id"],
    }


@app.get("/sessions/{session_id}/utterances")
async def list_utterances(
    session_id: str,
    token: str = Query(default=""),
) -> dict[str, list[str]]:
    """List rendered TTS WAVs for this session."""
    _validate_claims(session_id, token)
    session_dir = RUNTIME_AUDIO_DIR / session_id
    if not session_dir.exists():
        return {"utterances": []}
    files = sorted(
        p.name for p in session_dir.iterdir() if p.is_file() and p.suffix == ".wav"
    )
    return {"utterances": files}


@app.get("/sessions/{session_id}/utterances/{filename}")
async def get_utterance(
    session_id: str,
    filename: str,
    token: str = Query(default=""),
) -> FileResponse:
    """Stream a single rendered TTS WAV."""
    _validate_claims(session_id, token)
    # Guard against path traversal — only allow simple wav filenames.
    if "/" in filename or ".." in filename or not filename.endswith(".wav"):
        raise HTTPException(status_code=400, detail="Invalid filename")
    path = RUNTIME_AUDIO_DIR / session_id / filename
    if not path.exists():
        raise HTTPException(status_code=404, detail="Utterance not found")
    return FileResponse(path, media_type="audio/wav", filename=filename)
