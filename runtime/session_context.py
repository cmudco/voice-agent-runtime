"""Server-to-server lookup of rich per-session config from Django.

The signed session token only carries auth claims. Everything bigger
(system prompt, first message, voice, eagerness, turn-timeout, …) is
fetched once per session from
``GET /api/voice/runtime/sessions/<session_id>/context/`` using the same
``X-Internal-Key`` as the other runtime callbacks.
"""

from dataclasses import dataclass

import httpx
from loguru import logger

from runtime.settings import get_settings

settings = get_settings()


@dataclass(frozen=True)
class SessionContext:
    """Rich per-session config the pipeline needs to build a turn."""

    first_message: str = ""
    system_prompt: str = ""
    exam_instructions: str = ""
    temperature: float = 0.7
    language: str = "en"
    voice_id: str = ""
    eagerness: str = "normal"
    turn_timeout_secs: int = 7

    @classmethod
    def empty(cls) -> "SessionContext":
        return cls()


class SessionContextError(Exception):
    """Raised when the runtime cannot fetch session context from Django."""


async def fetch_session_context(session_id: str) -> SessionContext:
    """Fetch the rich session config from Django.

    Returns a populated :class:`SessionContext` on success. Raises
    :class:`SessionContextError` on a network or auth failure so the WebRTC
    offer handler can reject the session cleanly rather than crashing
    mid-pipeline.
    """
    url = (
        f"{settings.socratic_bots_backend_url.rstrip('/')}"
        f"/api/voice/runtime/sessions/{session_id}/context/"
    )
    headers = {"X-Internal-Key": settings.dare_internal_key}
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            response = await client.get(url, headers=headers)
            response.raise_for_status()
    except httpx.HTTPError as exc:
        logger.warning("Failed to fetch session context for {}: {}", session_id, exc)
        raise SessionContextError(str(exc)) from exc

    data = response.json()
    return SessionContext(
        first_message=data.get("first_message") or "",
        system_prompt=data.get("system_prompt") or "",
        exam_instructions=data.get("exam_instructions") or "",
        temperature=float(data.get("temperature", 0.7)),
        language=data.get("language") or "en",
        voice_id=data.get("voice_id") or "",
        eagerness=data.get("eagerness") or "normal",
        turn_timeout_secs=int(data.get("turn_timeout_secs", 7)),
    )
