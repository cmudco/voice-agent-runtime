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
    # Brain model chosen for the agent in Socratic Bots; empty = runtime default.
    llm_model: str = ""
    # The backend that owns this session. Resolved by probing the configured
    # backends (staging/prod share one runtime) so the later conversation
    # callbacks are sent back to the same backend that served the context.
    backend_url: str = ""


class SessionContextError(Exception):
    """Raised when the runtime cannot fetch session context from Django."""


class SessionContextRejected(SessionContextError):
    """A cancelled, expired, or already claimed attempt is not a service outage."""


# Above ~1.0 the LLM starts emitting unstable tokens that the TTS then tries to
# voice, which QA saw as brief bursts of gibberish/other-language speech at the
# start of turns (they were running temperature=1.4). Clamp defensively so a
# misconfigured agent can't destabilize a live exam; anything <=1.0 is untouched.
_MAX_EXAM_TEMPERATURE = 1.0


def _build_context(data: dict, backend_url: str) -> SessionContext:
    return SessionContext(
        first_message=data.get("first_message") or "",
        system_prompt=data.get("system_prompt") or "",
        exam_instructions=data.get("exam_instructions") or "",
        temperature=min(float(data.get("temperature", 0.7)), _MAX_EXAM_TEMPERATURE),
        language=data.get("language") or "en",
        voice_id=data.get("voice_id") or "",
        eagerness=data.get("eagerness") or "normal",
        turn_timeout_secs=int(data.get("turn_timeout_secs", 7)),
        llm_model=str(data.get("llm_model") or "").strip(),
        backend_url=backend_url,
    )


async def fetch_session_context(session_id: str) -> SessionContext:
    """Fetch the rich session config from the backend that owns the session.

    A single runtime serves multiple backends (staging/prod), so we probe each
    configured ``SOCRATIC_BOTS_BACKEND_URLS`` entry in order. A backend that
    doesn't recognize the session returns 404, so we move on; the first backend
    that returns the context owns the session, and its URL is recorded on the
    returned :class:`SessionContext` for the later callbacks.

    Raises :class:`SessionContextError` if no backend served the context so the
    WebRTC offer handler can reject the session cleanly rather than crashing
    mid-pipeline.
    """
    headers = {"X-Internal-Key": settings.dare_internal_key}
    backend_urls = settings.socratic_bots_backend_urls
    last_error: str = "no backends configured"

    async with httpx.AsyncClient(timeout=10.0) as client:
        for backend_url in backend_urls:
            url = f"{backend_url}/api/voice/runtime/sessions/{session_id}/context/"
            try:
                response = await client.post(url, headers=headers)
                # Rolling upgrade: old backends expose only GET. Enable early
                # reservation replacement only after all runtimes use POST.
                if response.status_code == 405:
                    response = await client.get(url, headers=headers)
            except httpx.HTTPError as exc:
                last_error = str(exc)
                logger.exception(
                    "Session context request to {} failed for {}: {}",
                    backend_url,
                    session_id,
                    exc,
                )
                continue

            # 404 means this backend doesn't own the session — try the next one.
            if response.status_code == 404:
                last_error = f"{backend_url} returned 404"
                continue
            if response.status_code == 409:
                raise SessionContextRejected('Session startup is no longer available')
            try:
                response.raise_for_status()
            except httpx.HTTPError as exc:
                last_error = str(exc)
                logger.exception(
                    "Session context lookup at {} errored for {}: {}",
                    backend_url,
                    session_id,
                    exc,
                )
                continue

            return _build_context(response.json(), backend_url)

    logger.info(
        "No backend served session context for {} (tried {}): {}",
        session_id,
        backend_urls,
        last_error,
    )
    raise SessionContextError(last_error)
