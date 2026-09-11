"""Sentry setup for the voice runtime.

Every ERROR-level loguru/stdlib log line becomes a Sentry event and INFO lines
become breadcrumbs, so existing ``logger.error(...)`` calls (STT watchdog
giving up, session failures) alert without further plumbing. Per-session tags
are attached in :func:`tag_session` so an alert can be traced back to the
exact session / agent / backend.
"""

from urllib.parse import urlsplit
import os

import sentry_sdk
from sentry_sdk.integrations.asyncio import AsyncioIntegration
from sentry_sdk.integrations.fastapi import FastApiIntegration
from sentry_sdk.integrations.loguru import LoggingLevels, LoguruIntegration
from sentry_sdk.integrations.starlette import StarletteIntegration

from runtime.settings import RuntimeSettings


def init_sentry(settings: RuntimeSettings) -> bool:
    """Initialise Sentry; returns False (no-op) when no DSN is configured."""
    local = settings.sentry_environment in {"development", "local", "test"}
    dsn = os.environ.get("LOCAL_SENTRY_DSN", "") if local else settings.sentry_dsn
    if not dsn:
        return False
    sentry_sdk.init(
        dsn=dsn,
        environment=settings.sentry_environment,
        traces_sample_rate=0 if local else settings.sentry_traces_sample_rate,
        send_default_pii=False,
        include_local_variables=False,
        before_send=_local_event if local else None,
        integrations=[
            StarletteIntegration(),
            FastApiIntegration(),
            AsyncioIntegration(),
            LoguruIntegration(
                level=LoggingLevels.INFO.value,
                event_level=LoggingLevels.ERROR.value,
            ),
        ],
    )
    sentry_sdk.set_tag("llm_model", settings.mistral_llm_model)
    sentry_sdk.set_tag("stt_model", settings.mistral_stt_model)
    sentry_sdk.set_tag("tts_provider", settings.tts_provider)
    sentry_sdk.set_tag("tts_transport", settings.tts_transport)
    return True


def tag_session(claims, backend_url: str = "") -> None:
    """Attach per-session identifiers to every event raised from this task.

    Each WebRTC session runs in its own asyncio task and the Asyncio
    integration gives every task its own scope, so tags set here do not leak
    across concurrent sessions.
    """
    sentry_sdk.set_tag("session_id", claims.session_id)
    sentry_sdk.set_tag("agent_id", str(claims.agent_id))
    sentry_sdk.set_tag("session_mode", claims.mode)
    # One runtime fronts staging and prod; the backend host is what tells
    # them apart in an alert.
    sentry_sdk.set_tag("backend", urlsplit(backend_url).netloc or "lab")
    sentry_sdk.set_user({"id": str(claims.user_id)})


def capture_browser_error(error: str, *, fatal: bool) -> None:
    """Report an error sent toward the browser; receipt/rendering is not proven."""
    sentry_sdk.capture_message(
        f"Voice session error surfaced to browser: {error}",
        level="fatal" if fatal else "error",
    )


def _local_event(event, hint):
    """Local testing exports exception locations, not payloads or student content."""
    safe = {key: event[key] for key in ("event_id", "timestamp", "level", "platform", "environment", "release") if key in event}
    values = []
    for exception in event.get("exception", {}).get("values", []):
        frames = [{key: frame[key] for key in ("filename", "function", "lineno", "module") if key in frame}
                  for frame in exception.get("stacktrace", {}).get("frames", [])]
        values.append({"type": exception.get("type", "Error"), "value": "Local error (message omitted)",
                       "stacktrace": {"frames": frames}})
    if values:
        safe["exception"] = {"values": values}
    else:
        safe["message"] = "Local runtime error log (message omitted)"
    return safe
