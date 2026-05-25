"""Validation for short-lived session tokens issued by Django.

Tokens carry **auth claims only** (session_id, agent_id, user_id, expiry +
HMAC signature). The rich per-session config (system prompt, voice,
eagerness, …) is fetched server-to-server by the runtime via
``/api/voice/runtime/sessions/<session_id>/context/``. This keeps tokens
small enough to fit cleanly in URL/POST query strings — multi-KB tokens
break HTTP request-line parsers and reject the handshake with 400.
"""

import base64
import hashlib
import hmac
import json
from dataclasses import dataclass
from datetime import datetime


@dataclass(frozen=True)
class SessionClaims:
    session_id: str
    agent_id: int
    user_id: int
    expires_at: datetime


class SessionTokenError(Exception):
    """Raised when a voice session token is invalid."""


def _decode_payload(encoded_payload: str) -> dict:
    padding = "=" * (-len(encoded_payload) % 4)
    payload_bytes = base64.urlsafe_b64decode(f"{encoded_payload}{padding}")
    return json.loads(payload_bytes)


def validate_session_token(token: str, secret: str) -> SessionClaims:
    """Validate a Django-issued voice runtime session token."""
    if not secret:
        raise SessionTokenError("Session secret is not configured")

    try:
        encoded_payload, received_signature = token.split(".", 1)
    except ValueError as exc:
        raise SessionTokenError("Invalid token format") from exc

    expected_signature = hmac.new(
        secret.encode("utf-8"),
        encoded_payload.encode("utf-8"),
        hashlib.sha256,
    ).hexdigest()

    if not hmac.compare_digest(expected_signature, received_signature):
        raise SessionTokenError("Invalid token signature")

    payload = _decode_payload(encoded_payload)
    expires_at = datetime.fromisoformat(payload["expires_at"])
    if datetime.now(tz=expires_at.tzinfo) >= expires_at:
        raise SessionTokenError("Session token expired")

    return SessionClaims(
        session_id=payload["session_id"],
        agent_id=int(payload["agent_id"]),
        user_id=int(payload["user_id"]),
        expires_at=expires_at,
    )
