"""Conversation transcript and callback recording for runtime sessions."""

import asyncio
import time

import httpx
from loguru import logger

from runtime.settings import get_settings


settings = get_settings()


class SessionRecorder:
    """Accumulates transcript + timing for the Django conversation callback.

    Produces the payload shape the Django runtime callback expects, so the existing
    ``/api/voice/runtime/conversation-ended/`` endpoint and transcript storage
    work unchanged.
    """

    def __init__(self, session_id: str):
        self.session_id = session_id
        self._started_at = time.monotonic()
        self.transcript: list[dict] = []
        self.metrics: dict = {
            "latency": {},
            "startup": {},
            "breakdowns": [],
        }
        self._client_ended = False
        self._notified = False
        self._started_notified = False
        self._started_notification_task: asyncio.Task | None = None

    def duration_seconds(self) -> int:
        return int(time.monotonic() - self._started_at)

    def add_transcript(self, role: str, message: str) -> dict:
        entry = {
            "role": role,
            "message": message,
            "timeInCallSecs": self.duration_seconds(),
        }
        self.transcript.append(entry)
        return entry

    def mark_client_ended(self) -> None:
        self._client_ended = True

    def was_client_ended(self) -> bool:
        return self._client_ended

    def mark_runtime_started(self) -> None:
        """Notify Django that this reserved session became an actual attempt."""
        if self._started_notified or self._started_notification_task:
            return
        self._started_at = time.monotonic()
        self._started_notification_task = asyncio.create_task(self.notify_backend_started())

    def set_latency_metric(self, key: str, value: float) -> None:
        self.metrics.setdefault("latency", {})[key] = round(value, 3)

    def add_latency_breakdown(self, events: list[str]) -> None:
        self.metrics.setdefault("breakdowns", []).append(events)

    def set_startup_metric(self, key: str, value: float) -> None:
        self.metrics.setdefault("startup", {})[key] = round(value, 3)

    async def notify_backend_started(self) -> None:
        """POST the first-audio start event back to Django (idempotent)."""
        if self._started_notified:
            return

        callback_url = (
            f"{settings.socratic_bots_backend_url.rstrip('/')}"
            "/api/voice/runtime/conversation-started/"
        )
        headers = {"X-Internal-Key": settings.dare_internal_key}
        try:
            async with httpx.AsyncClient(timeout=5.0) as client:
                response = await client.post(
                    callback_url,
                    json={"sessionId": self.session_id},
                    headers=headers,
                )
                response.raise_for_status()
            self._started_notified = True
            logger.info("Pipecat session {} marked started in Django", self.session_id)
        except Exception as exc:  # noqa: BLE001 - final callback can still complete the session
            logger.warning(
                "Pipecat session {} start callback failed: {}",
                self.session_id,
                exc,
            )

    async def wait_for_started_notification(self) -> None:
        if not self._started_notification_task:
            return
        try:
            await self._started_notification_task
        finally:
            self._started_notification_task = None

    async def ensure_backend_started(self) -> None:
        """Best-effort guarantee that completed sessions were activated first."""
        await self.wait_for_started_notification()
        if not self._started_notified:
            await self.notify_backend_started()

    async def notify_backend(self, status: str, ended_reason: str) -> None:
        """POST the completed conversation back to Django (idempotent)."""
        if self._notified:
            return
        self._notified = True

        callback_url = (
            f"{settings.socratic_bots_backend_url.rstrip('/')}"
            "/api/voice/runtime/conversation-ended/"
        )
        payload = {
            "sessionId": self.session_id,
            "status": status,
            "durationSeconds": self.duration_seconds(),
            "endedReason": ended_reason,
            "transcript": self.transcript,
            "metrics": {
                "runtime": "pipecat",
                "sttModel": settings.mistral_stt_model,
                "sttTargetDelayMs": settings.mistral_stt_target_delay_ms,
                "ttsModel": settings.mistral_tts_model,
                "ttsVoice": settings.mistral_tts_voice,
                **self.metrics,
            },
        }
        headers = {"X-Internal-Key": settings.dare_internal_key}
        try:
            async with httpx.AsyncClient(timeout=15.0) as client:
                response = await client.post(callback_url, json=payload, headers=headers)
                response.raise_for_status()
            logger.info("Pipecat session {} reported to Django ({})", self.session_id, status)
        except Exception as exc:  # noqa: BLE001 - callback must never crash teardown
            logger.warning(
                "Pipecat session {} callback failed: {}", self.session_id, exc
            )
