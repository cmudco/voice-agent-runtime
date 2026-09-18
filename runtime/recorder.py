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

    def __init__(self, session_id: str, backend_url: str):
        self.session_id = session_id
        # The backend that owns this session (resolved when its context was
        # fetched). All callbacks for this session go back to this same backend.
        self.backend_url = backend_url.rstrip("/")
        self._started_at = time.monotonic()
        self.transcript: list[dict] = []
        self.metrics: dict = {
            "latency": {},
            "startup": {},
            "breakdowns": [],
        }
        self._client_ended = False
        self.client_end_reason: str | None = None
        self.connected = False
        self._last_user_final_at: float | None = None
        self._last_agent_audio_at: float | None = None
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
        if role == "user":
            self._last_user_final_at = time.monotonic()
        return entry

    def mark_agent_audio_queued(self) -> None:
        self._last_agent_audio_at = time.monotonic()
        self.health_event("agent_audio_queued")

    def health_event(self, event: str) -> None:
        health = self.metrics.setdefault("sessionHealth", {})
        counts = health.setdefault("events", {})
        counts[event] = counts.get(event, 0) + 1
        logger.info("Pipecat session {} health event={}", self.session_id, event)

    def record_exit(self, reason: str, *, completed: bool = False) -> None:
        allowed = {"manual_end", "time_limit", "route_change", "page_hidden_exit",
                   "connection_failed", "connection_disconnected", "client_ended"}
        if not isinstance(reason, str) or reason not in allowed:
            reason = "client_ended" if completed else "client_cancelled"
        # A later teardown must never overwrite the explicit user action.
        if self.client_end_reason is None or completed:
            self.client_end_reason = reason
        if completed:
            self.mark_client_ended()
        self.health_event("exit_" + reason)

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

    def add_brain_fallback(self, failed: str, next_model: str, reason: str) -> None:
        self.metrics.setdefault("brainFallbacks", []).append(
            {
                "failed": failed,
                "next": next_model,
                "reason": reason[:300],
                "timeInCallSecs": self.duration_seconds(),
            }
        )

    async def notify_backend_started(self) -> None:
        """POST the first-audio start event back to Django (idempotent)."""
        if self._started_notified:
            return

        callback_url = f"{self.backend_url}/api/voice/runtime/conversation-started/"
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
        health = self.metrics.setdefault("sessionHealth", {})
        health.update({
            "connected": self.connected,
            "userMessages": sum(t["role"] == "user" for t in self.transcript),
            "agentMessages": sum(t["role"] == "agent" for t in self.transcript),
            "lastTranscriptRole": self.transcript[-1]["role"] if self.transcript else None,
            "exitReason": ended_reason,
            # This is an output-queue measurement, NOT proof of browser playback.
            "userFinalAfterLastAudioQueued": self._last_user_final_at is not None and (
                self._last_agent_audio_at is None or self._last_user_final_at > self._last_agent_audio_at
            ),
        })
        logger.info("Pipecat session {} health summary={}", self.session_id, health)

        callback_url = f"{self.backend_url}/api/voice/runtime/conversation-ended/"
        payload = {
            "sessionId": self.session_id,
            "status": status,
            "durationSeconds": self.duration_seconds(),
            "endedReason": ended_reason,
            "transcript": self.transcript,
            "metrics": {
                "runtime": "pipecat",
                "llmModel": settings.mistral_llm_model,
                "sttModel": settings.mistral_stt_model,
                "sttLanguageMode": "auto-detect",
                "sttTargetDelayMs": settings.mistral_stt_target_delay_ms,
                "ttsProvider": settings.tts_provider,
                "ttsTransport": settings.tts_transport,
                "ttsModel": settings.mistral_tts_model,
                "ttsVoice": settings.mistral_tts_voice,
                "kokoroVoice": settings.kokoro_voice,
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
