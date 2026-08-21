"""Brain fallback: bound the LLM's time-to-first-token and fail over.

Mistral's chat endpoint has been observed accepting a request and then
staying silent for 40-90s with no error (dev + prod, 2026-08-21). Pipecat's
service only reacts to exceptions, so the student just waited. This service
runs an ordered list of OpenAI-compatible candidates and moves to the next
one whenever the active candidate either raises or fails to produce its
first token within the deadline.

Failover is only attempted while nothing has been spoken for the turn yet:
once a token has reached the TTS, switching would make the agent repeat
itself, so from then on the stream is simply left to finish (bounded by the
client's read timeout).
"""

import asyncio
from collections.abc import Awaitable, Callable

import httpx
import sentry_sdk
from loguru import logger
from openai import AsyncOpenAI, DefaultAsyncHttpxClient
from pipecat.frames.frames import Frame, LLMTextFrame
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.processors.frame_processor import FrameDirection
from pipecat.services.openai.llm import OpenAILLMService

from runtime.llm_providers import LLMCandidate, shape_completion_params

FallbackCallback = Callable[[LLMCandidate, LLMCandidate, str], Awaitable[None]]


class LLMFirstTokenTimeout(Exception):
    """The active candidate produced no token within the TTFB deadline."""


class _BoundedFirstChunkStream:
    """Wraps an OpenAI stream so the wait for the FIRST chunk is deadline-bound."""

    def __init__(self, stream, deadline_secs: float):
        self._stream = stream
        self._deadline = deadline_secs
        self._iter = stream.__aiter__()
        self._first = True

    def __aiter__(self):
        return self

    async def __anext__(self):
        if not self._first:
            return await self._iter.__anext__()
        self._first = False
        try:
            return await asyncio.wait_for(self._iter.__anext__(), timeout=self._deadline)
        except asyncio.TimeoutError as exc:
            raise LLMFirstTokenTimeout(
                f"no first chunk within {self._deadline:.1f}s"
            ) from exc

    async def aclose(self):
        await self._stream.close()


class ResilientLLMService(OpenAILLMService):
    """OpenAI-compatible LLM service with TTFB deadline + ordered failover.

    Candidates that fail are demoted to the end of the order for the rest of
    the session, so a dead primary costs one deadline, not one per turn.
    """

    def __init__(
        self,
        *,
        candidates: list[LLMCandidate],
        ttfb_deadline_secs: float,
        response_timeout_secs: float,
        session_id: str,
        on_fallback: FallbackCallback | None = None,
        **kwargs,
    ):
        if not candidates:
            raise ValueError("ResilientLLMService needs at least one candidate")
        self._candidates = list(candidates)
        self._ttfb_deadline = ttfb_deadline_secs
        self._response_timeout = response_timeout_secs
        self._session_id = session_id
        self._on_fallback = on_fallback
        self._active: LLMCandidate = candidates[0]
        self._tokens_this_turn = 0
        primary = candidates[0]
        super().__init__(api_key=primary.api_key, base_url=primary.base_url, **kwargs)
        self._clients = {c.label: self._make_client(c) for c in self._candidates}

    # Mistral rejects the "developer" role; we never emit it, but keep the
    # conservative setting so the adapter downgrades it for every provider.
    supports_developer_role = False

    def _make_client(self, candidate: LLMCandidate) -> AsyncOpenAI:
        return AsyncOpenAI(
            api_key=candidate.api_key,
            base_url=candidate.base_url,
            max_retries=0,
            http_client=DefaultAsyncHttpxClient(
                timeout=httpx.Timeout(self._response_timeout, connect=5.0),
                limits=httpx.Limits(max_keepalive_connections=20, max_connections=100),
            ),
        )

    @property
    def active_candidate(self) -> LLMCandidate:
        return self._active

    def _use(self, candidate: LLMCandidate) -> None:
        self._active = candidate
        self._client = self._clients[candidate.label]
        self._settings.model = candidate.model

    def build_chat_completion_params(self, params_from_context) -> dict:
        params = super().build_chat_completion_params(params_from_context)
        return shape_completion_params(self._active, params)

    async def get_chat_completions(self, context: LLMContext):
        adapter = self.get_llm_adapter()
        params_from_context = adapter.get_llm_invocation_params(
            context,
            system_instruction=None,
            convert_developer_to_user=True,
        )
        params = self.build_chat_completion_params(params_from_context)
        try:
            stream = await asyncio.wait_for(
                self._client.chat.completions.create(**params),
                timeout=self._ttfb_deadline,
            )
        except asyncio.TimeoutError as exc:
            raise LLMFirstTokenTimeout(
                f"no response headers within {self._ttfb_deadline:.1f}s"
            ) from exc
        return _BoundedFirstChunkStream(stream, self._ttfb_deadline)

    async def push_frame(self, frame: Frame, direction: FrameDirection = FrameDirection.DOWNSTREAM):
        if isinstance(frame, LLMTextFrame):
            self._tokens_this_turn += 1
        await super().push_frame(frame, direction)

    async def _process_context(self, context: LLMContext):
        order = list(self._candidates)
        last_exc: Exception | None = None
        for attempt, candidate in enumerate(order, start=1):
            self._use(candidate)
            self._tokens_this_turn = 0
            try:
                await super()._process_context(context)
            except asyncio.CancelledError:
                raise
            except Exception as exc:  # noqa: BLE001 - any provider failure is a failover trigger
                if self._tokens_this_turn:
                    # Already speaking: switching now would repeat the answer.
                    raise
                last_exc = exc
                reason = f"{type(exc).__name__}: {exc}"
                self._demote(candidate)
                nxt = order[attempt] if attempt < len(order) else None
                logger.warning(
                    "Pipecat session {} brain {} failed before first token ({}) — {}",
                    self._session_id,
                    candidate.label,
                    reason,
                    f"falling back to {nxt.label}" if nxt else "no fallback left",
                )
                if nxt is not None and self._on_fallback is not None:
                    await self._on_fallback(candidate, nxt, reason)
                continue
            if attempt > 1:
                logger.info(
                    "Pipecat session {} brain fallback {} answered the turn",
                    self._session_id,
                    candidate.label,
                )
            return
        sentry_sdk.capture_message(
            f"Voice brain: all {len(order)} candidates failed for a turn "
            f"(last: {last_exc})",
            level="error",
        )
        raise last_exc  # type: ignore[misc]

    def _demote(self, candidate: LLMCandidate) -> None:
        if candidate in self._candidates:
            self._candidates.remove(candidate)
            self._candidates.append(candidate)
