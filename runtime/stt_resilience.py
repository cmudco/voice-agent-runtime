"""Bounded recognition recovery based on speech/transcript progress.

An open socket does not establish that recognition is working. Conversely,
VAD can classify noise as speech, so lack of text is a recovery heuristic,
not proof of an upstream fault. One bounded replay precedes optional Whisper.
Only retained audio can be recovered; failures remain explicit to the client.
"""

import asyncio
import time
from dataclasses import dataclass
from collections import deque
from typing import AsyncGenerator

from loguru import logger
from pipecat.frames.frames import (
    CancelFrame,
    EndFrame,
    ErrorFrame,
    Frame,
    InputAudioRawFrame,
    InputTransportMessageFrame,
    OutputTransportMessageUrgentFrame,
    StartFrame,
    InterimTranscriptionFrame,
    TranscriptionFrame,
    VADUserStartedSpeakingFrame,
    VADUserStoppedSpeakingFrame,
)
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor
from pipecat.services.mistral.stt import MistralSTTService
from pipecat.utils.time import time_now_iso8601
from runtime.speech_policy import SpeechProgress, SpeechRecoveryPolicy


@dataclass
class WhisperTranscriptionFrame(TranscriptionFrame):
    batch_id: int = 0
    queued_at: float = 0.0


class _TimeoutGuardedConnection:
    """Wraps the SDK's RealtimeConnection so no socket write can hang.

    ``send_audio``/``flush_audio`` are bare ``websocket.send()`` calls; on a
    dud socket whose server stopped reading, the write buffer fills and they
    block forever. The base STT service flushes on every turn end, so one
    hung flush wedges the STT frame loop and deafens the whole pipeline for
    the rest of the session (observed on dev: 3 minutes of loud speech with
    zero VAD events). Here every write gets a deadline; on a miss the
    connection is marked dead — the service's lazy reconnect then builds a
    fresh one — and the socket is closed in the background, itself bounded.
    """

    WRITE_TIMEOUT_SECS = 3.0
    CLOSE_TIMEOUT_SECS = 2.0

    def __init__(self, inner):
        self._inner = inner
        self._dead = False

    def __getattr__(self, name):
        # request_id, session, audio_format, events, ... delegate untouched.
        return getattr(self._inner, name)

    @property
    def is_closed(self) -> bool:
        return self._dead or self._inner.is_closed

    async def send_audio(self, audio) -> None:
        await self._bounded_write("send_audio", self._inner.send_audio(audio))

    async def flush_audio(self) -> None:
        await self._bounded_write("flush_audio", self._inner.flush_audio())

    async def close(self, **kwargs) -> None:
        self._dead = True
        try:
            await asyncio.wait_for(
                self._inner.close(**kwargs), timeout=self.CLOSE_TIMEOUT_SECS
            )
        except Exception:  # noqa: BLE001 - closing a broken socket must not raise
            pass

    async def _bounded_write(self, op: str, write) -> None:
        if self._dead:
            # Reconnect happens at the service layer; dropping a write to a
            # socket already declared dead is the recovery path, not an error.
            write.close()
            return
        try:
            await asyncio.wait_for(write, timeout=self.WRITE_TIMEOUT_SECS)
        except asyncio.TimeoutError:
            self._dead = True
            logger.warning(
                "Mistral socket {} blocked for {}s — declaring the connection "
                "dead so the service reconnects",
                op,
                self.WRITE_TIMEOUT_SECS,
            )
            asyncio.get_running_loop().create_task(self._force_close())

    async def _force_close(self) -> None:
        try:
            await asyncio.wait_for(
                self._inner.close(code=1001), timeout=self.CLOSE_TIMEOUT_SECS
            )
        except Exception:  # noqa: BLE001
            pass


class ResilientMistralSTTService(MistralSTTService):
    """MistralSTTService that can rebuild its socket and replay buffered speech.

    ``replay`` uses the same connection primitives the service itself uses
    (``_disconnect``/``_connect``/``send_audio``/``flush_audio``); the lock
    keeps the ordinary per-chunk streaming in ``run_stt`` from racing a
    reconnect and re-triggering the service's own lazy connect mid-rebuild.
    """

    def __init__(self, *args, start_in_fallback=False, **kwargs):
        super().__init__(*args, **kwargs)
        self._replay_lock = asyncio.Lock()
        self._session_over = False
        self.fallback_active = start_in_fallback

    async def activate_fallback(self):
        self.fallback_active = True
        async with self._replay_lock:
            await self._disconnect()

    def deactivate_fallback(self):
        # The next audio chunk lazily rebuilds the socket through run_stt.
        self.fallback_active = False

    async def publish_fallback(self, text: str, *, batch_id: int = 0, queued_at: float = 0.0):
        if not self._session_over:
            await super().push_frame(WhisperTranscriptionFrame(text, self._user_id, time_now_iso8601(),
                                                              batch_id=batch_id, queued_at=queued_at))

    async def push_frame(self, frame, direction=FrameDirection.DOWNSTREAM):
        # Late upstream results must not duplicate a recovered Whisper turn.
        if self.fallback_active and isinstance(frame, (TranscriptionFrame, InterimTranscriptionFrame)):
            return
        await super().push_frame(frame, direction)

    async def stop(self, frame: EndFrame):
        self._session_over = True
        await super().stop(frame)

    async def cancel(self, frame: CancelFrame):
        self._session_over = True
        await super().cancel(frame)

    async def _connect(self):
        if self.fallback_active or self._session_over:
            return
        await super()._connect()
        if self._connection is not None and not isinstance(
            self._connection, _TimeoutGuardedConnection
        ):
            self._connection = _TimeoutGuardedConnection(self._connection)

    async def run_stt(self, audio: bytes) -> AsyncGenerator[Frame | None, None]:
        # Audio frames queued behind the End/CancelFrame still reach this
        # service after stop() has closed the socket; the base class would
        # lazily reconnect for them, leaving an orphan Mistral session that
        # dies with 1011/timeout errors ~30s after the session ended.
        if self._session_over or self.fallback_active:
            yield None
            return
        async with self._replay_lock:
            if self._session_over or self.fallback_active:
                yield None
                return
            async for frame in super().run_stt(audio):
                yield frame

    # Mistral rejects audio chunks over 262144 raw bytes (~8s at 16kHz mono),
    # so replays must be sent in pieces. One second per send mirrors the
    # live streaming cadence; found the hard way when a QA user's 12.9s
    # answer made every replay fail with "Audio chunk exceeds maximum size"
    # while shorter answers recovered fine.
    REPLAY_CHUNK_BYTES = 32_000

    async def replay(self, audio: bytes) -> bool:
        """Rebuild the connection and re-send ``audio`` for transcription."""
        if self._session_over:
            return False
        async with self._replay_lock:
            await self._disconnect()
            await self._connect()
            for offset in range(0, len(audio), self.REPLAY_CHUNK_BYTES):
                if self._connection is None or self._connection.is_closed:
                    logger.warning(
                        "{}: replay aborted — connection lost after {}/{} bytes",
                        self,
                        offset,
                        len(audio),
                    )
                    return False
                await self._connection.send_audio(
                    audio[offset : offset + self.REPLAY_CHUNK_BYTES]
                )
            if self._connection is None or self._connection.is_closed:
                logger.warning("{}: replay aborted — reconnect failed", self)
                return False
            await self._connection.flush_audio()
            return True


class SttLivenessWatchdog(FrameProcessor):
    """Buffers each user turn and enforces a transcript deadline on the STT.

    Placed immediately upstream of the STT service, so it sees the inbound
    audio going down and the aggregator's VAD turn frames coming up. When VAD
    closes a turn and no interim/final has been seen (``notify_transcript`` is
    called by ``UserTranscriptProcessor``), it waits ``deadline_secs``; on a
    miss it replays the buffered turn through a fresh socket, and if even the
    replay stays silent it surfaces an error to the browser instead of leaving
    the student talking to a dead session.
    """

    # Rolling pre-speech context kept ahead of VAD's start signal, which
    # trails the actual first word by start_secs.
    PRE_ROLL_SECS = 1.0
    # Hard cap on buffered speech awaiting a transcript. Long enough for a
    # full oral-exam answer; bounds memory and replay cost.
    MAX_BUFFER_SECS = 120.0

    def __init__(
        self,
        session_id: str,
        stt: ResilientMistralSTTService,
        *,
        sample_rate: int,
        deadline_secs: float,
        event_sink=None,
        fallback=None,
        policy: SpeechRecoveryPolicy | None = None,
        start_in_fallback: bool = False,
    ):
        super().__init__()
        self._session_id = session_id
        self._stt = stt
        self._bytes_per_sec = sample_rate * 2
        self._deadline_secs = deadline_secs
        # Optional async callable(dict) — the STT Lab mirrors watchdog
        # lifecycle events to the browser through this.
        self._event_sink = event_sink
        self._pre_roll: deque[bytes] = deque()
        self._pre_roll_bytes = 0
        self._turn_audio = bytearray()
        self._awaiting_transcript = False
        self._transcript_event = asyncio.Event()
        self._watch_task = None
        self._fallback = fallback
        self._fallback_active = start_in_fallback
        self._fallback_queue = asyncio.Queue(maxsize=2)
        self._fallback_task = None
        self._speaking = False
        self._overflow = False
        self._closed = False
        self._policy = policy or SpeechRecoveryPolicy()
        self._progress = SpeechProgress(self._policy)
        self._progress_task = None
        self._switch_task = None
        self._switch_lock = asyncio.Lock()
        self._batch_id = 0
        # An automatic switch is provisional until the backup hears words:
        # VAD noise can look like a stall, and a session must not be
        # downgraded for the rest of the exam on that evidence alone.
        self._probation = False
        self._probation_used = False

    def notify_transcript(self, *, is_final: bool, text: str | None = None) -> None:
        """Called by UserTranscriptProcessor for every interim/final."""
        if self._fallback_active or (text is not None and not text.strip()):
            return  # A finished batch must not erase the next, live turn.
        progressed = self._progress.transcript(time.monotonic(), text or "")
        if progressed or is_final or text is None:
            self._transcript_event.set()
        if is_final:
            # Turn concluded successfully; buffered speech served its purpose.
            self._awaiting_transcript = self._speaking
            self._turn_audio = bytearray()
            self._overflow = False

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)

        if isinstance(frame, StartFrame):
            if self._fallback_active:
                self._fallback_task = self._spawn(self._consume_fallback())
            else:
                self._progress_task = self._spawn(self._monitor_progress())
        elif isinstance(frame, InputTransportMessageFrame) and isinstance(frame.message, dict) and frame.message.get("type") == "stt_switch":
            request_id = frame.message.get("requestId")
            if isinstance(request_id, str) and len(request_id) <= 128:
                if self._switch_task is None or self._switch_task.done():
                    self._switch_task = self._spawn(self._manual_switch(request_id))
            return
        elif isinstance(frame, InputAudioRawFrame):
            self._progress.audio(time.monotonic(), len(frame.audio) / self._bytes_per_sec, self._speaking)
            self._buffer_audio(frame.audio)
        elif isinstance(frame, VADUserStartedSpeakingFrame):
            self._speaking = True
            self._progress.begin(time.monotonic(), fresh=not self._awaiting_transcript)
            self._on_turn_started()
        elif isinstance(frame, VADUserStoppedSpeakingFrame):
            self._speaking = False
            self._on_turn_stopped()
        elif isinstance(frame, (EndFrame, CancelFrame)):
            self._closed = True
            await self._cancel_watch()
            for task in (self._progress_task, self._switch_task):
                if task is not None:
                    await self.cancel_task(task)
            if self._fallback_task is not None:
                await self.cancel_task(self._fallback_task)
                self._fallback_task = None
            self._turn_audio.clear()
            self._pre_roll.clear()
            while not self._fallback_queue.empty():
                self._fallback_queue.get_nowait()

        await self.push_frame(frame, direction)

    @property
    def fallback_available(self) -> bool:
        return self._fallback is not None

    def _spawn(self, coroutine):
        # Pipecat wraps the coroutine in its task manager. If cancelled before
        # that wrapper starts, close the still-unawaited inner coroutine too.
        task = self.create_task(coroutine)
        task.add_done_callback(lambda _: coroutine.close())
        return task

    async def _monitor_progress(self):
        while not self._closed and not self._fallback_active:
            await asyncio.sleep(0.5)
            switching = self._switch_task is not None and not self._switch_task.done()
            if self._progress.stalled(time.monotonic()) and self._watch_task is None and not switching:
                self._transcript_event.clear()
                self._watch_task = self._spawn(self._watch_turn(skip_initial_wait=True))

    async def _manual_switch(self, request_id: str):
        # Abort a pending replay so the user is not held to the reconnect
        # deadline — unless the watchdog is already inside switch_to_fallback,
        # where a cancel would strand the session flagged as switched with no
        # consumer. That call holds the switch lock; let it finish and join it.
        if self.fallback_available and not self._switch_lock.locked():
            await self._cancel_watch()
        accepted = await self.switch_to_fallback(reason="user_requested")
        if not self._closed:
            await self.push_frame(OutputTransportMessageUrgentFrame(message={
                "type": "stt_switch_ack", "requestId": request_id,
                "accepted": accepted,
                "provider": "whisper" if accepted else "mistral",
            }))

    async def switch_to_fallback(self, *, reason: str) -> bool:
        async with self._switch_lock:
            if self._closed or self._fallback is None:
                return False
            if self._fallback_active:
                if reason == "user_requested":
                    self._probation = False
                self._ensure_consumer()
                return True
            # Flag and consumer are set before any await: a cancellation
            # mid-switch must not leave turns queueing with nobody to drain.
            self._fallback_active = True
            self._ensure_consumer()
            try:
                async with asyncio.timeout(5):
                    await self._stt.activate_fallback()
            except Exception:
                # The service's fallback flag already blocks reconnects/results.
                await self._emit_event("upstream_close_timeout")
            if self._closed:
                return False
            if reason == "recognition_stalled" and not self._probation_used:
                self._probation = self._probation_used = True
            else:
                self._probation = False
            await self._emit_event("fallback_started", reason=reason, provider="whisper",
                                   provisional=self._probation)
            if not self._speaking:
                self._queue_fallback()
            return True

    async def _revert_to_primary(self) -> None:
        """First backup batch after an automatic switch heard nothing: treat the
        stall as noise and hand recognition back to Mistral."""
        self._probation = False
        self._fallback_active = False
        self._stt.deactivate_fallback()
        self._progress = SpeechProgress(self._policy)
        await self._emit_event("fallback_reverted", provider="mistral")
        if self._progress_task is None or self._progress_task.done():
            self._progress_task = self._spawn(self._monitor_progress())

    def _ensure_consumer(self) -> None:
        if self._closed:
            return
        if self._fallback_task is None or self._fallback_task.done():
            self._fallback_task = self._spawn(self._consume_fallback())

    def _buffer_audio(self, audio: bytes) -> None:
        self._pre_roll.append(audio)
        self._pre_roll_bytes += len(audio)
        while self._pre_roll_bytes > self.PRE_ROLL_SECS * self._bytes_per_sec:
            self._pre_roll_bytes -= len(self._pre_roll.popleft())
        if self._awaiting_transcript:
            self._turn_audio.extend(audio)
            max_bytes = int(self.MAX_BUFFER_SECS * self._bytes_per_sec)
            if len(self._turn_audio) > max_bytes:
                self._turn_audio = self._turn_audio[-max_bytes:]
                self._overflow = True

    def _on_turn_started(self) -> None:
        if not self._awaiting_transcript:
            # Fresh turn: seed with pre-roll so replay includes the first
            # word. If a previous turn is still untranscribed, keep
            # accumulating instead — its speech is not yet safe to drop.
            self._awaiting_transcript = True
            self._turn_audio = bytearray(b"".join(self._pre_roll))
        self._transcript_event.clear()

    def _on_turn_stopped(self) -> None:
        if self._fallback_active:
            self._queue_fallback()
            return
        if not self._awaiting_transcript or self._transcript_event.is_set():
            return
        if self._watch_task is None:
            self._watch_task = self._spawn(self._watch_turn())

    def _queue_fallback(self):
        if not self._awaiting_transcript or self._closed:
            return
        # Never silently treat a truncated answer as a complete answer.
        overflow = self._overflow
        audio = bytes(self._turn_audio)
        self._awaiting_transcript = False
        self._turn_audio.clear()
        self._overflow = False
        try:
            self._fallback_queue.put_nowait((audio, overflow, time.monotonic()))
        except asyncio.QueueFull:
            self._spawn(self._fallback_error("overloaded"))

    async def _fallback_error(self, reason):
        if self._closed:
            return
        await self._emit_event("fallback_failed", reason=reason)
        await self.push_frame(ErrorFrame(
            error="Backup speech recognition could not recover this answer. Please repeat your answer.",
            fatal=False,
        ))

    async def _consume_fallback(self):
        while not self._closed:
            audio, overflow, queued_at = await self._fallback_queue.get()
            if overflow:
                await self._fallback_error("buffer_limit")
                continue
            try:
                self._batch_id += 1
                started = time.monotonic()
                await self._emit_event("fallback_processing", batchId=self._batch_id)
                text = await self._fallback.transcribe(audio)
                if self._closed:
                    return
                if text:
                    self._probation = False
                    await self._stt.publish_fallback(text, batch_id=self._batch_id, queued_at=queued_at)
                    await self._emit_event("fallback_recovered", batchId=self._batch_id,
                                           recognitionMs=round((time.monotonic() - started) * 1000),
                                           audioSeconds=round(len(audio) / self._bytes_per_sec, 2))
                elif (self._probation and self._fallback_queue.empty()
                      and not self._awaiting_transcript and not self._speaking):
                    await self._revert_to_primary()
                    return
                else:
                    # An empty earlier batch must not abandon newer speech or
                    # override the user's explicit choice of recognition service.
                    self._probation = False
                    await self._fallback_error("no_speech")
            except asyncio.CancelledError:
                raise
            except Exception:
                # Do not log request bodies, student audio, or credentials.
                await self._fallback_error("unavailable")

    async def _cancel_watch(self) -> None:
        if self._watch_task is not None:
            task, self._watch_task = self._watch_task, None
            await self.cancel_task(task)

    async def _watch_turn(self, skip_initial_wait: bool = False) -> None:
        try:
            await self._run_deadlines(skip_initial_wait=skip_initial_wait)
        finally:
            self._watch_task = None

    async def _emit_event(self, event: str, **fields) -> None:
        if self._event_sink is None:
            return
        try:
            await self._event_sink({"event": event, **fields})
        except Exception as exc:  # noqa: BLE001 - telemetry must not break recovery
            logger.warning(
                "Pipecat session {} watchdog event sink failed: {}",
                self._session_id,
                exc,
            )

    async def _run_deadlines(self, skip_initial_wait: bool = False) -> None:
        if self._closed or self._fallback_active:
            return
        if not skip_initial_wait and await self._transcript_within(self._deadline_secs):
            return
        for attempt in range(1, self._policy.reconnect_attempts + 1):
            if self._closed or self._transcript_event.is_set():
                return
            # Re-read each attempt: the student may keep talking into the
            # stall and that speech belongs in the replay too.
            buffered = bytes(self._turn_audio)
            logger.warning(
                "Pipecat session {} STT stalled: no transcript progress "
                "— rebuild attempt {}/{}, replaying {:.1f}s of buffered audio",
                self._session_id,
                attempt,
                self._policy.reconnect_attempts,
                len(buffered) / self._bytes_per_sec,
            )
            await self._emit_event(
                "stalled",
                attempt=attempt,
                maxAttempts=self._policy.reconnect_attempts,
                bufferedSecs=round(len(buffered) / self._bytes_per_sec, 1),
            )
            deadline = self._policy.reconnect_deadline(len(buffered) / self._bytes_per_sec)
            try:
                # One deadline covers connection + replay + recognition, rather
                # than waiting another clip-length interval after a slow replay.
                async with asyncio.timeout(deadline):
                    replayed = await self._stt.replay(buffered)
                    recovered = replayed and await self._transcript_within(deadline)
            except Exception:  # Provider failures must not skip the backup path.
                recovered = False
            if recovered:
                logger.info(
                    "Pipecat session {} STT replay recovered the turn "
                    "(attempt {})",
                    self._session_id,
                    attempt,
                )
                await self._emit_event("recovered", attempt=attempt)
                return
        if self._closed or self._transcript_event.is_set():
            return
        if self._fallback is not None:
            await self.switch_to_fallback(reason="recognition_stalled")
            return
        logger.error(
            "Pipecat session {} STT still silent after {} rebuilds — "
            "surfacing error",
            self._session_id,
            self._policy.reconnect_attempts,
        )
        await self._emit_event("failed", attempts=self._policy.reconnect_attempts)
        await self.push_frame(
            ErrorFrame(
                error="Speech service stalled — please repeat your answer.",
                fatal=False,
            ),
            FrameDirection.DOWNSTREAM,
        )

    async def _transcript_within(self, timeout_secs: float) -> bool:
        try:
            await asyncio.wait_for(self._transcript_event.wait(), timeout=timeout_secs)
            return True
        except asyncio.TimeoutError:
            return False
