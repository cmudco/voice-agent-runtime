"""Liveness watchdog + replay recovery for the Mistral realtime STT socket.

Mistral's realtime gateway sometimes hands out dud sessions: the websocket
connects, ``Session created`` arrives, audio streams up — and no transcription
ever comes back, because the gateway's *internal* hop to the Voxtral backend
is broken (it eventually dies with ``1011 Upstream connection error``, long
after the student gave up). Pipecat's ``MistralSTTService`` only reconnects
when the socket is *closed*, so a mute-but-open socket silently eats every
turn of a session.

The inherent fix implemented here: never trust the socket, trust the
transcript. If VAD confirmed the student spoke, a transcript MUST arrive
within a deadline — otherwise the socket is presumed dead, is rebuilt, and
the student's buffered speech is replayed into the fresh connection. Speech
is kept until it has been transcribed, so a dud session degrades to a few
seconds of extra latency instead of a dead call.
"""

import asyncio
from collections import deque
from typing import AsyncGenerator

from loguru import logger
from pipecat.frames.frames import (
    CancelFrame,
    EndFrame,
    ErrorFrame,
    Frame,
    InputAudioRawFrame,
    VADUserStartedSpeakingFrame,
    VADUserStoppedSpeakingFrame,
)
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor
from pipecat.services.mistral.stt import MistralSTTService


class ResilientMistralSTTService(MistralSTTService):
    """MistralSTTService that can rebuild its socket and replay buffered speech.

    ``replay`` uses the same connection primitives the service itself uses
    (``_disconnect``/``_connect``/``send_audio``/``flush_audio``); the lock
    keeps the ordinary per-chunk streaming in ``run_stt`` from racing a
    reconnect and re-triggering the service's own lazy connect mid-rebuild.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._replay_lock = asyncio.Lock()
        self._session_over = False

    async def stop(self, frame: EndFrame):
        self._session_over = True
        await super().stop(frame)

    async def cancel(self, frame: CancelFrame):
        self._session_over = True
        await super().cancel(frame)

    async def run_stt(self, audio: bytes) -> AsyncGenerator[Frame | None, None]:
        # Audio frames queued behind the End/CancelFrame still reach this
        # service after stop() has closed the socket; the base class would
        # lazily reconnect for them, leaving an orphan Mistral session that
        # dies with 1011/timeout errors ~30s after the session ended.
        if self._session_over:
            yield None
            return
        async with self._replay_lock:
            async for frame in super().run_stt(audio):
                yield frame

    async def replay(self, audio: bytes) -> bool:
        """Rebuild the connection and re-send ``audio`` for transcription."""
        if self._session_over:
            return False
        async with self._replay_lock:
            await self._disconnect()
            await self._connect()
            if self._connection is None or self._connection.is_closed:
                logger.warning("{}: replay aborted — reconnect failed", self)
                return False
            await self._connection.send_audio(audio)
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
    MAX_BUFFER_SECS = 45.0

    def __init__(
        self,
        session_id: str,
        stt: ResilientMistralSTTService,
        *,
        sample_rate: int,
        deadline_secs: float,
    ):
        super().__init__()
        self._session_id = session_id
        self._stt = stt
        self._bytes_per_sec = sample_rate * 2
        self._deadline_secs = deadline_secs
        self._pre_roll: deque[bytes] = deque()
        self._pre_roll_bytes = 0
        self._turn_audio = bytearray()
        self._awaiting_transcript = False
        self._transcript_event = asyncio.Event()
        self._watch_task = None

    def notify_transcript(self, *, is_final: bool) -> None:
        """Called by UserTranscriptProcessor for every interim/final."""
        self._transcript_event.set()
        if is_final:
            # Turn concluded successfully; buffered speech served its purpose.
            self._awaiting_transcript = False
            self._turn_audio = bytearray()

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)

        if isinstance(frame, InputAudioRawFrame):
            self._buffer_audio(frame.audio)
        elif isinstance(frame, VADUserStartedSpeakingFrame):
            self._on_turn_started()
        elif isinstance(frame, VADUserStoppedSpeakingFrame):
            self._on_turn_stopped()
        elif isinstance(frame, (EndFrame, CancelFrame)):
            await self._cancel_watch()

        await self.push_frame(frame, direction)

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

    def _on_turn_started(self) -> None:
        if not self._awaiting_transcript:
            # Fresh turn: seed with pre-roll so replay includes the first
            # word. If a previous turn is still untranscribed, keep
            # accumulating instead — its speech is not yet safe to drop.
            self._awaiting_transcript = True
            self._turn_audio = bytearray(b"".join(self._pre_roll))
        self._transcript_event.clear()

    def _on_turn_stopped(self) -> None:
        if not self._awaiting_transcript or self._transcript_event.is_set():
            return
        if self._watch_task is None:
            self._watch_task = self.create_task(self._watch_turn())

    async def _cancel_watch(self) -> None:
        if self._watch_task is not None:
            task, self._watch_task = self._watch_task, None
            await self.cancel_task(task)

    async def _watch_turn(self) -> None:
        try:
            await self._run_deadlines()
        finally:
            self._watch_task = None

    async def _run_deadlines(self) -> None:
        if await self._transcript_within(self._deadline_secs):
            return
        buffered = bytes(self._turn_audio)
        logger.warning(
            "Pipecat session {} STT stalled: VAD closed a turn but no transcript "
            "within {}s — rebuilding socket and replaying {:.1f}s of speech",
            self._session_id,
            self._deadline_secs,
            len(buffered) / self._bytes_per_sec,
        )
        replayed = await self._stt.replay(buffered)
        # The realtime endpoint transcribes a replayed clip at roughly stream
        # pace, so the post-replay wait must scale with the clip length —
        # a fixed deadline made a good rebuilt socket look dead (observed on
        # dev: 6.6s replay, 4s deadline).
        replay_deadline = self._deadline_secs + len(buffered) / self._bytes_per_sec
        if replayed and await self._transcript_within(replay_deadline):
            logger.info(
                "Pipecat session {} STT replay recovered the turn",
                self._session_id,
            )
            return
        logger.error(
            "Pipecat session {} STT still silent after replay — surfacing error",
            self._session_id,
        )
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
