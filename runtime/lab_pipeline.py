"""STT Lab: an isolated mic → Voxtral pipeline with everything observable.

When transcription quality is questioned ("gibberish", wrong language, "not
what I said"), the exam pipeline has too many moving parts to argue about.
A lab session strips it to the minimum — browser mic → WebRTC → STT — and
mirrors every observable to the browser data channel so QA can see exactly
what the runtime hears and what Voxtral answers:

    lab_config       session/gate/model parameters at connect
    lab_levels       server-heard audio levels every second (vs the VAD gate)
    lab_vad          voice-activity start/stop as the runtime detects it
    lab_stt          STT socket lifecycle: connected (request_id), stalled,
                     recovered, failed — the watchdog's whole story
    transcript_update  interim/final text with detected language and the
                       turn's stop→final latency
    lab_error        every error the pipeline surfaces, none swallowed

No LLM, no TTS, no exam context, no Django callbacks: the only variables
left are the microphone, the network, and Voxtral.
"""

import time

from loguru import logger

from pipecat.audio.vad.vad_analyzer import VADState
from pipecat.frames.frames import (
    ErrorFrame,
    EndFrame,
    CancelFrame,
    Frame,
    InputAudioRawFrame,
    InterimTranscriptionFrame,
    OutputTransportMessageUrgentFrame,
    TranscriptionFrame,
    VADUserStartedSpeakingFrame,
    VADUserStoppedSpeakingFrame,
)
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.runner import PipelineRunner
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor
from pipecat.pipeline.task import PipelineParams, PipelineTask
from pipecat.services.mistral.stt import MistralSTTService
from pipecat.transcriptions.language import Language
from pipecat.transports.base_transport import TransportParams
from pipecat.transports.smallwebrtc.transport import SmallWebRTCTransport

from runtime.audio_constants import AGENT_OUTPUT_SAMPLE_RATE, BROWSER_INPUT_SAMPLE_RATE
from runtime.processors import AudioInputDebugProcessor
from runtime.sentry import tag_session
from runtime.session_tokens import SessionClaims
from runtime.settings import get_settings
from runtime.stt_resilience import ResilientMistralSTTService, SttLivenessWatchdog
from runtime.vad import _build_vad_analyzer
from runtime.whisper_client import WhisperClient
from runtime.speech_policy import SpeechRecoveryPolicy

settings = get_settings()


class LabVadProcessor(FrameProcessor):
    """Runs Silero VAD over inbound audio and emits VAD turn frames.

    The exam pipeline gets VAD from the LLM user aggregator's turn strategy;
    the lab has no aggregator, so this drives the same analyzer directly.
    The emitted ``VADUserStarted/StoppedSpeakingFrame`` are what make the
    STT service flush at turn end and the liveness watchdog arm — the lab
    exercises the exact code paths the exam relies on.
    """

    def __init__(self, session_id: str):
        super().__init__()
        self._session_id = session_id
        self._analyzer = _build_vad_analyzer()
        self._analyzer_ready = False
        self._speaking = False

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)
        if isinstance(frame, InputAudioRawFrame):
            if not self._analyzer_ready:
                # Normally the turn controller does this at StartFrame; the
                # lab owns the analyzer directly, so init from the first frame.
                self._analyzer.set_sample_rate(
                    frame.sample_rate or BROWSER_INPUT_SAMPLE_RATE
                )
                self._analyzer_ready = True
            state = await self._analyzer.analyze_audio(frame.audio)
            if state == VADState.SPEAKING and not self._speaking:
                self._speaking = True
                await self.push_frame(VADUserStartedSpeakingFrame())
            elif state == VADState.QUIET and self._speaking:
                self._speaking = False
                await self.push_frame(VADUserStoppedSpeakingFrame())
        await self.push_frame(frame, direction)


class LabEventMirror(FrameProcessor):
    """Pre-STT mirror: levels, VAD state, and upstream errors → browser.

    Sits before the STT service so it is guaranteed to see raw inbound audio
    (for the level meter) and the ``ErrorFrame``s the STT pushes upstream
    (which would otherwise never reach the browser-facing side).
    """

    LEVEL_INTERVAL_SECS = 1.0

    def __init__(self, session_id: str):
        super().__init__()
        self._session_id = session_id
        self._window = bytearray()
        self._sample_rate = BROWSER_INPUT_SAMPLE_RATE

    async def _emit(self, message: dict) -> None:
        await self.push_frame(
            OutputTransportMessageUrgentFrame(message=message),
            FrameDirection.DOWNSTREAM,
        )

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)

        if isinstance(frame, InputAudioRawFrame):
            self._sample_rate = frame.sample_rate or self._sample_rate
            self._window.extend(frame.audio)
            if len(self._window) >= self.LEVEL_INTERVAL_SECS * self._sample_rate * 2:
                await self._emit_levels()
        elif isinstance(frame, VADUserStartedSpeakingFrame):
            await self._emit({"type": "lab_vad", "speaking": True})
            await self._emit({"type": "listening", "isListening": True})
        elif isinstance(frame, VADUserStoppedSpeakingFrame):
            await self._emit({"type": "lab_vad", "speaking": False})
            await self._emit({"type": "listening", "isListening": False})
        elif isinstance(frame, ErrorFrame):
            # STT errors travel upstream; mirror them down to the browser so
            # the lab console shows every failure, not just fatal ones.
            await self._emit(
                {"type": "lab_error", "error": frame.error, "fatal": bool(frame.fatal)}
            )

        await self.push_frame(frame, direction)

    async def _emit_levels(self) -> None:
        import numpy as np

        from pipecat.audio.utils import calculate_audio_volume

        audio = bytes(self._window)
        self._window = bytearray()
        samples = np.frombuffer(audio, dtype=np.int16)
        if samples.size == 0:
            return
        peak = float(np.abs(samples.astype(np.int32)).max()) / 32768.0
        rms = float(np.sqrt(np.mean(np.square(samples.astype(np.float64))))) / 32768.0
        chunk = max(2, int(self._sample_rate * 0.25) * 2)
        volumes = [
            calculate_audio_volume(audio[o : o + chunk], self._sample_rate)
            for o in range(0, len(audio) - chunk + 1, chunk)
        ]
        await self._emit(
            {
                "type": "lab_levels",
                "peak": round(peak, 4),
                "rms": round(rms, 4),
                "vadVolume": round(max(volumes, default=0.0), 3),
                "gateMinVolume": settings.vad_min_volume,
                "gateConfidence": settings.vad_confidence,
            }
        )


class LabTranscriptMirror(FrameProcessor):
    """Post-STT mirror: interim/final transcripts with language + latency."""

    def __init__(self, session_id: str, watchdog: SttLivenessWatchdog | None):
        super().__init__()
        self._session_id = session_id
        self._watchdog = watchdog
        self._turn_id = 0
        self._turn_open = False
        self._turn_stopped_at: float | None = None

    async def _emit(self, message: dict) -> None:
        await self.push_frame(
            OutputTransportMessageUrgentFrame(message=message),
            FrameDirection.DOWNSTREAM,
        )

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)

        if isinstance(frame, VADUserStartedSpeakingFrame):
            if not self._turn_open:
                self._turn_open = True
                self._turn_id += 1
                self._turn_stopped_at = None
        elif isinstance(frame, VADUserStoppedSpeakingFrame):
            self._turn_stopped_at = time.monotonic()
        elif isinstance(frame, InterimTranscriptionFrame):
            if self._watchdog is not None:
                self._watchdog.notify_transcript(is_final=False, text=frame.text or "")
            await self._emit(
                {
                    "type": "transcript_update",
                    "role": "user",
                    "message": frame.text or "",
                    "content": frame.text or "",
                    "turnId": self._turn_id,
                    "isFinal": False,
                }
            )
        elif isinstance(frame, TranscriptionFrame):
            if self._watchdog is not None:
                self._watchdog.notify_transcript(is_final=True, text=frame.text or "")
            latency_ms = (
                round((time.monotonic() - self._turn_stopped_at) * 1000)
                if self._turn_stopped_at is not None
                else None
            )
            language = getattr(frame, "language", None)
            batch_id = getattr(frame, "batch_id", None)
            # A completed backup batch must not overwrite a newer live turn.
            turn_id = 1_000_000 + batch_id if batch_id is not None else self._turn_id
            if batch_id is not None:
                latency_ms = round((time.monotonic() - frame.queued_at) * 1000)
            logger.info(
                "Lab session {} final turn={} language={} latency_ms={} chars={}",
                self._session_id,
                self._turn_id,
                language,
                latency_ms,
                len(frame.text or ""),
            )
            await self._emit(
                {
                    "type": "transcript_update",
                    "role": "user",
                    "message": frame.text or "",
                    "content": frame.text or "",
                    "turnId": turn_id,
                    "isFinal": True,
                    "language": str(language) if language else None,
                    "latencyMs": latency_ms,
                }
            )
            if batch_id is None:
                self._turn_open = False
        elif isinstance(frame, ErrorFrame) and direction == FrameDirection.DOWNSTREAM:
            await self._emit({"type": "lab_error", "error": frame.error, "fatal": bool(frame.fatal)})

        await self.push_frame(frame, direction)


async def run_lab_webrtc_session(webrtc_connection, claims: SessionClaims, *, provider: str = "mistral") -> None:
    """Build and run the isolated STT Lab pipeline for one browser session."""
    tag_session(claims)
    logger.info(
        "STT Lab session {} starting for user {} (stt_model={}, gate min_volume={} "
        "confidence={}, liveness deadline={}s)",
        claims.session_id,
        claims.user_id,
        settings.mistral_stt_model,
        settings.vad_min_volume,
        settings.vad_confidence,
        settings.stt_liveness_deadline_secs,
    )

    transport = SmallWebRTCTransport(
        webrtc_connection=webrtc_connection,
        params=TransportParams(
            audio_in_enabled=True,
            audio_out_enabled=True,
            audio_in_sample_rate=BROWSER_INPUT_SAMPLE_RATE,
            audio_out_sample_rate=AGENT_OUTPUT_SAMPLE_RATE,
            video_in_enabled=False,
            video_out_enabled=False,
        ),
    )

    stt = ResilientMistralSTTService(
        api_key=settings.mistral_api_key or "unused-in-whisper-mode",
        start_in_fallback=provider == "whisper",
        sample_rate=BROWSER_INPUT_SAMPLE_RATE,
        target_streaming_delay_ms=settings.mistral_stt_target_delay_ms,
        settings=MistralSTTService.Settings(
            model=settings.mistral_stt_model,
            language=Language.EN,
        ),
    )

    task_holder: dict = {}

    async def _queue_message(message: dict) -> None:
        task = task_holder.get("task")
        if task is not None:
            await task.queue_frames(
                [OutputTransportMessageUrgentFrame(message=message)]
            )

    async def _watchdog_event(event: dict) -> None:
        await _queue_message({"type": "lab_stt", **event})

    watchdog = (
        SttLivenessWatchdog(
            claims.session_id,
            stt,
            sample_rate=BROWSER_INPUT_SAMPLE_RATE,
            deadline_secs=settings.stt_liveness_deadline_secs,
            event_sink=_watchdog_event,
            policy=SpeechRecoveryPolicy(progress_timeout=settings.stt_progress_timeout_secs,
                                        reconnect_timeout=settings.stt_reconnect_timeout_secs,
                                        reconnect_cap=settings.stt_reconnect_cap_secs),
            fallback=WhisperClient(settings.whisper_url, settings.whisper_token, language="en")
            if settings.whisper_url and settings.whisper_token else None,
            start_in_fallback=provider == "whisper",
        )
        if settings.stt_liveness_deadline_secs > 0 or provider == "whisper"
        else None
    )

    stages = [
        transport.input(),
        LabVadProcessor(claims.session_id),
        AudioInputDebugProcessor(
            claims.session_id,
            min_volume=settings.vad_min_volume,
            confidence=settings.vad_confidence,
        ),
        LabEventMirror(claims.session_id),
        *([watchdog] if watchdog else []),
        stt,
        LabTranscriptMirror(claims.session_id, watchdog),
        transport.output(),
    ]
    pipeline = Pipeline(stages)
    task = PipelineTask(
        pipeline,
        params=PipelineParams(
            audio_in_sample_rate=BROWSER_INPUT_SAMPLE_RATE,
            audio_out_sample_rate=AGENT_OUTPUT_SAMPLE_RATE,
        ),
        conversation_id=claims.session_id,
        # Pipecat prepends an RTVIProcessor by default, which forwards every
        # upstream ErrorFrame (e.g. a recoverable Mistral STT timeout the
        # watchdog is already handling) to the browser as {type: "error",
        # data: {...}} — a shape the SB frontend does not read, so it rendered
        # a generic fatal banner mid-session. The browser only understands the
        # messages BrowserEventProcessor emits.
        enable_rtvi=False,
    )
    task_holder["task"] = task

    def _ready_messages() -> list[OutputTransportMessageUrgentFrame]:
        return [
            OutputTransportMessageUrgentFrame(
                message={
                    "type": "runtime_ready",
                    "sessionId": claims.session_id,
                    "ttsTransport": settings.tts_transport,
                    "mode": "lab",
                    "sttFallbackAvailable": bool(watchdog and watchdog.fallback_available),
                    "sttProvider": "whisper" if stt.fallback_active else "mistral",
                }
            ),
            OutputTransportMessageUrgentFrame(
                message={
                    "type": "lab_config",
                    "sttModel": "self-hosted Whisper" if stt.fallback_active else settings.mistral_stt_model,
                    "sttProvider": "whisper" if stt.fallback_active else "mistral",
                    "gateMinVolume": settings.vad_min_volume,
                    "gateConfidence": settings.vad_confidence,
                    "livenessDeadlineSecs": settings.stt_liveness_deadline_secs,
                    "targetStreamingDelayMs": settings.mistral_stt_target_delay_ms,
                }
            ),
        ]

    @stt.event_handler("on_connected")
    async def _on_stt_connected(service):
        request_id = getattr(service._connection, "request_id", None)
        await _queue_message(
            {"type": "lab_stt", "event": "connected", "requestId": request_id}
        )

    @stt.event_handler("on_connection_error")
    async def _on_stt_error(service, error):
        await _queue_message(
            {"type": "lab_error", "error": str(error), "fatal": False}
        )

    @transport.event_handler("on_client_connected")
    async def _on_connected(_transport, _client):
        logger.info("STT Lab client connected: {}", claims.session_id)
        await task.queue_frames(_ready_messages())

    @transport.event_handler("on_app_message")
    async def _on_app_message(_transport, message, _sender):
        # Same lost-pre-channel-message race as the exam pipeline: reply to
        # client_ready with a fresh copy that cannot be dropped.
        if isinstance(message, dict) and message.get("type") == "client_ready":
            await task.queue_frames(_ready_messages())
        elif isinstance(message, dict) and message.get("type") in {"end", "cancel"}:
            await task.queue_frame(OutputTransportMessageUrgentFrame(message={
                "type": "session_end_ack", "requestId": message.get("requestId"),
            }))
            await task.queue_frame(EndFrame() if message["type"] == "end" else CancelFrame())

    @transport.event_handler("on_client_disconnected")
    async def _on_disconnected(_transport, _client):
        logger.info("STT Lab client disconnected: {}", claims.session_id)
        await task.cancel()

    runner = PipelineRunner(handle_sigint=False)
    try:
        await runner.run(task)
    except Exception:  # noqa: BLE001
        logger.exception("STT Lab session {} failed", claims.session_id)
