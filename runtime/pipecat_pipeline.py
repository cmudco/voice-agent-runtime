"""Pipecat full-duplex pipeline for the Socratic voice exam."""

from loguru import logger

from pipecat.frames.frames import OutputTransportMessageUrgentFrame, TTSSpeakFrame
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.runner import PipelineRunner
from pipecat.pipeline.task import PipelineParams, PipelineTask
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.processors.aggregators.llm_response_universal import (
    LLMContextAggregatorPair,
    LLMUserAggregatorParams,
)
from pipecat.observers.startup_timing_observer import StartupTimingObserver
from pipecat.observers.user_bot_latency_observer import UserBotLatencyObserver
from pipecat.services.mistral.llm import MistralLLMService
from pipecat.services.mistral.tts import MistralTTSService
from pipecat.transports.base_transport import TransportParams
from pipecat.transports.smallwebrtc.transport import SmallWebRTCTransport

from runtime.audio_constants import (
    AGENT_OUTPUT_SAMPLE_RATE,
    BROWSER_INPUT_SAMPLE_RATE,
    MISTRAL_TTS_SAMPLE_RATE,
)
from runtime.batch_stt import MistralBatchSTT
from runtime.processors import (
    AgentTurnCoordinator,
    BrowserEventProcessor,
    ClientControlProcessor,
    LLMInputDebugProcessor,
    UserTranscriptProcessor,
)
from runtime.recorder import SessionRecorder
from runtime.session_context import SessionContext
from runtime.session_tokens import SessionClaims
from runtime.settings import get_settings
from runtime.text_utils import (
    _build_system_prompt,
    _is_english_session,
    _normalize_tts_text,
    _resolve_first_message,
)
from runtime.tts_buffer import BufferedMistralTTSService
from runtime.vad import _build_vad_analyzer


settings = get_settings()


async def _run_transport_session(
    transport: SmallWebRTCTransport,
    claims: SessionClaims,
    context: "SessionContext",
) -> None:
    """Build and run the Pipecat WebRTC pipeline for one browser session."""
    recorder = SessionRecorder(claims.session_id, context.backend_url)
    logger.info(
        (
            "Pipecat WebRTC session {} starting with stt_target_delay_ms={}, "
            "vad_stop_secs={}, vad_start_secs={}, vad_confidence={}, vad_min_volume={}"
        ),
        claims.session_id,
        settings.mistral_stt_target_delay_ms,
        settings.vad_stop_secs,
        settings.vad_start_secs,
        settings.vad_confidence,
        settings.vad_min_volume,
    )

    # Batch transcription via the standard HTTP endpoint - the realtime
    # WebSocket endpoint is billing-gated and 403s on the Experiment tier.
    stt = MistralBatchSTT(
        api_key=settings.mistral_api_key,
        model=settings.mistral_stt_model,
        language="en" if _is_english_session(context.language) else None,
        sample_rate=BROWSER_INPUT_SAMPLE_RATE,
    )
    tts = BufferedMistralTTSService(
        api_key=settings.mistral_api_key,
        sample_rate=MISTRAL_TTS_SAMPLE_RATE,
        text_transforms=[("*", _normalize_tts_text)],
        settings=MistralTTSService.Settings(
            model=settings.mistral_tts_model,
            # Per-agent voice if set; otherwise the env default.
            voice=context.voice_id or settings.mistral_tts_voice,
        ),
        output_sample_rate=AGENT_OUTPUT_SAMPLE_RATE,
        # Enables stable HTTP playback of pre-rendered agent utterances.
        session_id=claims.session_id,
    )
    llm = MistralLLMService(
        api_key=settings.mistral_api_key,
        settings=MistralLLMService.Settings(
            model=settings.mistral_llm_model,
            temperature=context.temperature,
            max_tokens=settings.mistral_llm_max_tokens,
        ),
    )

    first_message = _resolve_first_message(context)
    # Bound to a distinct name so it doesn't shadow the SessionContext
    # parameter that later setup code still reads (eagerness/turn_timeout).
    llm_context = LLMContext(
        messages=[
            {"role": "system", "content": _build_system_prompt(context)},
            {"role": "assistant", "content": first_message},
        ]
    )
    aggregators = LLMContextAggregatorPair(
        llm_context,
        user_params=LLMUserAggregatorParams(
            vad_analyzer=_build_vad_analyzer(context.eagerness),
            # "Turn Timeout" — how long of silence before the agent prompts.
            user_idle_timeout=float(context.turn_timeout_secs),
        ),
    )

    @tts.event_handler("on_tts_request")
    async def _on_tts_request(_tts, _context_id, text):
        if settings.log_tts_request_text:
            logger.debug("Pipecat session {} TTS request text: {}", claims.session_id, text)

    user_transcript = UserTranscriptProcessor(recorder)
    client_control = ClientControlProcessor(recorder)
    llm_input_debug = LLMInputDebugProcessor(claims.session_id)
    agent_turns = AgentTurnCoordinator()
    browser_events = BrowserEventProcessor(recorder, agent_turns)

    pipeline = Pipeline(
        [
            transport.input(),
            client_control,
            stt,
            user_transcript,
            aggregators.user(),
            llm_input_debug,
            llm,
            tts,
            browser_events,
            transport.output(),
            aggregators.assistant(),
        ]
    )

    startup_observer = StartupTimingObserver()
    latency_observer = UserBotLatencyObserver()

    @latency_observer.event_handler("on_latency_measured")
    async def _on_latency_measured(_observer, latency_seconds):
        recorder.set_latency_metric("lastUserToBotSecs", latency_seconds)
        logger.info(
            "Pipecat session {} user-to-bot latency {:.3f}s",
            claims.session_id,
            latency_seconds,
        )

    @latency_observer.event_handler("on_first_bot_speech_latency")
    async def _on_first_bot_speech_latency(_observer, latency_seconds):
        recorder.set_latency_metric("firstBotSpeechSecs", latency_seconds)
        logger.info(
            "Pipecat session {} first bot speech latency {:.3f}s",
            claims.session_id,
            latency_seconds,
        )

    @latency_observer.event_handler("on_latency_breakdown")
    async def _on_latency_breakdown(_observer, breakdown):
        events = breakdown.chronological_events()
        recorder.add_latency_breakdown(events)
        for event in events:
            logger.info("Pipecat session {} latency breakdown: {}", claims.session_id, event)

    @startup_observer.event_handler("on_startup_timing_report")
    async def _on_startup_timing_report(_observer, report):
        recorder.set_startup_metric("totalSecs", report.total_duration_secs)

    @startup_observer.event_handler("on_transport_timing_report")
    async def _on_transport_timing_report(_observer, report):
        if report.client_connected_secs is not None:
            recorder.set_startup_metric("clientConnectedSecs", report.client_connected_secs)

    task = PipelineTask(
        pipeline,
        params=PipelineParams(
            audio_in_sample_rate=BROWSER_INPUT_SAMPLE_RATE,
            audio_out_sample_rate=AGENT_OUTPUT_SAMPLE_RATE,
            enable_metrics=True,
            enable_usage_metrics=True,
            send_initial_empty_metrics=False,
        ),
        conversation_id=claims.session_id,
        observers=[startup_observer, latency_observer],
    )

    @transport.event_handler("on_client_connected")
    async def _on_connected(_transport, _client):
        logger.info("Pipecat client connected: {}", claims.session_id)
        # Voice the opening line. It is already seeded into the LLM context
        # above so the model stays consistent with what was spoken. The
        # transcript + browser event are emitted by BrowserEventProcessor
        # when this speech flows through (avoids a double-recorded line).
        agent_turns.start_preview_turn()
        frames: list = [
            OutputTransportMessageUrgentFrame(
                message={"type": "runtime_ready", "sessionId": claims.session_id}
            ),
        ]
        frames.append(TTSSpeakFrame(first_message))
        await task.queue_frames(frames)

    @transport.event_handler("on_client_disconnected")
    async def _on_disconnected(_transport, _client):
        logger.info("Pipecat client disconnected: {}", claims.session_id)
        await task.cancel()

    runner = PipelineRunner(handle_sigint=False)
    status = "completed"
    ended_reason = "client_ended"
    try:
        await runner.run(task)
    except Exception as exc:  # noqa: BLE001 - always report something to Django
        logger.exception("Pipecat session {} failed", claims.session_id)
        status, ended_reason = "failed", str(exc)
    finally:
        if status == "completed" and not recorder.was_client_ended():
            status = "cancelled"
            ended_reason = "client_disconnected"
        if status == "completed":
            await recorder.ensure_backend_started()
        await recorder.notify_backend(status=status, ended_reason=ended_reason)


async def run_pipecat_webrtc_session(
    webrtc_connection,
    claims: SessionClaims,
    context: "SessionContext",
) -> None:
    """Build and run the WebRTC Pipecat pipeline for one browser session."""
    transport = SmallWebRTCTransport(
        webrtc_connection=webrtc_connection,
        params=TransportParams(
            audio_in_enabled=True,
            audio_out_enabled=True,
            audio_in_sample_rate=BROWSER_INPUT_SAMPLE_RATE,
            audio_out_sample_rate=AGENT_OUTPUT_SAMPLE_RATE,
            audio_out_10ms_chunks=2,
            video_in_enabled=False,
            video_out_enabled=False,
        ),
    )

    await _run_transport_session(transport, claims, context)
