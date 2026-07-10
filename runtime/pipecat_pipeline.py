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
from pipecat.services.kokoro.tts import KokoroTTSService
from pipecat.services.mistral.llm import MistralLLMService
from pipecat.services.mistral.stt import MistralSTTService
from pipecat.services.mistral.tts import MistralTTSService
from pipecat.transcriptions.language import Language
from pipecat.transports.base_transport import TransportParams
from pipecat.transports.smallwebrtc.transport import SmallWebRTCTransport

from runtime.audio_constants import (
    AGENT_OUTPUT_SAMPLE_RATE,
    BROWSER_INPUT_SAMPLE_RATE,
    KOKORO_TTS_SAMPLE_RATE,
    MISTRAL_TTS_SAMPLE_RATE,
)
from runtime.processors import (
    AgentTurnCoordinator,
    AudioInputDebugProcessor,
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
from runtime.tts_buffer import BufferedKokoroTTSService, BufferedMistralTTSService
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
            "vad_stop_secs={}, vad_start_secs={}, vad_confidence={}, vad_min_volume={}, "
            "user_turn_stop_timeout={}, stt_model={}, llm_model={}, tts_provider={}, "
            "tts_transport={}, kokoro_voice={}"
        ),
        claims.session_id,
        settings.mistral_stt_target_delay_ms,
        settings.vad_stop_secs,
        settings.vad_start_secs,
        settings.vad_confidence,
        settings.vad_min_volume,
        settings.user_turn_stop_timeout,
        settings.mistral_stt_model,
        settings.mistral_llm_model,
        settings.tts_provider,
        settings.tts_transport,
        settings.kokoro_voice,
    )

    # Native streaming STT. Pipecat's MistralSTTService uses the official
    # Mistral realtime SDK path (`/v1/audio/transcriptions/realtime`) and emits
    # interim + final transcription frames, so the user turn can be finalized by
    # Pipecat's normal streaming turn model instead of the old batch-STT shim.
    # Important: pipecat-ai 1.2.1 exposes a `language` setting here, but the
    # installed Mistral realtime SDK connect/update_session methods do not
    # accept a language parameter. Treat realtime STT as multilingual auto-detect
    # in logs/metrics rather than assuming the provider is hard-locked to
    # English.
    stt = MistralSTTService(
        api_key=settings.mistral_api_key,
        sample_rate=BROWSER_INPUT_SAMPLE_RATE,
        target_streaming_delay_ms=settings.mistral_stt_target_delay_ms,
        settings=MistralSTTService.Settings(
            model=settings.mistral_stt_model,
            language=Language.EN if _is_english_session(context.language) else None,
        ),
    )
    logger.info(
        "Pipecat session {} STT language_hint={} effective_realtime_language_mode=auto-detect",
        claims.session_id,
        "en" if _is_english_session(context.language) else "auto",
    )
    # TTS engine (VOICE_RUNTIME_TTS_PROVIDER) and transport
    # (VOICE_RUNTIME_TTS_TRANSPORT) select one of three paths:
    #   kokoro + webrtc — streaming over the WebRTC track (tested production
    #     configuration; in-sync with transcript events, no buffering pause)
    #   kokoro + http   — buffered WAVs over HTTP (fallback)
    #   mistral + http  — buffered Voxtral TTS WAVs over HTTP (fallback; Mistral's
    #     irregular streaming cadence rules out sending it over WebRTC directly)
    # The browser learns the active transport from the runtime_ready message.
    if settings.tts_provider == "kokoro" and settings.tts_transport == "webrtc":
        # Streaming path: plain KokoroTTSService emits audio frames straight to
        # transport.output() (WebRTC), interleaved in-order with the transcript
        # events from BrowserEventProcessor — so audio and text stay in sync and
        # there is no per-utterance buffering pause. No WAVs are written, so the
        # frontend must play the WebRTC track (it's told to via runtime_ready).
        # It resamples Kokoro's native 24k straight to the 48k transport rate.
        tts = KokoroTTSService(
            sample_rate=AGENT_OUTPUT_SAMPLE_RATE,
            text_transforms=[("*", _normalize_tts_text)],
            settings=KokoroTTSService.Settings(
                voice=settings.kokoro_voice,
                language=Language.EN,
            ),
            model_path=settings.kokoro_model_path or None,
            voices_path=settings.kokoro_voices_path or None,
        )
    elif settings.tts_provider == "kokoro":
        tts = BufferedKokoroTTSService(
            sample_rate=KOKORO_TTS_SAMPLE_RATE,
            text_transforms=[("*", _normalize_tts_text)],
            settings=KokoroTTSService.Settings(
                voice=settings.kokoro_voice,
                language=Language.EN,
            ),
            # Point at pre-fetched model files if configured; else auto-download.
            model_path=settings.kokoro_model_path or None,
            voices_path=settings.kokoro_voices_path or None,
            output_sample_rate=AGENT_OUTPUT_SAMPLE_RATE,
            session_id=claims.session_id,
        )
    else:
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
            # Fallback cap for semantic-turn misses. Without this, Pipecat's
            # default 5s wait can leave the browser in a misleading "listening"
            # state after the student has clearly stopped talking.
            user_turn_stop_timeout=settings.user_turn_stop_timeout,
            # "Turn Timeout" — how long of silence before the agent prompts.
            user_idle_timeout=float(context.turn_timeout_secs),
        ),
    )

    @tts.event_handler("on_tts_request")
    async def _on_tts_request(_tts, _context_id, text):
        cleaned = " ".join(str(text or "").split())
        preview = cleaned if len(cleaned) <= 140 else f"{cleaned[:137]}..."
        logger.info(
            "Pipecat session {} TTS request ctx={} chars={} transport={} text={!r}",
            claims.session_id,
            _context_id,
            len(cleaned),
            settings.tts_transport,
            preview,
        )

    user_transcript = UserTranscriptProcessor(recorder)
    client_control = ClientControlProcessor(recorder)
    agent_turns = AgentTurnCoordinator()
    browser_events = BrowserEventProcessor(recorder, agent_turns)

    stages = [
        transport.input(),
        client_control,
        stt,
        user_transcript,
        aggregators.user(),
        llm,
        tts,
        browser_events,
        transport.output(),
        aggregators.assistant(),
    ]
    if settings.llm_input_debug:
        # Debug-only tap that dumps every LLM request context to /tmp; gated
        # because the dumps contain student answers.
        stages.insert(stages.index(llm), LLMInputDebugProcessor(claims.session_id))
    if settings.audio_input_debug:
        # Inbound mic-level telemetry, placed right after the transport so it
        # sees the audio exactly as the VAD/STT will.
        stages.insert(
            stages.index(client_control),
            AudioInputDebugProcessor(
                claims.session_id,
                min_volume=settings.vad_min_volume,
                confidence=settings.vad_confidence,
            ),
        )

    pipeline = Pipeline(stages)

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

    def _runtime_ready_frame() -> OutputTransportMessageUrgentFrame:
        return OutputTransportMessageUrgentFrame(
            message={
                "type": "runtime_ready",
                "sessionId": claims.session_id,
                # Tells the browser how to play agent audio: "webrtc" (play the
                # live WebRTC track) or "http" (fetch pre-rendered WAVs).
                "ttsTransport": settings.tts_transport,
            }
        )

    @transport.event_handler("on_client_connected")
    async def _on_connected(_transport, _client):
        logger.info("Pipecat client connected: {}", claims.session_id)
        # Voice the opening line. It is already seeded into the LLM context
        # above so the model stays consistent with what was spoken. The
        # transcript + browser event are emitted by BrowserEventProcessor
        # when this speech flows through (avoids a double-recorded line).
        agent_turns.start_preview_turn()
        logger.info(
            "Pipecat session {} runtime_ready sent with ttsTransport={}",
            claims.session_id,
            settings.tts_transport,
        )
        await task.queue_frames([_runtime_ready_frame(), TTSSpeakFrame(first_message)])

    @transport.event_handler("on_app_message")
    async def _on_app_message(_transport, message, _sender):
        # The browser announces client_ready once its data channel opens.
        # Answer with a fresh runtime_ready: the copy queued above at
        # on_client_connected is lost whenever ICE connects before the
        # browser's channel arrives — pipecat 1.2.1 never flushes its
        # pre-channel message queue for channels that arrive already open —
        # which leaves the remote audio track muted on real-network links
        # (localhost never loses that race). A reply here cannot be lost:
        # the request itself just arrived over the open channel.
        if isinstance(message, dict) and message.get("type") == "client_ready":
            await task.queue_frames([_runtime_ready_frame()])
            logger.info(
                "Pipecat session {} runtime_ready re-sent after client_ready",
                claims.session_id,
            )

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
