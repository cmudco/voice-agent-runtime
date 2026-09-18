"""Local integration: synthetic WAV -> Pipecat -> HTTP Whisper -> transcript.

Usage: VOICE_RUNTIME_WHISPER_TOKEN=... PYTHONPATH=. python tests/smoke_whisper.py fixture.wav [mode]
Modes: auto (stall -> failed replay -> Whisper), manual (stt_switch), direct
(session starts on Whisper), aggregated (auto, through the production user
aggregator: proves a backup transcript that lands after the 2s user-turn stop
timeout still produces exactly one LLM request with the recovered text).
No upstream API calls: Mistral connections are deliberately replaced with stubs.
"""

import asyncio
import os
import sys
import wave
from unittest.mock import AsyncMock

from pipecat.frames.frames import (
    EndFrame, InputAudioRawFrame, InputTransportMessageFrame, LLMContextFrame,
    TranscriptionFrame, VADUserStartedSpeakingFrame, VADUserStoppedSpeakingFrame,
)
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.runner import PipelineRunner
from pipecat.pipeline.task import PipelineParams, PipelineTask
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.processors.aggregators.llm_response_universal import (
    LLMContextAggregatorPair, LLMUserAggregatorParams,
)
from pipecat.processors.frame_processor import FrameProcessor

from runtime.stt_resilience import ResilientMistralSTTService, SttLivenessWatchdog
from runtime.whisper_client import WhisperClient


async def main(path, mode="auto"):
    assert mode in {"auto", "manual", "direct", "aggregated"}
    with wave.open(path) as wav:
        assert wav.getframerate() == 16000 and wav.getnchannels() == 1 and wav.getsampwidth() == 2
        audio = wav.readframes(wav.getnframes())
    results = []
    llm_requests = []
    done = asyncio.Event()

    class Capture(FrameProcessor):
        async def process_frame(self, frame, direction):
            await super().process_frame(frame, direction)
            if isinstance(frame, TranscriptionFrame):
                results.append(frame.text)
                if mode != "aggregated":
                    done.set()
            elif isinstance(frame, LLMContextFrame):
                # The user aggregator consumes transcription frames; the LLM
                # request is the only downstream evidence of the recovered turn.
                llm_requests.append(frame.context.get_messages()[-1]["content"])
                results.append(llm_requests[-1])
                done.set()
            await self.push_frame(frame, direction)

    stt = ResilientMistralSTTService(api_key="synthetic-test-no-upstream", sample_rate=16000,
                                     start_in_fallback=mode == "direct")
    stt._connect = AsyncMock()
    stt._disconnect = AsyncMock()
    stt.replay = AsyncMock(return_value=False)
    watchdog = SttLivenessWatchdog(
        "synthetic-test", stt, sample_rate=16000, deadline_secs=0.01,
        fallback=WhisperClient("http://127.0.0.1:8011", os.environ["VOICE_RUNTIME_WHISPER_TOKEN"]),
        start_in_fallback=mode == "direct",
    )
    stages = [watchdog, stt]
    if mode == "aggregated":
        # Force the backup result to land after the aggregator's 2s stop
        # timeout, which is the case a slow production model will hit.
        real_transcribe = watchdog._fallback.transcribe

        async def late_transcribe(audio):
            await asyncio.sleep(3)
            return await real_transcribe(audio)

        watchdog._fallback = type("LateWhisper", (), {"transcribe": staticmethod(late_transcribe)})()
        # Same aggregator configuration as the exam pipeline: default turn
        # strategies and the 2s stop timeout that a Whisper batch can exceed.
        aggregators = LLMContextAggregatorPair(
            LLMContext(messages=[{"role": "system", "content": "smoke"}]),
            user_params=LLMUserAggregatorParams(user_turn_stop_timeout=2.0),
        )
        stages.append(aggregators.user())
    task = PipelineTask(Pipeline([*stages, Capture()]),
                        params=PipelineParams(audio_in_sample_rate=16000),
                        enable_rtvi=False, idle_timeout_secs=None)
    runner = asyncio.create_task(PipelineRunner(handle_sigint=False).run(task))
    try:
        await task.queue_frames([
            *([InputTransportMessageFrame(message={"type": "stt_switch", "requestId": "synthetic"})]
              if mode == "manual" else []),
            VADUserStartedSpeakingFrame(),
            InputAudioRawFrame(audio=audio, sample_rate=16000, num_channels=1),
            VADUserStoppedSpeakingFrame(),
        ])
        await asyncio.wait_for(done.wait(), timeout=45)
        assert stt.replay.await_count == (1 if mode in {"auto", "aggregated"} else 0)
        assert len(results) == 1 and "translation" in results[0].lower(), results
        print(f"PASS: {mode} -> local Whisper -> one recovered transcript")
        print(results[0])  # Synthetic fixture only, never run on student recordings.
        if mode == "aggregated":
            # Give the aggregator a chance to emit a second (wrong) request.
            await asyncio.sleep(3)
            assert len(llm_requests) == 1 and "translation" in llm_requests[0].lower(), llm_requests
            print("PASS: late backup transcript produced exactly one LLM request with the recovered text")
            return
        done.clear()
        await stt.push_frame(TranscriptionFrame("late upstream duplicate", "", ""))
        await task.queue_frames([
            VADUserStartedSpeakingFrame(),
            InputAudioRawFrame(audio=audio, sample_rate=16000, num_channels=1),
            VADUserStoppedSpeakingFrame(),
        ])
        await asyncio.wait_for(done.wait(), timeout=45)
        assert len(results) == 2 and "translation" in results[1].lower(), results
        assert stt.replay.await_count == (1 if mode == "auto" else 0)
        print("PASS: next turn stays on Whisper; late upstream transcript is ignored")
    finally:
        await task.queue_frame(EndFrame())
        await asyncio.wait_for(runner, timeout=10)


if __name__ == "__main__":
    asyncio.run(main(sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else "auto"))
