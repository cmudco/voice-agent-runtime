"""Local integration: synthetic WAV -> Pipecat -> HTTP Whisper -> transcript.

Usage: VOICE_RUNTIME_WHISPER_TOKEN=... PYTHONPATH=. python tests/smoke_whisper.py fixture.wav
No upstream API calls: Mistral connections are deliberately replaced with stubs.
"""

import asyncio
import os
import sys
import wave
from unittest.mock import AsyncMock

from pipecat.frames.frames import (
    EndFrame, InputAudioRawFrame, TranscriptionFrame,
    VADUserStartedSpeakingFrame, VADUserStoppedSpeakingFrame,
)
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.runner import PipelineRunner
from pipecat.pipeline.task import PipelineParams, PipelineTask
from pipecat.processors.frame_processor import FrameProcessor

from runtime.stt_resilience import ResilientMistralSTTService, SttLivenessWatchdog
from runtime.whisper_client import WhisperClient


async def main(path):
    with wave.open(path) as wav:
        assert wav.getframerate() == 16000 and wav.getnchannels() == 1 and wav.getsampwidth() == 2
        audio = wav.readframes(wav.getnframes())
    results = []
    done = asyncio.Event()

    class Capture(FrameProcessor):
        async def process_frame(self, frame, direction):
            await super().process_frame(frame, direction)
            if isinstance(frame, TranscriptionFrame):
                results.append(frame.text)
                done.set()
            await self.push_frame(frame, direction)

    stt = ResilientMistralSTTService(api_key="synthetic-test-no-upstream", sample_rate=16000)
    stt._connect = AsyncMock()
    stt._disconnect = AsyncMock()
    stt.replay = AsyncMock(return_value=False)
    watchdog = SttLivenessWatchdog(
        "synthetic-test", stt, sample_rate=16000, deadline_secs=0.01,
        fallback=WhisperClient("http://127.0.0.1:8011", os.environ["VOICE_RUNTIME_WHISPER_TOKEN"]),
    )
    task = PipelineTask(Pipeline([watchdog, stt, Capture()]),
                        params=PipelineParams(audio_in_sample_rate=16000),
                        enable_rtvi=False, idle_timeout_secs=None)
    runner = asyncio.create_task(PipelineRunner(handle_sigint=False).run(task))
    try:
        await task.queue_frames([
            VADUserStartedSpeakingFrame(),
            InputAudioRawFrame(audio=audio, sample_rate=16000, num_channels=1),
            VADUserStoppedSpeakingFrame(),
        ])
        await asyncio.wait_for(done.wait(), timeout=45)
        assert stt.replay.await_count == 2
        assert len(results) == 1 and "translation" in results[0].lower(), results
        print("PASS: two failed recoveries -> local Whisper -> one recovered transcript")
        print(results[0])  # Synthetic fixture only, never run on student recordings.
        done.clear()
        await stt.push_frame(TranscriptionFrame("late upstream duplicate", "", ""))
        await task.queue_frames([
            VADUserStartedSpeakingFrame(),
            InputAudioRawFrame(audio=audio, sample_rate=16000, num_channels=1),
            VADUserStoppedSpeakingFrame(),
        ])
        await asyncio.wait_for(done.wait(), timeout=45)
        assert len(results) == 2 and "translation" in results[1].lower(), results
        assert stt.replay.await_count == 2
        print("PASS: next turn stays on Whisper; late upstream transcript is ignored")
    finally:
        await task.queue_frame(EndFrame())
        await asyncio.wait_for(runner, timeout=10)


if __name__ == "__main__":
    asyncio.run(main(sys.argv[1]))
