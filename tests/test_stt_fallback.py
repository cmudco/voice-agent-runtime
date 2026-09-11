import asyncio
import unittest
import threading
import inspect
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

from fastapi.testclient import TestClient
from pipecat.frames.frames import CancelFrame, TranscriptionFrame
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor
from pipecat.services.mistral.stt import MistralSTTService

from runtime.stt_resilience import ResilientMistralSTTService, SttLivenessWatchdog
from runtime.whisper_worker import create_app, MAX_AUDIO_BYTES


class FallbackTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.stt = SimpleNamespace(replay=AsyncMock(return_value=False),
                                   activate_fallback=AsyncMock(), publish_fallback=AsyncMock())
        self.client = SimpleNamespace(transcribe=AsyncMock(return_value="Recovered answer"))
        self.events = AsyncMock()
        self.watch = SttLivenessWatchdog("test", self.stt, sample_rate=16000,
                                       deadline_secs=0.001, fallback=self.client,
                                       event_sink=self.events)
        self.tasks = []
        def create(coroutine):
            task = asyncio.create_task(coroutine)
            self.tasks.append(task)
            return task
        self.watch.create_task = create
        self.watch.push_frame = AsyncMock()
        self.watch._awaiting_transcript = True
        self.watch._turn_audio.extend(b"\x01\x00" * 160)

    async def asyncTearDown(self):
        self.watch._closed = True
        for task in self.tasks:
            task.cancel()
        await asyncio.gather(*self.tasks, return_exceptions=True)

    async def test_one_failure_switches_and_replays_once(self):
        await self.watch._run_deadlines()
        await asyncio.sleep(0.01)
        self.assertEqual(self.stt.replay.await_count, 1)
        self.stt.activate_fallback.assert_awaited_once()
        self.stt.publish_fallback.assert_awaited_once()
        self.assertEqual(self.stt.publish_fallback.call_args.args[0], "Recovered answer")
        self.assertFalse(self.watch._turn_audio)
        self.watch.push_frame.assert_not_awaited()

    async def test_healthy_mistral_does_not_switch(self):
        self.watch.notify_transcript(is_final=True)
        await self.watch._run_deadlines()
        self.stt.activate_fallback.assert_not_awaited()

    async def test_manual_switch_is_idempotent_and_acknowledged(self):
        await self.watch._manual_switch("request-1")
        await self.watch._manual_switch("request-2")
        self.stt.activate_fallback.assert_awaited_once()
        self.stt.replay.assert_not_awaited()
        self.assertTrue(self.watch.push_frame.call_args.args[0].message["accepted"])

    async def test_cancel_before_task_wrapper_starts_closes_inner_coroutine(self):
        async def wrapped(coroutine):
            await coroutine
        self.watch.create_task = lambda coroutine: asyncio.create_task(wrapped(coroutine))
        coroutine = self.watch._watch_turn()
        task = self.watch._spawn(coroutine)
        task.cancel()
        await asyncio.gather(task, return_exceptions=True)
        self.assertEqual(inspect.getcoroutinestate(coroutine), inspect.CORO_CLOSED)

    async def test_missing_backup_does_not_cancel_watchdog(self):
        self.watch._fallback = None
        self.watch._cancel_watch = AsyncMock()
        await self.watch._manual_switch("request-1")
        self.watch._cancel_watch.assert_not_awaited()
        self.assertFalse(self.watch.push_frame.call_args.args[0].message["accepted"])

    async def test_close_failure_still_starts_backup(self):
        self.stt.activate_fallback.side_effect = TimeoutError()
        self.assertTrue(await self.watch.switch_to_fallback(reason="test"))
        await asyncio.sleep(0.01)
        self.stt.publish_fallback.assert_awaited_once()

    async def test_reconnect_recovery_does_not_switch(self):
        async def replay(audio):
            self.watch.notify_transcript(is_final=True)
            return True
        self.stt.replay.side_effect = replay
        await self.watch._run_deadlines()
        self.assertEqual(self.stt.replay.await_count, 1)
        self.stt.activate_fallback.assert_not_awaited()

    async def test_keep_listening_while_fallback_transcribes(self):
        await self.watch._run_deadlines()
        self.watch._on_turn_started()
        self.watch._buffer_audio(b"\x02\x00" * 160)
        await asyncio.sleep(0.01)
        self.watch.notify_transcript(is_final=True)
        self.assertTrue(self.watch._turn_audio)
        self.watch._on_turn_stopped()
        await asyncio.sleep(0.01)
        self.assertEqual(self.stt.publish_fallback.await_count, 2)

    async def test_no_replay_mid_speech(self):
        self.watch._speaking = True
        await self.watch._run_deadlines()
        await asyncio.sleep(0.01)
        self.client.transcribe.assert_not_awaited()
        self.watch._speaking = False
        self.watch._on_turn_stopped()
        await asyncio.sleep(0.01)
        self.client.transcribe.assert_awaited_once()

    async def test_truncated_audio_is_not_silently_transcribed(self):
        self.watch._overflow = True
        await self.watch._run_deadlines()
        await asyncio.sleep(0.01)
        self.client.transcribe.assert_not_awaited()
        self.watch.push_frame.assert_awaited_once()

    async def test_fallback_failure_is_visible(self):
        self.client.transcribe.side_effect = TimeoutError()
        await self.watch._run_deadlines()
        await asyncio.sleep(0.01)
        self.watch.push_frame.assert_awaited_once()
        self.stt.publish_fallback.assert_not_awaited()

    async def test_late_upstream_transcript_suppressed(self):
        service = object.__new__(ResilientMistralSTTService)
        service.fallback_active = True
        with patch.object(MistralSTTService, "push_frame", new_callable=AsyncMock) as push:
            await service.push_frame(TranscriptionFrame("late", "", ""))
            push.assert_not_awaited()

    async def test_end_cancels_inflight_inference_and_drops_buffers(self):
        entered = asyncio.Event()
        async def hanging(audio):
            entered.set()
            await asyncio.sleep(60)
        self.client.transcribe.side_effect = hanging
        await self.watch._run_deadlines()
        await entered.wait()
        async def cancel(task):
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)
        self.watch.cancel_task = cancel
        with patch.object(FrameProcessor, "process_frame", new_callable=AsyncMock):
            await self.watch.process_frame(CancelFrame(), FrameDirection.DOWNSTREAM)
        self.assertTrue(self.watch._closed)
        self.assertFalse(self.watch._turn_audio)
        self.assertIsNone(self.watch._fallback_task)
        self.stt.publish_fallback.assert_not_awaited()

    async def test_no_reconnect_once_in_fallback(self):
        service = object.__new__(ResilientMistralSTTService)
        service.fallback_active = True
        service._session_over = False
        with patch.object(MistralSTTService, "_connect", new_callable=AsyncMock) as connect:
            await service._connect()
            self.assertEqual([f async for f in service.run_stt(b"x")], [None])
            connect.assert_not_awaited()


class WorkerTests(unittest.TestCase):
    def test_worker_rejects_overload_without_starting_another_inference(self):
        started, release = threading.Event(), threading.Event()
        def transcribe(*args, **kwargs):
            started.set()
            release.wait(5)
            return iter([SimpleNamespace(text="hello")]), None
        model = SimpleNamespace(transcribe=transcribe)
        with TestClient(create_app(lambda: model, token="test-only")) as client:
            headers = {"Authorization": "Bearer test-only"}
            worker = threading.Thread(target=lambda: client.post("/transcribe", content=b"\0\0", headers=headers))
            worker.start()
            try:
                self.assertTrue(started.wait(3))
                self.assertEqual(client.post("/transcribe", content=b"\0\0", headers=headers).status_code, 503)
            finally:
                release.set()
                worker.join(3)

    def test_auth_limits_and_transcription_contract(self):
        model = SimpleNamespace(transcribe=lambda *args, **kwargs: (
            iter([SimpleNamespace(text=" hello ")]), None))
        with TestClient(create_app(lambda: model, token="test-only")) as client:
            self.assertTrue(client.get("/health").json()["ready"])
            self.assertEqual(client.post("/transcribe", content=b"xx").status_code, 401)
            headers = {"Authorization": "Bearer test-only"}
            self.assertEqual(client.post("/transcribe", content=b"x", headers=headers).status_code, 400)
            self.assertEqual(client.post("/transcribe", content=b"x" * (MAX_AUDIO_BYTES + 2), headers=headers).status_code, 413)
            self.assertEqual(client.post("/transcribe", content=b"\0\0" * 160, headers=headers).json(), {"text": "hello"})
