import asyncio
import unittest
import threading
import inspect
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock, patch

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


class SwitchRaceTests(unittest.IsolatedAsyncioTestCase):
    """Manual and automatic switches racing must still leave a working backup."""

    async def asyncSetUp(self):
        self.activate_entered = asyncio.Event()
        self.activate_release = asyncio.Event()

        async def slow_activate():
            self.activate_entered.set()
            await self.activate_release.wait()

        self.stt = SimpleNamespace(replay=AsyncMock(return_value=False),
                                   activate_fallback=AsyncMock(side_effect=slow_activate),
                                   publish_fallback=AsyncMock())
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

        async def cancel(task):
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)

        self.watch.create_task = create
        self.watch.cancel_task = cancel
        self.watch.push_frame = AsyncMock()
        self.watch._awaiting_transcript = True
        self.watch._turn_audio.extend(b"\x01\x00" * 160)

    async def asyncTearDown(self):
        self.watch._closed = True
        self.activate_release.set()
        for task in self.tasks:
            task.cancel()
        await asyncio.gather(*self.tasks, return_exceptions=True)

    async def test_manual_switch_during_automatic_switch_keeps_a_consumer(self):
        # Automatic path: stall -> replay fails -> switch_to_fallback awaiting upstream close.
        self.watch._watch_task = self.watch._spawn(self.watch._watch_turn())
        await self.activate_entered.wait()
        # Manual request arrives mid-switch.
        manual = self.watch._spawn(self.watch._manual_switch("request-1"))
        await asyncio.sleep(0.01)
        self.activate_release.set()
        await manual
        ack = self.watch.push_frame.call_args.args[0].message
        self.assertTrue(ack["accepted"])
        # The retained turn must actually be transcribed by a live consumer.
        await asyncio.sleep(0.02)
        self.stt.publish_fallback.assert_awaited_once()
        # And the next turn must not pile up unconsumed.
        self.watch._on_turn_started()
        self.watch._buffer_audio(b"\x02\x00" * 160)
        self.watch._speaking = False
        self.watch._on_turn_stopped()
        await asyncio.sleep(0.02)
        self.assertEqual(self.stt.publish_fallback.await_count, 2)
        self.assertTrue(self.watch._fallback_task is not None and not self.watch._fallback_task.done())

    async def test_disconnect_during_manual_switch_leaves_nothing_running(self):
        manual = self.watch._spawn(self.watch._manual_switch("request-1"))
        await self.activate_entered.wait()
        with patch.object(FrameProcessor, "process_frame", new_callable=AsyncMock):
            await self.watch.process_frame(CancelFrame(), FrameDirection.DOWNSTREAM)
        await asyncio.gather(manual, return_exceptions=True)
        self.assertTrue(self.watch._closed)
        self.assertIsNone(self.watch._fallback_task)
        self.assertTrue(self.watch._fallback_queue.empty())
        # No acknowledgement is fabricated for a session that is already gone.
        acks = [c.args[0].message for c in self.watch.push_frame.call_args_list
                if getattr(c.args[0], "message", {}).get("type") == "stt_switch_ack"]
        self.assertEqual(acks, [])
        self.stt.publish_fallback.assert_not_awaited()


class WhisperClientTests(unittest.IsolatedAsyncioTestCase):
    async def test_busy_worker_is_retried_briefly_then_surfaced(self):
        import httpx
        from runtime.whisper_client import WhisperClient

        calls = []

        def handler(request):
            calls.append(dict(request.url.params))
            if len(calls) < 3:
                return httpx.Response(503, text="busy")
            return httpx.Response(200, json={"text": " hello "})

        client = WhisperClient("http://worker", "t", language="en", transport=httpx.MockTransport(handler))
        client.BUSY_RETRY_SECS = 0.001
        self.assertEqual(await client.transcribe(b"\0\0"), "hello")
        self.assertEqual(len(calls), 3)
        self.assertEqual(calls[0], {"language": "en"})

        always_busy = WhisperClient("http://worker", "t",
                                    transport=httpx.MockTransport(lambda r: httpx.Response(503)))
        always_busy.BUSY_RETRY_SECS = 0.001
        with self.assertRaises(httpx.HTTPStatusError):
            await always_busy.transcribe(b"\0\0")

    async def test_non_busy_errors_are_not_retried(self):
        import httpx
        from runtime.whisper_client import WhisperClient

        calls = []
        client = WhisperClient("http://worker", "t", transport=httpx.MockTransport(
            lambda r: (calls.append(1), httpx.Response(500))[1]))
        with self.assertRaises(httpx.HTTPStatusError):
            await client.transcribe(b"\0\0")
        self.assertEqual(len(calls), 1)


class WorkerLanguageTests(unittest.TestCase):
    def test_language_is_pinned_when_supplied_and_validated(self):
        seen = []

        def transcribe(samples, **kwargs):
            seen.append(kwargs.get("language"))
            return iter([SimpleNamespace(text="ok")]), None

        model = SimpleNamespace(transcribe=transcribe)
        with TestClient(create_app(lambda: model, token="test-only")) as client:
            headers = {"Authorization": "Bearer test-only"}
            self.assertEqual(client.post("/transcribe?language=en", content=b"\0\0", headers=headers).status_code, 200)
            self.assertEqual(client.post("/transcribe", content=b"\0\0", headers=headers).status_code, 200)
            self.assertEqual(client.post("/transcribe?language=EN-us", content=b"\0\0", headers=headers).status_code, 400)
        self.assertEqual(seen, ["en", None])


class ProbationTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.stt = SimpleNamespace(replay=AsyncMock(return_value=False),
                                   activate_fallback=AsyncMock(), deactivate_fallback=Mock(),
                                   publish_fallback=AsyncMock())
        self.client = SimpleNamespace(transcribe=AsyncMock(return_value=""))
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

    def _events(self):
        return [c.args[0]["event"] for c in self.events.await_args_list]

    async def test_noise_stall_reverts_to_mistral_and_second_stall_is_permanent(self):
        await self.watch._run_deadlines()
        await asyncio.sleep(0.02)
        self.assertIn("fallback_reverted", self._events())
        self.assertFalse(self.watch._fallback_active)
        self.stt.deactivate_fallback.assert_called_once()
        self.watch.push_frame.assert_not_awaited()  # no "please repeat" for noise
        self.assertTrue(self.watch._progress_task is not None and not self.watch._progress_task.done())
        # Mistral really is dead: the next stall switches for good.
        self.watch._awaiting_transcript = True
        self.watch._turn_audio.extend(b"\x01\x00" * 160)
        self.watch._transcript_event.clear()
        await self.watch._run_deadlines()
        await asyncio.sleep(0.02)
        self.assertTrue(self.watch._fallback_active)
        self.assertEqual(self._events().count("fallback_reverted"), 1)
        self.watch.push_frame.assert_awaited_once()  # empty batch is now an explicit error

    async def test_words_confirm_the_switch(self):
        self.client.transcribe.return_value = "real words"
        await self.watch._run_deadlines()
        await asyncio.sleep(0.02)
        self.assertTrue(self.watch._fallback_active)
        self.assertNotIn("fallback_reverted", self._events())
        self.client.transcribe.return_value = ""
        self.watch._on_turn_started()
        self.watch._buffer_audio(b"\x02\x00" * 160)
        self.watch._on_turn_stopped()
        await asyncio.sleep(0.02)
        self.assertTrue(self.watch._fallback_active)
        self.watch.push_frame.assert_awaited_once()

    async def test_manual_switch_is_never_provisional(self):
        await self.watch._manual_switch("request-1")
        await asyncio.sleep(0.02)
        self.assertTrue(self.watch._fallback_active)
        self.assertNotIn("fallback_reverted", self._events())
        errors = [c for c in self.watch.push_frame.call_args_list
                  if not hasattr(c.args[0], "message")]
        self.assertEqual(len(errors), 1)

    async def test_empty_probation_batch_preserves_queued_answer(self):
        started, release = asyncio.Event(), asyncio.Event()

        async def transcribe(audio):
            if not started.is_set():
                started.set()
                await release.wait()
                return ""
            return "My answer"

        self.client.transcribe.side_effect = transcribe
        await self.watch.switch_to_fallback(reason="recognition_stalled")
        await asyncio.wait_for(started.wait(), timeout=1)
        self.watch._on_turn_started()
        self.watch._buffer_audio(b"\x02\x00" * 160)
        self.watch._on_turn_stopped()
        release.set()
        await asyncio.sleep(0.02)
        self.stt.publish_fallback.assert_awaited_once()
        self.assertEqual(self.stt.publish_fallback.call_args.args[0], "My answer")
        self.stt.deactivate_fallback.assert_not_called()

    async def test_manual_choice_confirms_existing_provisional_switch(self):
        self.watch._speaking = True
        await self.watch.switch_to_fallback(reason="recognition_stalled")
        self.assertTrue(self.watch._probation)
        await self.watch.switch_to_fallback(reason="user_requested")
        self.assertFalse(self.watch._probation)
        self.assertTrue(self.watch._fallback_active)

    async def test_empty_probation_batch_preserves_answer_being_spoken(self):
        started, release = asyncio.Event(), asyncio.Event()

        async def transcribe(audio):
            started.set()
            await release.wait()
            return ""

        self.client.transcribe.side_effect = transcribe
        await self.watch.switch_to_fallback(reason="recognition_stalled")
        await asyncio.wait_for(started.wait(), timeout=1)
        self.watch._speaking = True
        self.watch._on_turn_started()
        self.watch._buffer_audio(b"\x02\x00" * 160)
        release.set()
        await asyncio.sleep(0.02)
        self.assertTrue(self.watch._turn_audio)
        self.assertTrue(self.watch._fallback_active)
        self.stt.deactivate_fallback.assert_not_called()

    async def test_reconnect_wait_scales_with_replayed_audio(self):
        seen = []
        original = asyncio.timeout

        def spy(delay):
            seen.append(delay)
            return original(delay)

        self.watch._turn_audio = bytearray(b"\0" * 16000 * 2 * 10)  # 10s of speech
        with patch("runtime.stt_resilience.asyncio.timeout", spy):
            await self.watch._run_deadlines()
        self.assertEqual(seen[0], 22.0)
