import asyncio
import unittest
from unittest.mock import AsyncMock, Mock, patch

from pipecat.frames.frames import LLMContextFrame, LLMFullResponseEndFrame
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.services.openai.llm import OpenAILLMService

from runtime.turn_recovery import TurnRecovery
from runtime.llm_resilience import ResilientLLMService
from runtime.recorder import SessionRecorder


class RecoveryTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.r = TurnRecovery(Mock(), silence_seconds=0.01)
        self.r.send = AsyncMock()
        self.context = LLMContext(messages=[{"role": "user", "content": "My answer"}])
        self.r.request(self.context)

    async def asyncTearDown(self):
        self.r.close()
        await asyncio.sleep(0)

    async def test_noise_retries_once(self):
        self.r.interrupt()
        await asyncio.sleep(0.04)
        self.assertEqual(self.r.send.await_count, 2)
        self.assertIsInstance(self.r.send.call_args.args[0], LLMContextFrame)
        self.r.interrupt()
        await asyncio.sleep(0.04)
        self.assertEqual(self.r.send.await_count, 2)

    async def test_real_words_cancel_retry(self):
        self.r.interrupt()
        self.r.words()
        await asyncio.sleep(0.04)
        self.r.send.assert_not_awaited()

    async def test_never_retry_while_user_speaking(self):
        self.r.speaking = True
        self.r.interrupt()
        await asyncio.sleep(0.04)
        self.r.send.assert_not_awaited()
        self.r.speaking = False
        self.r.schedule()
        await asyncio.sleep(0.04)
        self.assertEqual(self.r.send.await_count, 2)

    async def test_end_cancels_pending_retry(self):
        self.r.interrupt()
        self.r.close()
        await asyncio.sleep(0.04)
        self.r.send.assert_not_awaited()

    async def test_completed_reply_is_not_replayed(self):
        self.r.pending = False
        self.r.interrupt()
        await asyncio.sleep(0.04)
        self.r.send.assert_not_awaited()

    async def test_new_answer_cancels_stale_retry(self):
        self.r.interrupt()
        self.r.request(LLMContext(messages=[{"role": "user", "content": "New answer"}]))
        await asyncio.sleep(0.04)
        self.r.send.assert_not_awaited()

    async def test_cancelled_model_does_not_flush_partial_word(self):
        # Reproduce base LLM service's response-end in a cancellation finally.
        service = object.__new__(ResilientLLMService)
        with patch.object(OpenAILLMService, "push_frame", new_callable=AsyncMock) as push:
            async def generating():
                try:
                    await asyncio.sleep(10)
                finally:
                    await service.push_frame(LLMFullResponseEndFrame())
            task = asyncio.create_task(generating())
            await asyncio.sleep(0)
            task.cancel()
            with self.assertRaises(asyncio.CancelledError):
                await task
            push.assert_not_awaited()

    async def test_normal_model_completion_is_forwarded(self):
        service = object.__new__(ResilientLLMService)
        with patch.object(OpenAILLMService, "push_frame", new_callable=AsyncMock) as push:
            await service.push_frame(LLMFullResponseEndFrame())
            push.assert_awaited_once()


class ExitTests(unittest.TestCase):
    def test_explicit_end_wins_over_teardown(self):
        recorder = SessionRecorder("test", "http://localhost")
        recorder.record_exit("manual_end", completed=True)
        recorder.record_exit("connection_disconnected")
        self.assertTrue(recorder.was_client_ended())
        self.assertEqual(recorder.client_end_reason, "manual_end")

    def test_navigation_is_not_completed(self):
        recorder = SessionRecorder("test", "http://localhost")
        recorder.record_exit("route_change")
        self.assertFalse(recorder.was_client_ended())
        self.assertEqual(recorder.client_end_reason, "route_change")

    def test_unknown_reason_does_not_log_arbitrary_client_text(self):
        recorder = SessionRecorder("test", "http://localhost")
        recorder.record_exit("untrusted arbitrary text")
        self.assertEqual(recorder.client_end_reason, "client_cancelled")
