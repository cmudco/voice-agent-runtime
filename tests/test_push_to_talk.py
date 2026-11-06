import asyncio
import unittest
from unittest.mock import AsyncMock, MagicMock

from pipecat.frames.frames import (
    BotStartedSpeakingFrame,
    EndFrame,
    InputAudioRawFrame,
    InputTransportMessageFrame,
    OutputTransportMessageUrgentFrame,
    UserStartedSpeakingFrame,
    UserStoppedSpeakingFrame,
    VADUserStartedSpeakingFrame,
    VADUserStoppedSpeakingFrame,
    InterimTranscriptionFrame,
    TranscriptionFrame,
)
from pipecat.processors.frame_processor import FrameDirection

from runtime import push_to_talk as module
from runtime.processors import UserTranscriptProcessor
from runtime.push_to_talk import PushToTalkProcessor
from runtime.recorder import SessionRecorder


def control(kind):
    return InputTransportMessageFrame(message={"type": kind})


class PushToTalkTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self._saved = (module.STOP_TAIL_SECS, module.LAST_FINAL_WAIT_SECS, module.NO_SPEECH_NOTICE_SECS)
        module.STOP_TAIL_SECS, module.LAST_FINAL_WAIT_SECS, module.NO_SPEECH_NOTICE_SECS = 0.01, 0.01, 0.01
        self.p = self._processor()

    async def asyncTearDown(self):
        module.STOP_TAIL_SECS, module.LAST_FINAL_WAIT_SECS, module.NO_SPEECH_NOTICE_SECS = self._saved
        await self.p.process_frame(EndFrame(), FrameDirection.DOWNSTREAM)

    def _processor(self, max_hold_secs=30.0, empty_turn_secs=30.0):
        p = PushToTalkProcessor(max_hold_secs=max_hold_secs, empty_turn_secs=empty_turn_secs)
        p.push_frame = AsyncMock()
        p.broadcast_interruption = AsyncMock()
        p.create_task = asyncio.create_task
        return p

    def pushed(self):
        return [c.args[0] for c in self.p.push_frame.await_args_list]

    def kinds(self, *types):
        return [type(f) for f in self.pushed() if isinstance(f, types)]

    def messages(self):
        return [f.message for f in self.pushed() if isinstance(f, OutputTransportMessageUrgentFrame)]

    async def send(self, frame):
        await self.p.process_frame(frame, FrameDirection.DOWNSTREAM)

    async def test_start_interrupts_and_opens_the_turn(self):
        await self.send(control("ptt_start"))
        self.p.broadcast_interruption.assert_awaited_once()
        self.assertEqual(self.kinds(UserStartedSpeakingFrame), [UserStartedSpeakingFrame])
        self.assertEqual(self.kinds(VADUserStartedSpeakingFrame), [])
        self.assertEqual(self.messages(), [{"type": "ptt_state", "capturing": True}])

    async def test_stop_waits_for_the_last_speech_segment_to_close(self):
        await self.send(control("ptt_start"))
        await self.send(VADUserStartedSpeakingFrame())
        await self.send(control("ptt_stop"))
        await asyncio.sleep(0.05)
        self.assertEqual(self.kinds(UserStoppedSpeakingFrame), [])
        await self.send(VADUserStoppedSpeakingFrame())
        await asyncio.sleep(0.05)
        self.assertEqual(self.kinds(UserStoppedSpeakingFrame), [UserStoppedSpeakingFrame])

    async def test_stop_without_detected_speech_still_flushes_stt(self):
        await self.send(control("ptt_start"))
        await self.send(control("ptt_stop"))
        await asyncio.sleep(0.05)
        self.assertEqual(
            self.kinds(VADUserStoppedSpeakingFrame, UserStoppedSpeakingFrame),
            [VADUserStoppedSpeakingFrame, UserStoppedSpeakingFrame],
        )

    async def test_audio_is_silence_outside_a_turn(self):
        idle = InputAudioRawFrame(audio=b"\x01\x02" * 4, sample_rate=16000, num_channels=1)
        await self.send(idle)
        self.assertEqual(idle.audio, bytes(8))
        await self.send(control("ptt_start"))
        live = InputAudioRawFrame(audio=b"\x01\x02" * 4, sample_rate=16000, num_channels=1)
        await self.send(live)
        self.assertEqual(live.audio, b"\x01\x02" * 4)

    async def test_last_word_passes_during_the_tail(self):
        await self.send(control("ptt_start"))
        await self.send(control("ptt_stop"))
        tail = InputAudioRawFrame(audio=b"\x01\x02", sample_rate=16000, num_channels=1)
        await self.send(tail)
        self.assertEqual(tail.audio, b"\x01\x02")

    async def test_duplicate_start_and_stop_are_idempotent(self):
        await self.send(control("ptt_start"))
        await self.send(control("ptt_start"))
        await self.send(control("ptt_stop"))
        await self.send(control("ptt_stop"))
        await asyncio.sleep(0.05)
        self.assertEqual(len(self.kinds(UserStartedSpeakingFrame)), 1)
        self.assertEqual(len(self.kinds(UserStoppedSpeakingFrame)), 1)

    async def test_pressing_again_during_the_tail_continues_the_answer(self):
        await self.send(control("ptt_start"))
        await self.send(control("ptt_stop"))
        await self.send(control("ptt_start"))
        await asyncio.sleep(0.05)
        self.assertEqual(len(self.kinds(UserStartedSpeakingFrame)), 1)
        self.assertEqual(self.kinds(UserStoppedSpeakingFrame), [])
        self.assertTrue(self.p.capturing)

    async def test_max_hold_ends_the_turn_and_tells_the_browser(self):
        self.p = self._processor(max_hold_secs=0.01)
        await self.send(control("ptt_start"))
        await asyncio.sleep(0.06)
        self.assertEqual(len(self.kinds(UserStoppedSpeakingFrame)), 1)
        self.assertIn({"type": "ptt_auto_stopped"}, self.messages())

    async def test_empty_turn_says_nothing_was_heard_and_recovers(self):
        await self.send(control("ptt_start"))
        await self.send(control("ptt_stop"))
        await asyncio.sleep(0.05)
        self.assertIn({"type": "ptt_no_speech"}, self.messages())
        await self.send(control("ptt_start"))
        self.assertEqual(len(self.kinds(UserStartedSpeakingFrame)), 2)

    async def test_transcript_suppresses_the_empty_notice(self):
        module.NO_SPEECH_NOTICE_SECS = 0.05
        await self.send(control("ptt_start"))
        await self.send(control("ptt_stop"))
        await asyncio.sleep(0.03)
        self.p.notify_final_transcript()
        await asyncio.sleep(0.08)
        self.assertNotIn({"type": "ptt_no_speech"}, self.messages())

    async def test_agent_resuming_suppresses_the_empty_notice(self):
        module.NO_SPEECH_NOTICE_SECS = 0.05
        await self.send(control("ptt_start"))
        await self.send(control("ptt_stop"))
        await asyncio.sleep(0.03)
        await self.send(BotStartedSpeakingFrame())
        await asyncio.sleep(0.08)
        self.assertNotIn({"type": "ptt_no_speech"}, self.messages())

    async def test_messages_after_end_are_ignored(self):
        await self.send(EndFrame())
        await self.send(control("ptt_start"))
        self.assertEqual(self.kinds(UserStartedSpeakingFrame), [])


class PushToTalkTranscriptTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.recorder = SessionRecorder.__new__(SessionRecorder)
        self.recorder.session_id = "s"
        self.recorder.transcript = []
        self.recorder.metrics = {}
        self.recorder.health_event = lambda *a, **k: None
        self.recorder.duration_seconds = lambda: 1.0
        self.recorder._last_user_final_at = None
        self.p = UserTranscriptProcessor(self.recorder, push_to_talk=MagicMock())
        self.p.push_frame = AsyncMock()

    async def send(self, frame):
        await self.p.process_frame(frame, FrameDirection.DOWNSTREAM)

    def sent(self):
        return [
            c.args[0].message
            for c in self.p.push_frame.call_args_list
            if isinstance(c.args[0], OutputTransportMessageUrgentFrame)
        ]

    def interim(self, text):
        return InterimTranscriptionFrame(text=text, user_id="u", timestamp="t")

    def final(self, text):
        return TranscriptionFrame(text=text, user_id="u", timestamp="t")

    async def test_pause_keeps_one_open_bubble_without_processing(self):
        await self.send(UserStartedSpeakingFrame())
        await self.send(self.final("Yeah, last"))
        await self.send(self.interim("week was fun"))

        sent = self.sent()
        self.assertEqual(len({m["turnId"] for m in sent}), 1)
        self.assertEqual(sent[-1]["message"], "Yeah, last week was fun")
        self.assertFalse(any(m.get("isFinal") for m in sent))
        self.assertFalse(any(m["type"] == "processing" for m in sent))

    async def test_done_closes_the_merged_answer_once(self):
        await self.send(UserStartedSpeakingFrame())
        await self.send(self.final("One."))
        await self.send(self.final("Two."))
        await self.send(UserStoppedSpeakingFrame())

        sent = self.sent()
        self.assertEqual(sent[-2]["message"], "One. Two.")
        self.assertTrue(sent[-2]["isFinal"])
        self.assertEqual(sent[-1], {"type": "processing", "isProcessing": True})
        self.assertEqual([t["message"] for t in self.recorder.transcript], ["One. Two."])

    async def test_ending_mid_answer_saves_every_segment(self):
        await self.send(UserStartedSpeakingFrame())
        await self.send(self.final("One."))
        await self.send(self.final("Two."))

        self.assertEqual([t["message"] for t in self.recorder.transcript], ["One. Two."])

    async def test_late_final_joins_the_closed_answer(self):
        await self.send(UserStartedSpeakingFrame())
        await self.send(self.final("One."))
        await self.send(UserStoppedSpeakingFrame())
        await self.send(self.final("Two."))

        turn_ids = {m["turnId"] for m in self.sent() if m["type"] == "transcript_update"}
        self.assertEqual(len(turn_ids), 1)
        self.assertEqual([t["message"] for t in self.recorder.transcript], ["One. Two."])

    async def test_late_interim_does_not_reopen_the_answer(self):
        await self.send(UserStartedSpeakingFrame())
        await self.send(self.final("One."))
        await self.send(UserStoppedSpeakingFrame())
        count = len(self.sent())
        await self.send(self.interim("stray"))

        self.assertEqual(len(self.sent()), count)

    async def test_final_after_done_with_no_earlier_words_still_replies(self):
        await self.send(UserStartedSpeakingFrame())
        await self.send(UserStoppedSpeakingFrame())
        self.assertEqual(self.sent(), [])
        await self.send(self.final("One."))

        self.assertTrue(self.sent()[-2]["isFinal"])
        self.assertEqual(self.sent()[-1]["type"], "processing")

    async def test_next_press_starts_a_new_bubble(self):
        await self.send(UserStartedSpeakingFrame())
        await self.send(self.final("One."))
        await self.send(UserStoppedSpeakingFrame())
        await self.send(UserStartedSpeakingFrame())
        await self.send(self.final("Two."))

        turn_ids = [m["turnId"] for m in self.sent() if m["type"] == "transcript_update"]
        self.assertEqual(len(set(turn_ids)), 2)
        self.assertEqual([t["message"] for t in self.recorder.transcript], ["One.", "Two."])
