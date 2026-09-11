import unittest
from unittest.mock import AsyncMock

from runtime.speech_policy import SpeechProgress, SpeechRecoveryPolicy
from runtime.voice_stage import VoiceStage, VoiceStageTracker
from runtime.local_diagnostics import summarize_line
from runtime.sentry import _local_event


class ProgressTests(unittest.TestCase):
    def test_silence_and_raw_noise_are_not_stalls(self):
        progress = SpeechProgress(SpeechRecoveryPolicy())
        progress.begin(0)
        progress.audio(30, 30, speaking=False)
        self.assertFalse(progress.stalled(30))

    def test_continuous_speech_without_progress_is_bounded(self):
        progress = SpeechProgress(SpeechRecoveryPolicy())
        progress.begin(0)
        progress.audio(12, 12, speaking=True)
        self.assertTrue(progress.stalled(12))
        self.assertTrue(progress.transcript(12, "hello"))
        progress.audio(24, 12, speaking=True)
        self.assertFalse(progress.transcript(24, "hello"))
        self.assertTrue(progress.stalled(24))

    def test_fresh_turn_does_not_inherit_idle_time(self):
        progress = SpeechProgress(SpeechRecoveryPolicy())
        progress.begin(0)
        progress.transcript(2, "yes")
        progress.begin(90, fresh=True)
        progress.audio(92, 2, speaking=True)
        self.assertFalse(progress.stalled(92))

    def test_no_audio_arriving_is_not_attributed_to_stt(self):
        progress = SpeechProgress(SpeechRecoveryPolicy())
        progress.begin(0)
        progress.audio(3, 3, speaking=True)
        self.assertFalse(progress.stalled(30))

    def test_local_diagnostics_drop_payload_and_token(self):
        summary = summarize_line('ERROR ValueError: secret student words token=private')
        self.assertEqual(summary['exceptionType'], 'ValueError')
        self.assertNotIn('secret', str(summary))
        self.assertNotIn('private', str(summary))

    def test_local_sentry_drops_payloads_and_frame_variables(self):
        event = _local_event({'request': {'token': 'secret'}, 'extra': {'transcript': 'private'},
                             'exception': {'values': [{'type': 'ValueError', 'value': 'private',
                             'stacktrace': {'frames': [{'lineno': 2, 'vars': {'key': 'secret'}}]}}]}}, {})
        self.assertNotIn('secret', str(event))
        self.assertNotIn('private', str(event))
        self.assertIn('ValueError', str(event))


class StageTests(unittest.IsolatedAsyncioTestCase):
    async def test_dedup_and_sequence(self):
        emit = AsyncMock()
        tracker = VoiceStageTracker(emit)
        await tracker.set(VoiceStage.THINKING)
        await tracker.set(VoiceStage.THINKING)
        await tracker.set(VoiceStage.PREPARING_VOICE)
        self.assertEqual(emit.await_count, 2)
        self.assertEqual(tracker.sequence, 2)

    async def test_speaking_and_recovery_override_generation(self):
        tracker = VoiceStageTracker(AsyncMock())
        tracker.speaking = True
        await tracker.set(VoiceStage.PREPARING_VOICE)
        self.assertEqual(tracker.stage, VoiceStage.SPEAKING)
        await tracker.recognition_event('stalled')
        self.assertEqual(tracker.stage, VoiceStage.RECOVERING)
        tracker.closed = True
        await tracker.set(VoiceStage.LISTENING)
        self.assertEqual(tracker.stage, VoiceStage.RECOVERING)
