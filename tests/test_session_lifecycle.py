import unittest
from unittest.mock import AsyncMock, Mock, patch

from runtime.recorder import SessionRecorder


class SessionTimelineTests(unittest.IsolatedAsyncioTestCase):
    def test_audio_receipt_is_counted_without_claiming_speech_or_transcription(self):
        recorder = SessionRecorder('test', 'http://backend.test')
        recorder.record_input_audio(320)
        recorder.record_input_audio(320)
        health = recorder.metrics['sessionHealth']
        self.assertEqual(health['inputAudioBytes'], 640)
        self.assertEqual(health['events']['first_input_audio'], 1)
        self.assertEqual(recorder.transcript, [])

    def test_timeline_keeps_first_last_and_count_without_transcript_content(self):
        recorder = SessionRecorder('test', 'http://backend.test')
        recorder.health_event('connected')
        recorder.add_transcript('user', 'private test words')
        recorder.add_transcript('user', 'more private words')
        health = recorder.metrics['sessionHealth']
        self.assertEqual(health['events']['transcript_user'], 2)
        self.assertGreaterEqual(health['eventTiming']['transcript_user']['lastSecs'],
                                health['eventTiming']['transcript_user']['firstSecs'])
        self.assertNotIn('private', str(health))

    def test_unknown_disconnect_is_not_claimed_as_a_tab_close(self):
        recorder = SessionRecorder('test', 'http://backend.test')
        recorder.record_exit('page_hidden_exit')
        self.assertEqual(recorder.client_end_reason, 'page_hidden_exit')
        self.assertFalse(recorder.was_client_ended())

    async def test_rejected_activation_is_not_success(self):
        recorder = SessionRecorder('test', 'http://backend.test')
        client = AsyncMock()
        client.post.return_value = Mock()
        client.post.return_value.raise_for_status.side_effect = RuntimeError('expired')
        with patch('runtime.recorder.httpx.AsyncClient') as factory:
            factory.return_value.__aenter__.return_value = client
            self.assertFalse(await recorder.notify_backend_started())
        self.assertFalse(recorder._started_notified)

    async def test_failed_final_callback_can_be_retried(self):
        recorder = SessionRecorder('test', 'http://backend.test')
        client = AsyncMock()
        client.post.return_value = Mock()
        client.post.return_value.raise_for_status.side_effect = RuntimeError('offline')
        with patch('runtime.recorder.httpx.AsyncClient') as factory:
            factory.return_value.__aenter__.return_value = client
            await recorder.notify_backend('cancelled', 'unknown_disconnect')
            self.assertFalse(recorder._notified)
            client.post.return_value.raise_for_status.side_effect = None
            await recorder.notify_backend('cancelled', 'unknown_disconnect')
            self.assertTrue(recorder._notified)
        self.assertEqual(recorder.metrics['sessionHealth']['activity'], 'never_connected')
