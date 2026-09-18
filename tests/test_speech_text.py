import unittest
from types import SimpleNamespace

from runtime.speech_text import SpeechTextGuard
from runtime.processors import ElapsedTimestampProcessor


class SpeechTextTests(unittest.TestCase):
    def test_fragmented_header_never_reaches_speech(self):
        guard = SpeechTextGuard()
        output = ''.join(guard.feed(chunk) for chunk in [' ', '[0', ':35]', ' (age', 'nt)', ' Thank you.'])
        self.assertEqual(output, ' Thank you.')

    def test_header_only_reply_does_not_emit_text(self):
        guard = SpeechTextGuard()
        self.assertEqual(guard.feed('[0:35] (agent)'), '')
        self.assertEqual(guard.feed('', final=True), '')

    def test_normal_text_and_brackets_preserved(self):
        for text in ['Thank you.', '[An example] is useful.', 'We meet at [0:35].']:
            guard = SpeechTextGuard()
            self.assertEqual(guard.feed(text) + guard.feed('', final=True), text)

    def test_timing_note_is_replaced_and_dialogue_is_not_modified(self):
        recorder = SimpleNamespace(duration_seconds=lambda: 35, transcript=[
            {'role': 'user', 'message': 'Hello', 'timeInCallSecs': 12}])
        processor = ElapsedTimestampProcessor(recorder)
        messages = [{'role': 'system', 'content': 'Exam instructions'},
                    {'role': 'user', 'content': 'Hello'}, {'role': 'assistant', 'content': 'Welcome'}]
        result = processor._stamp(messages)
        again = processor._stamp(result)
        self.assertEqual(len(again), 4)
        self.assertEqual(messages[1]['content'], 'Hello')
        self.assertEqual(messages[2]['content'], 'Welcome')
        self.assertIn('"currentElapsedSeconds": 35', again[0]['content'])
        self.assertNotIn('"message":', again[0]['content'])
