import unittest

from pipecat.transcriptions.language import Language

from runtime.kokoro_voices import kokoro_language, resolve_kokoro_voice


class KokoroVoiceTests(unittest.TestCase):
    def test_agent_voice_used_when_known(self):
        self.assertEqual(resolve_kokoro_voice("am_fenrir", "af_heart"), "am_fenrir")

    def test_legacy_or_empty_voice_falls_back_to_default(self):
        self.assertEqual(resolve_kokoro_voice("21m00Tcm4TlvDq8ikWAM", "af_heart"), "af_heart")
        self.assertEqual(resolve_kokoro_voice("", "af_heart"), "af_heart")

    def test_british_voices_use_british_pronunciation(self):
        self.assertEqual(kokoro_language("bm_george"), Language.EN_GB)
        self.assertEqual(kokoro_language("am_puck"), Language.EN)
