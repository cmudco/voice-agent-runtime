"""Audio sample-rate constants for the voice runtime."""

BROWSER_INPUT_SAMPLE_RATE = 16000
MISTRAL_TTS_SAMPLE_RATE = 24000  # Mistral Voxtral TTS native rate.
KOKORO_TTS_SAMPLE_RATE = 24000  # Kokoro ONNX native rate.
AGENT_OUTPUT_SAMPLE_RATE = 48000  # WebRTC/Opus RTP clock domain.
