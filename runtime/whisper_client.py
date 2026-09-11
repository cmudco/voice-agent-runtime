"""Private, bounded HTTP client for the separately hosted Whisper worker."""

import asyncio

import httpx


class WhisperClient:
    def __init__(self, url: str, token: str, timeout: float = 30):
        self.url = url.rstrip("/")
        self.token = token
        self.timeout = timeout

    async def transcribe(self, audio: bytes) -> str:
        # A total deadline also bounds a peer that trickles response bytes.
        async with asyncio.timeout(self.timeout):
            async with httpx.AsyncClient(timeout=self.timeout, trust_env=False) as client:
                response = await client.post(
                    self.url + "/transcribe",
                    content=audio,
                    headers={"Authorization": "Bearer " + self.token,
                             "Content-Type": "application/octet-stream"},
                )
                response.raise_for_status()
                text = response.json()["text"]
                if not isinstance(text, str):
                    raise ValueError("Invalid Whisper response")
                return text.strip()
