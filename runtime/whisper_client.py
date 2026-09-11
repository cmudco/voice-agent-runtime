"""Private, bounded HTTP client for the separately hosted Whisper worker."""

import asyncio

import httpx


class WhisperClient:
    # The worker admits one inference at a time and answers 503 to the rest.
    # Two students pausing at once must not cost one of them their answer, so
    # wait briefly for the slot; the total deadline still bounds everything.
    BUSY_RETRIES = 4
    BUSY_RETRY_SECS = 1.0

    def __init__(self, url: str, token: str, timeout: float = 30, *, language: str | None = None,
                 transport: httpx.AsyncBaseTransport | None = None):
        self.url = url.rstrip("/")
        self.token = token
        self.timeout = timeout
        self.language = language
        self._transport = transport

    async def transcribe(self, audio: bytes) -> str:
        params = {"language": self.language} if self.language else None
        # A total deadline also bounds a peer that trickles response bytes.
        async with asyncio.timeout(self.timeout):
            async with httpx.AsyncClient(timeout=self.timeout, trust_env=False,
                                         transport=self._transport) as client:
                for attempt in range(self.BUSY_RETRIES + 1):
                    response = await client.post(
                        self.url + "/transcribe",
                        content=audio,
                        params=params,
                        headers={"Authorization": "Bearer " + self.token,
                                 "Content-Type": "application/octet-stream"},
                    )
                    if response.status_code == 503 and attempt < self.BUSY_RETRIES:
                        await asyncio.sleep(self.BUSY_RETRY_SECS)
                        continue
                    response.raise_for_status()
                    text = response.json()["text"]
                    if not isinstance(text, str):
                        raise ValueError("Invalid Whisper response")
                    return text.strip()
        raise RuntimeError("unreachable")
