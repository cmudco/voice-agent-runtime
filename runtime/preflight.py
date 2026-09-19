"""Explicit deployment gate; no credentials or student audio are printed.

Run with the runtime's production environment BEFORE restarting it. This makes
one authenticated, synthetic-silence inference request to the configured worker.
It is not a throughput/load test or proof of transcription accuracy.
"""
import asyncio

import httpx

from runtime.settings import get_settings
from runtime.whisper_client import WhisperClient


async def check_backup(settings, transport=None):
    if not settings.whisper_url or not settings.whisper_token:
        raise RuntimeError('Backup speech recognition is not configured; deployment blocked.')
    async with httpx.AsyncClient(timeout=5, trust_env=False, transport=transport) as client:
        response = await client.get(settings.whisper_url.rstrip('/') + '/health')
        response.raise_for_status()
        if response.json().get('ready') is not True:
            raise RuntimeError('Backup model is not ready; deployment blocked.')
    await WhisperClient(settings.whisper_url, settings.whisper_token, timeout=15,
                        transport=transport).transcribe(bytes(16000 * 2))


if __name__ == '__main__':
    asyncio.run(check_backup(get_settings()))
    print('Backup configured, model loaded, and authenticated inference passed.')
