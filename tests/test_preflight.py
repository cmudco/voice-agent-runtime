import unittest
from types import SimpleNamespace
import httpx

from runtime.preflight import check_backup


class BackupPreflightTests(unittest.IsolatedAsyncioTestCase):
    async def test_missing_configuration_blocks_deployment(self):
        with self.assertRaisesRegex(RuntimeError, 'not configured'):
            await check_backup(SimpleNamespace(whisper_url='', whisper_token=''))

    async def test_requires_ready_worker_and_authenticated_inference(self):
        requests = []
        def handler(request):
            requests.append(request)
            if request.url.path == '/health':
                return httpx.Response(200, json={'ready': True})
            self.assertEqual(request.headers['Authorization'], 'Bearer test-token')
            self.assertEqual(len(request.content), 32000)
            return httpx.Response(200, json={'text': ''})
        await check_backup(SimpleNamespace(whisper_url='http://worker.test', whisper_token='test-token'),
                           transport=httpx.MockTransport(handler))
        self.assertEqual(len(requests), 2)

    async def test_invalid_worker_credentials_fail_the_gate(self):
        transport = httpx.MockTransport(lambda r: httpx.Response(200, json={'ready': True})
                                       if r.url.path == '/health' else httpx.Response(401))
        with self.assertRaises(httpx.HTTPStatusError):
            await check_backup(SimpleNamespace(whisper_url='http://worker.test', whisper_token='bad'), transport)
