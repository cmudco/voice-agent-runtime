import asyncio
import unittest
from unittest.mock import Mock, patch

from runtime.health import linux_kb_values, monitor_session


class HealthTests(unittest.IsolatedAsyncioTestCase):
    async def test_sampling_is_bounded_and_stops(self):
        recorder = Mock(metrics={})
        with patch("runtime.health.linux_kb_values", return_value={
            "MemAvailable": 100, "SwapFree": 10, "VmRSS": 200,
        }):
            task = asyncio.create_task(monitor_session(recorder, interval=0.01))
            await asyncio.sleep(0.03)
            task.cancel()
            await task
        health = recorder.metrics["sessionHealth"]
        self.assertEqual(health["minAvailableMemoryKb"], 100)
        self.assertEqual(health["maxRuntimeRssKb"], 200)
        self.assertGreaterEqual(health["maxEventLoopLagSecs"], 0)

    def test_non_linux_is_supported(self):
        self.assertEqual(linux_kb_values("/nonexistent-voice-test-proc"), {})
