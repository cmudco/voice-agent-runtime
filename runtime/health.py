"""Small, content-free resource samples correlated with a session's lifetime."""

import asyncio
import time
from pathlib import Path


def linux_kb_values(path):
    try:
        return {parts[0].rstrip(":"): int(parts[1])
                for line in Path(path).read_text().splitlines()
                if len(parts := line.split()) == 3 and parts[2] == "kB"}
    except (OSError, ValueError):
        return {}


async def monitor_session(recorder, interval=10.0):
    health = recorder.metrics.setdefault("sessionHealth", {})
    expected = time.monotonic()
    try:
        while True:
            now = time.monotonic()
            lag = round(max(0, now - expected), 3)
            health["maxEventLoopLagSecs"] = max(health.get("maxEventLoopLagSecs", 0), lag)
            memory = linux_kb_values("/proc/meminfo")
            process = linux_kb_values("/proc/self/status")
            if "MemAvailable" in memory:
                health["minAvailableMemoryKb"] = min(
                    health.get("minAvailableMemoryKb", memory["MemAvailable"]),
                    memory["MemAvailable"],
                )
                health["lastSwapFreeKb"] = memory.get("SwapFree")
            if "VmRSS" in process:
                health["maxRuntimeRssKb"] = max(health.get("maxRuntimeRssKb", 0), process["VmRSS"])
            expected = time.monotonic() + interval
            await asyncio.sleep(interval)
    except asyncio.CancelledError:
        pass
