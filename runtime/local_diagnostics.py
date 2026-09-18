"""Opt-in, bounded local diagnostics without storing log bodies or student text."""

import json
import re
from datetime import datetime, timezone
from logging import Formatter, getLogger
from logging.handlers import RotatingFileHandler
from pathlib import Path


def summarize_line(line: str) -> dict:
    """Allowlisted metadata only. Raw lines may contain transcripts or credentials."""
    result = {"at": datetime.now(timezone.utc).isoformat()}
    level = re.search(r"\b(DEBUG|INFO|WARNING|ERROR|CRITICAL)\b", line)
    if level:
        result["level"] = level.group(1)
    session = re.search(r"\bsession[ =:]([a-f0-9]{32})\b", line)
    if session:
        result["sessionId"] = session.group(1)
    event = re.search(r"\bhealth event=([a-z_]{1,80})\b", line)
    if event:
        result["healthEvent"] = event.group(1)
    exception = re.search(r"\b([A-Za-z][A-Za-z0-9_]*(?:Error|Exception))\b", line)
    if exception:
        result["exceptionType"] = exception.group(1)
    response = re.search(r'"(?:GET|POST|PUT|PATCH|DELETE|OPTIONS) [^"\n]+"\s+(\d{3})\b', line)
    if response:
        result["httpStatus"] = int(response.group(1))
    # Never persist arbitrary exception messages, source lines, or request bodies.
    result["lineObserved"] = True
    return result


def make_log(service: str, directory: str):
    folder = Path(directory).resolve()
    folder.mkdir(parents=True, exist_ok=True, mode=0o700)
    handler = RotatingFileHandler(folder / f"{service}.jsonl", maxBytes=5_000_000, backupCount=4)
    Path(handler.baseFilename).chmod(0o600)
    handler.setFormatter(Formatter("%(message)s"))
    log = getLogger(f"local-diagnostics.{service}")
    log.handlers = [handler]
    log.propagate = False
    log.setLevel(20)
    return log


def write_line(log, line: str):
    log.info(json.dumps(summarize_line(line)))
