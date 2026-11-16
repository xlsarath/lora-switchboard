"""Stdlib logging setup that keeps ``extra={...}`` context visible.

Log lines look like::

    12:00:00 INFO  lora_switchboard.service alias flipped alias=support-bot flip_ms=0.41

or, with ``LSB_LOG_JSON=true``, one JSON object per line. Request and
response bodies are never logged at INFO or above.
"""

from __future__ import annotations

import json
import logging
from datetime import UTC, datetime
from typing import Any

_RESERVED = frozenset(vars(logging.makeLogRecord({})).keys()) | {
    "message",
    "asctime",
    "taskName",
    "color_message",  # uvicorn duplicates the message with ANSI codes
}


def _context(record: logging.LogRecord) -> dict[str, Any]:
    return {k: v for k, v in vars(record).items() if k not in _RESERVED and not k.startswith("_")}


class KeyValueFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        ts = datetime.fromtimestamp(record.created, UTC).strftime("%Y-%m-%dT%H:%M:%S")
        parts = [ts, f"{record.levelname:<5}", record.name, record.getMessage()]
        parts.extend(f"{k}={_fmt(v)}" for k, v in _context(record).items())
        line = " ".join(parts)
        if record.exc_info:
            line += "\n" + self.formatException(record.exc_info)
        return line


class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "ts": datetime.fromtimestamp(record.created, UTC).isoformat(timespec="milliseconds"),
            "level": record.levelname,
            "logger": record.name,
            "msg": record.getMessage(),
            **_context(record),
        }
        if record.exc_info:
            payload["exc"] = self.formatException(record.exc_info)
        return json.dumps(payload, default=str)


def _fmt(value: Any) -> str:
    text = str(value)
    return json.dumps(text) if (" " in text or not text) else text


def configure_logging(level: str = "INFO", *, json_lines: bool = False) -> None:
    handler = logging.StreamHandler()
    handler.setFormatter(JsonFormatter() if json_lines else KeyValueFormatter())
    root = logging.getLogger()
    root.handlers[:] = [handler]
    root.setLevel(level.upper())
    # httpx logs every request line at INFO; that is noise for a proxy.
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)
