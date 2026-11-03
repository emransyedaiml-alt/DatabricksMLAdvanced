"""Structured JSON logging with secret and PII redaction.

Every record passes through :func:`src.security.redaction.redact` before it is written, so tokens, e-mail
addresses and similar values never reach the logs even if a caller passes them by mistake.
"""
from __future__ import annotations

import json
import logging
import sys
import time
from typing import Any

from src.security.redaction import redact


class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "ts": time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime(record.created)),
            "level": record.levelname,
            "logger": record.name,
            "event": redact(record.getMessage()),
        }
        fields = getattr(record, "fields", None)
        if isinstance(fields, dict):
            payload.update({k: redact(v) if isinstance(v, str) else v for k, v in fields.items()})
        if record.exc_info:
            payload["error"] = redact(self.formatException(record.exc_info).splitlines()[-1])
        return json.dumps(payload, default=str)


def get_logger(name: str, level: str = "INFO") -> logging.Logger:
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(JsonFormatter())
        logger.addHandler(handler)
        logger.propagate = False
    logger.setLevel(level)
    return logger


def log_event(logger: logging.Logger, event: str, **fields: Any) -> None:
    """Log one structured event: ``log_event(log, "retrieval.done", hits=4, latency_ms=12)``."""
    logger.info(event, extra={"fields": fields})
