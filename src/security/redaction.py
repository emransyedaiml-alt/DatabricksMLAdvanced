"""Remove secrets and personal data from text before it is logged or stored in audit records."""
from __future__ import annotations

import re

_PATTERNS: list[tuple[re.Pattern[str], str]] = [
    (re.compile(r"dapi[0-9a-f]{32}(-\d+)?"), "[DATABRICKS_TOKEN]"),
    (re.compile(r"gh[pousr]_[A-Za-z0-9]{30,}"), "[GITHUB_TOKEN]"),
    (re.compile(r"(?i)bearer\s+[A-Za-z0-9._\-]{16,}"), "Bearer [TOKEN]"),
    (re.compile(r"(?i)(password|secret|api[_-]?key|token)\s*[:=]\s*\S+"), r"\1=[REDACTED]"),
    (re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}"), "[EMAIL]"),
    (re.compile(r"\b(?:\d[ -]?){13,16}\b"), "[CARD_OR_ACCOUNT]"),
    (re.compile(r"\b\+?\d{1,3}[ -]?\(?\d{3}\)?[ -]?\d{3}[ -]?\d{4}\b"), "[PHONE]"),
]


def redact(text: str) -> str:
    """Return ``text`` with tokens, credentials, e-mail addresses, card-like numbers and phone numbers masked."""
    if not isinstance(text, str):
        return text
    for pattern, replacement in _PATTERNS:
        text = pattern.sub(replacement, text)
    return text


def preview(text: str, limit: int = 80) -> str:
    """A short, redacted preview for logs: never log full document contents."""
    text = redact(" ".join(text.split()))
    return text if len(text) <= limit else text[:limit] + "…"
