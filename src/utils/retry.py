"""Bounded exponential backoff for calls to model-serving endpoints and other remote services."""
from __future__ import annotations

import random
import time
from typing import Callable, Iterable, TypeVar

T = TypeVar("T")


class RetryError(RuntimeError):
    """Raised when every attempt failed; ``last_error`` holds the final exception."""

    def __init__(self, message: str, last_error: BaseException | None = None) -> None:
        super().__init__(message)
        self.last_error = last_error


def retry_call(fn: Callable[[], T], *, attempts: int = 4, base_delay: float = 0.5, max_delay: float = 8.0,
               retry_on: Iterable[type[BaseException]] = (Exception,),
               sleep: Callable[[float], None] = time.sleep) -> T:
    """Call ``fn`` until it succeeds, waiting ``base_delay * 2**n`` (capped at ``max_delay``, with jitter), or the
    server's ``retry_after`` hint when the exception carries one.

    Only exceptions listed in ``retry_on`` are retried; anything else is raised immediately.
    """
    retry_on = tuple(retry_on)
    last: BaseException | None = None
    for attempt in range(attempts):
        try:
            return fn()
        except retry_on as exc:                       # noqa: PERF203 - retry loop
            last = exc
            if attempt == attempts - 1:
                break
            delay = min(max_delay, base_delay * (2 ** attempt)) * (0.5 + random.random() / 2)
            server_hint = getattr(exc, "retry_after", None)        # e.g. HTTP 429 Retry-After header
            if isinstance(server_hint, (int, float)) and server_hint > 0:
                delay = max(delay, min(float(server_hint), max_delay))
            sleep(delay)
    raise RetryError(f"Gave up after {attempts} attempts: {last}", last)
