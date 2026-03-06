"""Async token-bucket rate limiter for per-model API rate limits."""
from __future__ import annotations

import asyncio
import time


class TokenBucketRateLimiter:
    """
    Async token-bucket rate limiter.
    Enforces a minimum interval between requests based on RPM.
    Thread-safe via asyncio.Lock.
    """

    def __init__(self, requests_per_minute: int) -> None:
        self.rpm = max(1, requests_per_minute)
        self._interval = 60.0 / self.rpm
        self._last_call: float = 0.0
        self._lock = asyncio.Lock()

    async def acquire(self) -> None:
        """Wait until a request slot is available."""
        async with self._lock:
            now = time.monotonic()
            wait = self._interval - (now - self._last_call)
            if wait > 0:
                await asyncio.sleep(wait)
            self._last_call = time.monotonic()
