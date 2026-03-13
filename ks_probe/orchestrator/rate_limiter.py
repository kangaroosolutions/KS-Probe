"""Async rate limiter supporting both request-rate and token-budget limits.

For providers with input-token-per-minute limits (e.g. Anthropic Tier-1 at 30K/min),
the token-budget mode waits after each call proportional to the tokens consumed,
ensuring the next call doesn't breach the provider's sliding-window token limit.
"""
from __future__ import annotations

import asyncio
import time


class TokenBucketRateLimiter:
    """
    Async rate limiter with dual-mode support:
      - requests_per_minute: classic RPM cap (minimum interval between calls)
      - tokens_per_minute:   token-budget cap (wait after each call proportional
                             to the tokens it consumed)

    Both can be active simultaneously; the longer wait wins.
    Thread-safe via asyncio.Lock.
    """

    def __init__(
        self,
        requests_per_minute: float = 0,
        tokens_per_minute: int = 0,
    ) -> None:
        self._request_interval = (
            60.0 / max(0.01, requests_per_minute)
            if requests_per_minute > 0 else 0.0
        )
        self._tpm = tokens_per_minute
        self._last_call: float = 0.0
        self._last_tokens: int = 0
        self._lock = asyncio.Lock()

    async def acquire(self, estimated_tokens: int = 0) -> None:
        """Wait until a request slot is available.

        Args:
            estimated_tokens: Expected input token count for the upcoming call.
                Used to calculate token-budget wait time based on the
                PREVIOUS call's consumption.
        """
        async with self._lock:
            now = time.monotonic()
            elapsed = (now - self._last_call) if self._last_call > 0 else float("inf")

            # Request-rate wait
            wait = max(0.0, self._request_interval - elapsed)

            # Token-budget wait: ensure enough time has passed since the
            # previous call for its tokens to clear the rate-limit window
            if self._tpm > 0 and self._last_tokens > 0:
                token_interval = self._last_tokens / self._tpm * 60.0
                token_wait = max(0.0, token_interval - elapsed)
                wait = max(wait, token_wait)

            if wait > 0:
                await asyncio.sleep(wait)

            self._last_call = time.monotonic()
            # Track this call's tokens so the NEXT call waits appropriately.
            # If estimated_tokens is 0 (e.g. exp3 with no context_tokens),
            # use a conservative fallback of half the token budget.
            if estimated_tokens > 0:
                self._last_tokens = estimated_tokens
            elif self._tpm > 0:
                self._last_tokens = self._tpm // 2
            else:
                self._last_tokens = 0
