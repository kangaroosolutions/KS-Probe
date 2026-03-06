"""
Abstract base class for all model adapters.

Every adapter must implement:
  - send_prompt(prompt, max_output_tokens, temperature=0) → (str, TokenUsage)
  - send_conversation(messages, max_output_tokens, temperature=0) → (str, TokenUsage)
  - count_tokens(text) → int
"""
from __future__ import annotations

import asyncio
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class TokenUsage:
    input_tokens: int = 0
    output_tokens: int = 0
    model_version: str = ""
    latency_ms: int = 0


@dataclass
class AdapterResponse:
    text: str
    usage: TokenUsage


class BaseModelAdapter(ABC):
    """Abstract base for all model adapters."""

    MAX_RETRIES: int = 5
    BASE_BACKOFF: float = 1.0   # seconds

    def __init__(self, model_id: str, **kwargs: Any) -> None:
        self.model_id = model_id

    @abstractmethod
    async def send_prompt(
        self,
        prompt: str,
        max_output_tokens: int = 2000,
        temperature: float = 0.0,
        top_p: float = 1.0,
    ) -> AdapterResponse:
        """Send a single-turn prompt and return the response + usage."""

    @abstractmethod
    async def send_conversation(
        self,
        messages: List[Dict[str, str]],
        max_output_tokens: int = 2000,
        temperature: float = 0.0,
        top_p: float = 1.0,
    ) -> AdapterResponse:
        """Send a multi-turn conversation and return the response + usage."""

    @abstractmethod
    def count_tokens(self, text: str) -> int:
        """Count tokens using this model's native tokenizer."""

    async def _retry(self, coro_fn, *args, **kwargs) -> Any:
        """Execute a coroutine with exponential backoff on rate-limit errors."""
        for attempt in range(self.MAX_RETRIES):
            try:
                return await coro_fn(*args, **kwargs)
            except Exception as e:
                err = str(e).lower()
                is_rate_limit = any(x in err for x in ["429", "rate limit", "too many requests"])
                is_retryable = any(x in err for x in ["503", "502", "timeout", "overloaded"])
                if (is_rate_limit or is_retryable) and attempt < self.MAX_RETRIES - 1:
                    wait = self.BASE_BACKOFF * (2 ** attempt)
                    await asyncio.sleep(wait)
                else:
                    raise
        raise RuntimeError("Max retries exceeded")
