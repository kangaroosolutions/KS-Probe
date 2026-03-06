"""Anthropic adapter — supports claude-opus-4-6, claude-sonnet-4-6."""
from __future__ import annotations

import os
import time
from typing import List

from ks_probe.adapters.base import AdapterResponse, BaseModelAdapter, TokenUsage

# Model ID mapping: our short IDs → Anthropic API IDs
MODEL_MAP = {
    "claude-opus-4-6": "claude-opus-4-6",
    "claude-sonnet-4-6": "claude-sonnet-4-6",
}


class AnthropicAdapter(BaseModelAdapter):
    def __init__(self, model_id: str = "claude-sonnet-4-6", **kwargs) -> None:
        super().__init__(model_id)
        try:
            import anthropic
            self._client = anthropic.AsyncAnthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
        except ImportError:
            raise ImportError("anthropic package required: pip install anthropic")
        self._api_model = MODEL_MAP.get(model_id, model_id)

    def count_tokens(self, text: str) -> int:
        # Anthropic uses ~1.3 tokens/word for typical English text
        return max(1, int(len(text.split()) * 1.3))

    async def send_prompt(self, prompt, max_output_tokens=2000, temperature=0.0, top_p=1.0):
        messages = [{"role": "user", "content": prompt}]
        return await self._retry(self._call, messages, max_output_tokens, temperature)

    async def send_conversation(self, messages, max_output_tokens=2000, temperature=0.0, top_p=1.0):
        return await self._retry(self._call, messages, max_output_tokens, temperature)

    async def _call(self, messages, max_output_tokens, temperature):
        t0 = time.monotonic()
        resp = await self._client.messages.create(
            model=self._api_model,
            max_tokens=max_output_tokens,
            temperature=temperature,
            messages=messages,
        )
        latency = int((time.monotonic() - t0) * 1000)
        text = "".join(b.text for b in resp.content if hasattr(b, "text"))
        return AdapterResponse(
            text=text,
            usage=TokenUsage(
                input_tokens=resp.usage.input_tokens,
                output_tokens=resp.usage.output_tokens,
                model_version=resp.model,
                latency_ms=latency,
            ),
        )
