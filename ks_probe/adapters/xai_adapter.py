"""xAI adapter — supports grok-4.1-fast (2M context window). OpenAI-compatible endpoint."""
from __future__ import annotations

import os
import time
from typing import List

from ks_probe.adapters.base import AdapterResponse, BaseModelAdapter, TokenUsage

XAI_BASE_URL = "https://api.x.ai/v1"

MODEL_MAP = {
    "grok-4.1-fast": "grok-4-1-fast",
    "grok-4.1": "grok-4-1",
}


class XAIAdapter(BaseModelAdapter):
    def __init__(self, model_id: str = "grok-4.1-fast", **kwargs) -> None:
        super().__init__(model_id)
        try:
            import openai
            self._client = openai.AsyncOpenAI(
                api_key=os.environ["XAI_API_KEY"],
                base_url=XAI_BASE_URL,
            )
        except ImportError:
            raise ImportError("openai package required: pip install openai")
        self._api_model = MODEL_MAP.get(model_id, model_id)

    def count_tokens(self, text: str) -> int:
        try:
            import tiktoken
            enc = tiktoken.get_encoding("cl100k_base")
            return len(enc.encode(text))
        except Exception:
            return max(1, int(len(text.split()) * 1.3))

    async def send_prompt(self, prompt, max_output_tokens=2000, temperature=0.0, top_p=1.0):
        return await self._retry(self._call, [{"role": "user", "content": prompt}],
                                 max_output_tokens, temperature, top_p)

    async def send_conversation(self, messages, max_output_tokens=2000, temperature=0.0, top_p=1.0):
        return await self._retry(self._call, messages, max_output_tokens, temperature, top_p)

    async def _call(self, messages, max_output_tokens, temperature, top_p):
        t0 = time.monotonic()
        resp = await self._client.chat.completions.create(
            model=self._api_model,
            messages=messages,
            max_tokens=max_output_tokens,
            temperature=temperature,
            top_p=top_p,
        )
        latency = int((time.monotonic() - t0) * 1000)
        choice = resp.choices[0]
        return AdapterResponse(
            text=choice.message.content or "",
            usage=TokenUsage(
                input_tokens=resp.usage.prompt_tokens,
                output_tokens=resp.usage.completion_tokens,
                model_version=resp.model,
                latency_ms=latency,
            ),
        )
