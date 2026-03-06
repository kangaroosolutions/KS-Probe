"""OpenAI adapter — supports gpt-5.2, gpt-4.1, and any openai-compatible model."""
from __future__ import annotations

import os
import time
from typing import List

from ks_probe.adapters.base import AdapterResponse, BaseModelAdapter, TokenUsage


class OpenAIAdapter(BaseModelAdapter):
    def __init__(self, model_id: str = "gpt-5.2", **kwargs) -> None:
        super().__init__(model_id)
        try:
            import openai
            self._client = openai.AsyncOpenAI(api_key=os.environ["OPENAI_API_KEY"])
        except ImportError:
            raise ImportError("openai package required: pip install openai")
        self._model_id = model_id

    def count_tokens(self, text: str) -> int:
        try:
            import tiktoken
            enc = tiktoken.encoding_for_model("gpt-4o")  # best approximation for gpt-5 family
            return len(enc.encode(text))
        except Exception:
            return len(text.split())

    async def send_prompt(self, prompt, max_output_tokens=2000, temperature=0.0, top_p=1.0):
        return await self._retry(self._call, [{"role": "user", "content": prompt}],
                                 max_output_tokens, temperature, top_p)

    async def send_conversation(self, messages, max_output_tokens=2000, temperature=0.0, top_p=1.0):
        return await self._retry(self._call, messages, max_output_tokens, temperature, top_p)

    async def _call(self, messages, max_output_tokens, temperature, top_p):
        t0 = time.monotonic()
        resp = await self._client.chat.completions.create(
            model=self._model_id,
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
