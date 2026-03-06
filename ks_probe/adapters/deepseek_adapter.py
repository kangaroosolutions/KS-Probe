"""DeepSeek adapter — supports deepseek-v3.2 and deepseek-r1."""
from __future__ import annotations

import os
import time

from ks_probe.adapters.base import AdapterResponse, BaseModelAdapter, TokenUsage

DEEPSEEK_BASE_URL = "https://api.deepseek.com"

MODEL_MAP = {
    "deepseek-v3.2": "deepseek-chat",
    "deepseek-r1": "deepseek-reasoner",
}


class DeepSeekAdapter(BaseModelAdapter):
    def __init__(self, model_id: str = "deepseek-v3.2", **kwargs) -> None:
        super().__init__(model_id)
        try:
            import openai
            self._client = openai.AsyncOpenAI(
                api_key=os.environ["DEEPSEEK_API_KEY"],
                base_url=DEEPSEEK_BASE_URL,
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
