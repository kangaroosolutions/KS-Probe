"""vLLM adapter — OpenAI-compatible localhost server for Llama 4 Maverick/Scout."""
from __future__ import annotations

import os
import time
from typing import List

from ks_probe.adapters.base import AdapterResponse, BaseModelAdapter, TokenUsage

MODEL_MAP = {
    "llama-4-maverick": "meta-llama/Llama-4-Maverick-17B-128E-Instruct",
    "llama-4-scout": "meta-llama/Llama-4-Scout-17B-16E-Instruct",
}


class VLLMAdapter(BaseModelAdapter):
    def __init__(self, model_id: str = "llama-4-maverick", **kwargs) -> None:
        super().__init__(model_id)
        base_url = os.environ.get("VLLM_BASE_URL", "http://localhost:8000") + "/v1"
        try:
            import openai
            self._client = openai.AsyncOpenAI(
                api_key="EMPTY",  # vLLM doesn't require a key
                base_url=base_url,
            )
        except ImportError:
            raise ImportError("openai package required: pip install openai")
        self._api_model = MODEL_MAP.get(model_id, model_id)

    def count_tokens(self, text: str) -> int:
        try:
            from transformers import AutoTokenizer
            tok = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3-8B", use_fast=True)
            return len(tok.encode(text))
        except Exception:
            return max(1, int(len(text.split()) * 1.25))

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
