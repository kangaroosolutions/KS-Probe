"""Google Gemini adapter — supports gemini-3.1-pro, gemini-2.5-pro."""
from __future__ import annotations

import os
import time
from typing import List

from ks_probe.adapters.base import AdapterResponse, BaseModelAdapter, TokenUsage

MODEL_MAP = {
    "gemini-3.1-pro": "gemini-3.1-pro",
    "gemini-2.5-pro": "gemini-2.5-pro",
    "gemini-1.5-pro": "gemini-1.5-pro",
}


class GoogleAdapter(BaseModelAdapter):
    def __init__(self, model_id: str = "gemini-3.1-pro", **kwargs) -> None:
        super().__init__(model_id)
        try:
            import google.generativeai as genai
            genai.configure(api_key=os.environ["GOOGLE_API_KEY"])
            self._genai = genai
        except ImportError:
            raise ImportError("google-generativeai required: pip install google-generativeai")
        self._api_model = MODEL_MAP.get(model_id, model_id)

    def count_tokens(self, text: str) -> int:
        # Gemini tokenizes slightly more than GPT-4 for English text
        return max(1, int(len(text.split()) * 1.35))

    async def send_prompt(self, prompt, max_output_tokens=2000, temperature=0.0, top_p=1.0):
        return await self._retry(self._call, prompt, max_output_tokens, temperature)

    async def send_conversation(self, messages, max_output_tokens=2000, temperature=0.0, top_p=1.0):
        # Flatten conversation to a single prompt for Gemini
        flat = "\n".join(f"{m['role'].upper()}: {m['content']}" for m in messages)
        return await self.send_prompt(flat, max_output_tokens, temperature, top_p)

    async def _call(self, prompt, max_output_tokens, temperature):
        import asyncio
        t0 = time.monotonic()
        model = self._genai.GenerativeModel(
            model_name=self._api_model,
            generation_config=self._genai.GenerationConfig(
                temperature=temperature,
                max_output_tokens=max_output_tokens,
            ),
        )
        # Run synchronous SDK in executor to avoid blocking
        loop = asyncio.get_event_loop()
        resp = await loop.run_in_executor(None, model.generate_content, prompt)
        latency = int((time.monotonic() - t0) * 1000)
        text = resp.text if hasattr(resp, "text") else ""
        # Token counts from usage_metadata when available
        usage = getattr(resp, "usage_metadata", None)
        input_tokens = getattr(usage, "prompt_token_count", self.count_tokens(prompt))
        output_tokens = getattr(usage, "candidates_token_count", self.count_tokens(text))
        return AdapterResponse(
            text=text,
            usage=TokenUsage(
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                model_version=self._api_model,
                latency_ms=latency,
            ),
        )
