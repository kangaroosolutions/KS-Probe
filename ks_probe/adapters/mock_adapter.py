"""
MockAdapter — deterministic responses with no API calls.

Scans the prompt for embedded probe fact IDs ([PROBE:PF-XXX]) and returns
gold answers for all questions found. A configurable noise level simulates
realistic PRA < 100%.
"""
from __future__ import annotations

import random
import re
import time
from typing import Dict, List, Optional

from ks_probe.adapters.base import AdapterResponse, BaseModelAdapter, TokenUsage
from ks_probe.probes.loader import load_probe_pool


class MockAdapter(BaseModelAdapter):
    """
    Returns gold answers for probe questions found in the prompt.

    noise_level (0.0–1.0): fraction of answers intentionally wrong.
    seed: for reproducibility of noise injection.
    """

    def __init__(
        self,
        model_id: str = "mock",
        noise_level: float = 0.1,
        seed: int = 42,
        words_per_token: float = 0.75,
        **kwargs,
    ) -> None:
        super().__init__(model_id)
        self.noise_level = noise_level
        self._rng = random.Random(seed)
        self._words_per_token = words_per_token

    def count_tokens(self, text: str) -> int:
        return max(1, int(len(text.split()) / self._words_per_token))

    def _build_response(self, prompt: str) -> str:
        """Find probe IDs in prompt, look up gold answers, return Q&A block."""
        pool = load_probe_pool()
        probe_ids = re.findall(r"\[PROBE:(PF-\d+)\]", prompt)

        # Build lookup: question text → gold answer
        gold_map: Dict[str, str] = {}
        for pid in probe_ids:
            try:
                pf = pool.get(pid)
                for q in pf.questions:
                    gold_map[q.q] = q.gold
            except KeyError:
                pass

        # Parse questions from prompt (look for "Q\d+: ...")
        q_matches = re.findall(r"Q(\d+): (.+?)\nAnswer:", prompt)

        lines = []
        for q_num, q_text in q_matches:
            gold = gold_map.get(q_text.strip())
            if gold and self._rng.random() > self.noise_level:
                lines.append(f"Q{q_num}: {gold}")
            else:
                lines.append(f"Q{q_num}: I don't have enough information to answer that.")

        return "\n".join(lines) if lines else "I cannot find the relevant information."

    async def send_prompt(
        self,
        prompt: str,
        max_output_tokens: int = 2000,
        temperature: float = 0.0,
        top_p: float = 1.0,
    ) -> AdapterResponse:
        t0 = time.monotonic()
        text = self._build_response(prompt)
        latency = int((time.monotonic() - t0) * 1000)
        return AdapterResponse(
            text=text,
            usage=TokenUsage(
                input_tokens=self.count_tokens(prompt),
                output_tokens=self.count_tokens(text),
                model_version="mock-v1",
                latency_ms=latency,
            ),
        )

    async def send_conversation(
        self,
        messages: List[dict],
        max_output_tokens: int = 2000,
        temperature: float = 0.0,
        top_p: float = 1.0,
    ) -> AdapterResponse:
        full_text = "\n".join(m.get("content", "") for m in messages)
        return await self.send_prompt(full_text, max_output_tokens, temperature, top_p)
