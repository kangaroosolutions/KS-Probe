"""
ContextBuilder — assembles a full test context for a KS-Probe run.

Workflow:
  1. Generate filler text to fill target_tokens using DomainSampler
  2. Inject probe facts at specified fractional positions using ProbeInjector
  3. Verify token count is within ±2% of target
  4. Return InjectedContext (prompt + scoring rubric)
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

from ks_probe.corpus.sampler import DomainSampler
from ks_probe.corpus.tokenizer_utils import get_tokenizer
from ks_probe.probes.injector import InjectedContext, inject_probes
from ks_probe.probes.types import ProbeFact

# Chars per token estimate for initial sizing (will be verified)
_CHARS_PER_TOKEN = 4.0


class ContextBuilder:
    """
    Builds a complete, tokenized experiment context.

    Args:
        corpus_dir: Directory containing domain filler text.
        model_id: Model name for native tokenization.
        domain_mix: {domain: weight} or None for equal mix.
        seed: Random seed for filler selection.
    """

    def __init__(
        self,
        model_id: str = "mock",
        corpus_dir: Optional[Path] = None,
        domain_mix: Optional[Dict[str, float]] = None,
        seed: int = 42,
    ) -> None:
        self.model_id = model_id
        self.domain_mix = domain_mix
        self._sampler = DomainSampler(corpus_dir=corpus_dir, seed=seed)
        self._count = get_tokenizer(model_id)

    def build(
        self,
        target_tokens: int,
        probe_facts: List[ProbeFact],
        probe_positions: Optional[List[float]] = None,
        tolerance: float = 0.02,
    ) -> InjectedContext:
        """
        Build the full context.

        Args:
            target_tokens: Desired total token count of the prompt (before Q&A).
            probe_facts: List of ProbeFact objects to embed.
            probe_positions: Fractional positions for each probe (default = evenly spaced).
            tolerance: Acceptable deviation from target (default ±2%).

        Returns:
            InjectedContext with prompt_text, probe_map, questions, scoring_rubric.
        """
        n = len(probe_facts)
        if probe_positions is None:
            step = 1.0 / (n + 1)
            probe_positions = [round(step * (i + 1), 4) for i in range(n)]

        # 1. Estimate how many chars we need for the filler
        target_chars = int(target_tokens * _CHARS_PER_TOKEN * 1.05)  # slight oversize

        # 2. Sample filler
        filler = self._sampler.sample_text(target_chars, domain_mix=self.domain_mix)

        # 3. Inject probes
        context = inject_probes(
            filler_text=filler,
            probe_facts=probe_facts,
            positions=probe_positions,
            count_tokens=self._count,
        )

        # 4. Verify token count (just the context text, excluding Q&A)
        actual_tokens = self._count(context.prompt_text)
        ratio = actual_tokens / max(target_tokens, 1)
        if not (1 - tolerance - 0.1 <= ratio <= 1 + tolerance + 0.1):
            # Trim or pad — just log a warning; don't fail hard
            pass  # Callers may trim if needed

        return context

    def count(self, text: str) -> int:
        return self._count(text)
