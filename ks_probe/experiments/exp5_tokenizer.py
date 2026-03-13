"""
Exp5: Cross-Model Tokenizer Divergence
RQ5 — How much do token counts diverge across model tokenizers for the same text?

This experiment is entirely LOCAL — no API calls required.
Results are written to a CSV (not the DB) since they don't fit ExperimentRun schema.
"""
from __future__ import annotations

import csv
import logging
from pathlib import Path
from typing import Any, Dict, List

from ks_probe.corpus.sampler import DomainSampler, DOMAINS
from ks_probe.corpus.tokenizer_utils import get_tokenizer
from ks_probe.core.config import ExperimentConfig
from ks_probe.core.seed import set_seed
from ks_probe.experiments.base import BaseExperiment
from ks_probe.orchestrator.planner import RunSpec

logger = logging.getLogger(__name__)


class Exp5TokenizerDivergence(BaseExperiment):

    experiment_id = "exp5_tokenizer_divergence"

    def __init__(self, config: ExperimentConfig, **kwargs) -> None:
        super().__init__(config, **kwargs)
        # Force mock mode — no API calls ever
        self.config = config.model_copy(update={"mock_mode": True})

    def build_run_specs(self) -> List[RunSpec]:
        # This experiment doesn't use RunSpec in the normal way;
        # run() is overridden entirely.
        return []

    async def execute_run(self, spec: RunSpec) -> Dict[str, Any]:
        return {}  # Not called — run() is overridden

    def run(self) -> List[Dict[str, Any]]:  # type: ignore[override]
        """Run local tokenizer comparison. No API calls, no cost confirmation."""
        set_seed(self.config.seeds[0] if self.config.seeds else 42)

        tokenizer_ids = self.config.extra.get(
            "tokenizer_ids",
            list(self.config.models),
        )
        n_samples: int = self.config.extra.get("n_samples", 1000)
        tokens_per_sample: int = self.config.extra.get("tokens_per_sample", 500)
        seed: int = self.config.seeds[0] if self.config.seeds else 42

        sampler = DomainSampler(seed=seed)
        results: List[Dict[str, Any]] = []

        for sample_idx in range(n_samples):
            # Cycle through domains for stratified coverage (enables Fig 11 domain breakdown)
            domain = DOMAINS[sample_idx % len(DOMAINS)]
            text = sampler.sample_text(
                target_chars=tokens_per_sample * 4,
                domain_mix={domain: 1.0},
            )

            counts: Dict[str, int] = {}
            for tok_id in tokenizer_ids:
                try:
                    counter = get_tokenizer(tok_id)
                    counts[tok_id] = counter(text)
                except Exception as e:
                    logger.warning("Tokenizer %s failed: %s", tok_id, e)
                    counts[tok_id] = -1

            # Compute pairwise ratios
            for i, tok_a in enumerate(tokenizer_ids):
                for tok_b in tokenizer_ids[i + 1:]:
                    cnt_a = counts.get(tok_a, -1)
                    cnt_b = counts.get(tok_b, -1)
                    ratio = (cnt_a / cnt_b) if (cnt_a > 0 and cnt_b > 0) else None
                    results.append({
                        "sample_idx": sample_idx,
                        "domain": domain,
                        "tokenizer_a": tok_a,
                        "tokenizer_b": tok_b,
                        "count_a": cnt_a,
                        "count_b": cnt_b,
                        "ratio_ab": ratio,
                    })

        self._save_results(results)
        logger.info("[exp5] %d pairwise comparisons computed.", len(results))
        return results

    def _save_results(self, results: List[Dict]) -> None:
        csv_path = self.output_dir / "tokenizer_divergence.csv"
        if not results:
            return
        fieldnames = list(results[0].keys())
        with csv_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(results)
        logger.info("[exp5] Results saved to %s", csv_path)
