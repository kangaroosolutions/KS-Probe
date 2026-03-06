"""
Exp2: Positional Recall Mapping
RQ2 — At a fixed context size, how does PRA vary with probe position (0.0–1.0)?

Design:
  len(models) × len(positions) × len(seeds) runs.
  Each run injects ONE probe fact at a specific fractional position.
"""
from __future__ import annotations

import logging
import time
from typing import Any, Dict, List

from ks_probe.corpus.builder import ContextBuilder
from ks_probe.core.seed import set_seed
from ks_probe.db.queries import RunData
from ks_probe.experiments.base import BaseExperiment
from ks_probe.orchestrator.planner import RunSpec
from ks_probe.scorer.composite import score_all_questions, average_pra

logger = logging.getLogger(__name__)


class Exp2PositionalRecall(BaseExperiment):

    experiment_id = "exp2_positional_recall"

    def build_run_specs(self) -> List[RunSpec]:
        return self._planner.plan(self.config)

    async def execute_run(self, spec: RunSpec) -> Dict[str, Any]:
        set_seed(spec.seed)
        adapter = self._get_adapter(spec.model_id)

        # Use first context size if no per-run size set
        ctx_tokens = spec.context_tokens or (
            self.config.context_sizes[0] if self.config.context_sizes else 50_000
        )
        pos = spec.probe_position if spec.probe_position is not None else 0.5

        # One probe fact injected at the target position
        probes = self._sample_probes(1, spec.seed)
        builder = ContextBuilder(model_id=spec.model_id, seed=spec.seed)
        injected = builder.build(
            target_tokens=ctx_tokens,
            probe_facts=probes,
            probe_positions=[pos],
        )

        t0 = time.monotonic()
        response = await adapter.send_prompt(
            injected.prompt_text,
            max_output_tokens=self.config.max_output_tokens,
            temperature=self.config.temperature,
        )
        latency_ms = int((time.monotonic() - t0) * 1000)

        results = score_all_questions(response.text, injected.scoring_rubric)
        pra = average_pra(results)
        hallucinations = sum(1 for r in results if r.hallucination_flag)
        per_q = {
            r["q_num"]: {"pra": s.pra_score, "exact_match": s.exact_match}
            for r, s in zip(injected.scoring_rubric, results)
        }

        run_data = RunData(
            experiment_name=self.experiment_id,
            model=spec.model_id,
            seed=spec.seed,
            threshold_tokens=ctx_tokens,
            model_version=response.usage.model_version,
            condition="positional",
            probe_position=pos,
            pra_score=pra,
            hallucination_count=hallucinations,
            response_latency_ms=response.usage.latency_ms or latency_ms,
            input_tokens=response.usage.input_tokens,
            output_tokens=response.usage.output_tokens,
            raw_prompt=injected.prompt_text[:2000],
            raw_response=response.text[:2000],
            per_question_scores=per_q,
        )
        run_id = self._write_result(run_data)

        logger.info(
            "[exp2] run_id=%d model=%s pos=%.2f pra=%.3f",
            run_id, spec.model_id, pos, pra,
        )
        return {
            "run_key": spec.run_key,
            "run_id": run_id,
            "pra": pra,
            "probe_position": pos,
            "model_id": spec.model_id,
        }
