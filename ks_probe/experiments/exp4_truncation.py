"""
Exp4: Silent Truncation Detection
RQ4 — At what fill level (% of stated max context) do models silently drop content?

Design:
  Probe fact is placed at the END of the context (most likely to be truncated).
  Context is filled to fill_level × model.max_context_tokens.
  If end-placed probes drop to near-zero PRA while mid-placed probes remain
  high, silent truncation is confirmed.

len(models) × len(fill_levels) × len(seeds) runs.
"""
from __future__ import annotations

import logging
import time
from typing import Any, Dict, List

from ks_probe.corpus.builder import ContextBuilder
from ks_probe.core.seed import set_seed
from ks_probe.db.queries import RunData
from ks_probe.experiments.base import BaseExperiment
from ks_probe.orchestrator.planner import RunSpec, make_run_key
from ks_probe.scorer.composite import score_all_questions, average_pra

logger = logging.getLogger(__name__)


class Exp4SilentTruncation(BaseExperiment):

    experiment_id = "exp4_silent_truncation"

    def build_run_specs(self) -> List[RunSpec]:
        """
        fill_levels are stored in config.extra["fill_levels"].
        context_sizes is derived per model from model.max_context_tokens × fill_level.
        """
        fill_levels: List[float] = self.config.extra.get(
            "fill_levels", [0.80, 0.85, 0.90, 0.95, 0.98, 1.00]
        )

        specs: List[RunSpec] = []
        for model_id in self.config.models:
            try:
                max_ctx = self.model_registry.get(model_id).max_context_tokens
            except KeyError:
                max_ctx = 128_000

            for rep_idx, _ in enumerate(self.config.seeds):
                seed = self.seed_manager.get_seed(self.experiment_id, model_id, rep_idx)
                for fill in fill_levels:
                    ctx_tokens = int(max_ctx * fill)
                    run_key = make_run_key(
                        self.experiment_id, model_id, seed,
                        ctx_tokens, None, "end_probe", None,
                    )
                    specs.append(RunSpec(
                        run_key=run_key,
                        experiment_name=self.experiment_id,
                        model_id=model_id,
                        seed=seed,
                        context_tokens=ctx_tokens,
                        condition="end_probe",
                        payload={"fill_level": fill, "max_ctx": max_ctx},
                    ))
        return specs

    async def execute_run(self, spec: RunSpec) -> Dict[str, Any]:
        set_seed(spec.seed)
        adapter = self._get_adapter(spec.model_id)
        ctx_tokens = spec.context_tokens or 100_000

        # Single probe placed at end (position 0.95)
        probes = self._sample_probes(1, spec.seed)
        builder = ContextBuilder(model_id=spec.model_id, seed=spec.seed)
        injected = builder.build(
            target_tokens=ctx_tokens,
            probe_facts=probes,
            probe_positions=[0.95],  # near end
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
        fill_level = spec.payload.get("fill_level", "?")

        run_data = RunData(
            experiment_name=self.experiment_id,
            model=spec.model_id,
            seed=spec.seed,
            threshold_tokens=ctx_tokens,
            model_version=response.usage.model_version,
            condition="end_probe",
            probe_position=0.95,
            pra_score=pra,
            hallucination_count=hallucinations,
            response_latency_ms=response.usage.latency_ms or latency_ms,
            input_tokens=response.usage.input_tokens,
            output_tokens=response.usage.output_tokens,
            raw_response=response.text[:2000],
            per_question_scores={r["q_num"]: s.pra_score for r, s in zip(injected.scoring_rubric, results)},
        )
        run_id = self._write_result(run_data)

        logger.info(
            "[exp4] run_id=%d model=%s fill=%.0f%% pra=%.3f",
            run_id, spec.model_id, float(fill_level) * 100, pra,
        )
        return {
            "run_key": spec.run_key,
            "run_id": run_id,
            "pra": pra,
            "fill_level": fill_level,
            "context_tokens": ctx_tokens,
            "model_id": spec.model_id,
        }
