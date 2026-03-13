"""
Exp3: Multi-Turn Conversational Degradation
RQ3 — How does PRA change as the conversation grows from 10 to 120 turns?

Design:
  - Turn 1: Inject context with probes as a "document" in the user message.
  - Turns 2..N: Filler conversation (user asks generic Q, assistant gives filler A).
  - Final user message: Ask all probe questions.
  - Compares against a single-prompt baseline.

len(models) × len(turns) × len(seeds) runs.
"""
from __future__ import annotations

import logging
import time
from typing import Any, Dict, List

from ks_probe.corpus.builder import ContextBuilder
from ks_probe.corpus.sampler import DomainSampler
from ks_probe.core.seed import set_seed
from ks_probe.db.queries import RunData
from ks_probe.experiments.base import BaseExperiment
from ks_probe.orchestrator.planner import RunSpec
from ks_probe.scorer.composite import score_all_questions, average_pra

logger = logging.getLogger(__name__)

# Short filler prompt exchanges to simulate conversation turns
_FILLER_PAIRS = [
    ("Can you summarize what we discussed?", "Sure — we covered the document provided earlier."),
    ("Do you have any follow-up thoughts?", "Nothing specific — let me know if you'd like me to elaborate."),
    ("Is there anything important I might have missed?", "The key points are still available in the earlier context."),
    ("Let's continue our discussion.", "Of course, I'm here to help."),
    ("Please confirm you still have the context.", "Yes, the earlier information is still in my context window."),
]

# Small context used for the multi-turn case (to stay within model limits)
_MULTITURN_CONTEXT_TOKENS = 8_000


class Exp3MultiTurn(BaseExperiment):

    experiment_id = "exp3_multiturn_degradation"

    def build_run_specs(self) -> List[RunSpec]:
        return self._planner.plan(self.config)

    async def execute_run(self, spec: RunSpec) -> Dict[str, Any]:
        set_seed(spec.seed)
        adapter = self._get_adapter(spec.model_id)
        n_turns = spec.turn_number or 10
        n_probes = max(1, self.config.probes_per_run // 2)  # fewer probes for conversation

        probes = self._sample_probes(n_probes, spec.seed)
        builder = ContextBuilder(model_id=spec.model_id, seed=spec.seed)
        injected = builder.build(
            target_tokens=_MULTITURN_CONTEXT_TOKENS,
            probe_facts=probes,
        )

        # Build conversation: context as first user turn
        messages: List[Dict[str, str]] = [
            {"role": "user", "content": (
                "Please read the following document carefully. "
                "I will ask you questions about it later.\n\n"
                + injected.prompt_text.split("--- QUESTIONS ---")[0].strip()
            )},
            {"role": "assistant", "content": "I have read the document. Please ask your questions."},
        ]

        # Add N-2 filler turns (N turns total before the final question)
        import random
        rng = random.Random(spec.seed + 1)
        for i in range(max(0, n_turns - 2)):
            pair = _FILLER_PAIRS[i % len(_FILLER_PAIRS)]
            messages.append({"role": "user", "content": pair[0]})
            messages.append({"role": "assistant", "content": pair[1]})

        # Build question block from scoring_rubric
        q_block = "Now please answer these questions based on the document:\n\n"
        for r in injected.scoring_rubric:
            q_block += f"Q{r['q_num']}: {r['gold']}\nAnswer: \n"  # re-ask questions
        # Fix: use actual question text
        q_block = "Now please answer these questions based on the document:\n\n"
        for q in injected.questions:
            q_block += f"Q{q['q_num']}: {q['q']}\nAnswer: \n"

        messages.append({"role": "user", "content": q_block})

        t0 = time.monotonic()
        response = await adapter.send_conversation(
            messages,
            max_output_tokens=self.config.max_output_tokens,
            temperature=self.config.temperature,
        )
        latency_ms = int((time.monotonic() - t0) * 1000)

        results = score_all_questions(response.text, injected.scoring_rubric)
        pra = average_pra(results)
        hallucinations = sum(1 for r in results if r.hallucination_flag)

        run_data = RunData(
            experiment_name=self.experiment_id,
            model=spec.model_id,
            seed=spec.seed,
            threshold_tokens=n_turns,  # turn count stored here for checkpoint dedup
            model_version=response.usage.model_version,
            condition="conversation",
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
            "[exp3] run_id=%d model=%s turns=%d pra=%.3f",
            run_id, spec.model_id, n_turns, pra,
        )
        return {
            "run_key": spec.run_key,
            "run_id": run_id,
            "pra": pra,
            "turns": n_turns,
            "model_id": spec.model_id,
        }
