"""
RunPlanner — expands an ExperimentConfig into a flat list of RunSpec objects.

Each RunSpec is one atomic unit of work: one model × one seed × one
context_size / position / turn combination.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from ks_probe.core.config import ExperimentConfig
from ks_probe.core.seed import SeedManager


@dataclass
class RunSpec:
    """One atomic unit of work dispatched by the orchestrator."""
    run_key: str                          # composite dedup key
    experiment_name: str
    model_id: str
    seed: int
    context_tokens: Optional[int] = None
    probe_position: Optional[float] = None
    condition: Optional[str] = None
    turn_number: Optional[int] = None
    payload: Dict[str, Any] = field(default_factory=dict)


def make_run_key(
    experiment_name: str,
    model_id: str,
    seed: int,
    context_tokens: Optional[int],
    probe_position: Optional[float],
    condition: Optional[str],
    turn_number: Optional[int],
) -> str:
    import hashlib, json
    parts = {
        "exp": experiment_name,
        "model": model_id,
        "seed": seed,
        "ctx": context_tokens,
        "pos": probe_position,
        "cond": condition,
        "turn": turn_number,
    }
    raw = json.dumps(parts, sort_keys=True)
    return hashlib.sha256(raw.encode()).hexdigest()[:32]


class RunPlanner:
    """
    Expands an ExperimentConfig into a deterministic list of RunSpecs.

    Supported experiment shapes:
      - context_sizes: for exp1 (context fidelity), exp4 (truncation)
      - positions:     for exp2 (positional recall) — list of floats
      - turns:         for exp3 (multi-turn degradation)
      - No dimensions: exp5 (tokenizer, local only)
    """

    def __init__(self, seed_manager: SeedManager) -> None:
        self._seeds = seed_manager

    def plan(self, config: ExperimentConfig, condition: Optional[str] = None) -> List[RunSpec]:
        specs: List[RunSpec] = []

        for model_id in config.models:
            for rep_idx, _ in enumerate(config.seeds):
                seed = self._seeds.get_seed(config.experiment_id, model_id, rep_idx)

                # exp1 / exp4 — vary context size
                if config.context_sizes:
                    for ctx in config.context_sizes:
                        specs.append(RunSpec(
                            run_key=make_run_key(
                                config.experiment_id, model_id, seed,
                                ctx, None, condition, None,
                            ),
                            experiment_name=config.experiment_id,
                            model_id=model_id,
                            seed=seed,
                            context_tokens=ctx,
                            condition=condition or "single_prompt",
                            payload={"context_tokens": ctx},
                        ))

                # exp2 — vary probe position
                elif config.positions is not None:
                    positions = config.positions  # list of floats
                    for pos in positions:
                        specs.append(RunSpec(
                            run_key=make_run_key(
                                config.experiment_id, model_id, seed,
                                None, pos, condition, None,
                            ),
                            experiment_name=config.experiment_id,
                            model_id=model_id,
                            seed=seed,
                            probe_position=pos,
                            condition=condition or "positional",
                            payload={"probe_position": pos},
                        ))

                # exp3 — vary turn count
                # context_tokens=turn so checkpoint can distinguish different
                # turn counts for the same (model, seed); stored in DB as
                # threshold_tokens so the checkpoint query matches correctly.
                elif config.turns is not None:
                    for turn in config.turns:
                        specs.append(RunSpec(
                            run_key=make_run_key(
                                config.experiment_id, model_id, seed,
                                turn, None, condition, turn,
                            ),
                            experiment_name=config.experiment_id,
                            model_id=model_id,
                            seed=seed,
                            context_tokens=turn,
                            turn_number=turn,
                            condition=condition or "conversation",
                            payload={"turns": turn},
                        ))

                # exp5 / fallback — single run per (model, seed)
                else:
                    specs.append(RunSpec(
                        run_key=make_run_key(
                            config.experiment_id, model_id, seed,
                            None, None, condition, None,
                        ),
                        experiment_name=config.experiment_id,
                        model_id=model_id,
                        seed=seed,
                        condition=condition or "default",
                        payload={},
                    ))

        return specs
