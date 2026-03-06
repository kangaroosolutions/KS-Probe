"""
Cost estimation and pre-run budget guard for KS-Probe.

Estimates API spend before a batch starts and requires manual confirmation
if estimated cost exceeds a configurable threshold.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List

from ks_probe.core.config import ExperimentConfig, ModelConfig, ModelRegistry


@dataclass
class CostBreakdown:
    model_id: str
    num_runs: int
    avg_input_tokens: int
    avg_output_tokens: int
    cost_per_run: float
    total_cost: float


@dataclass
class CostEstimate:
    experiment_id: str
    breakdown: List[CostBreakdown] = field(default_factory=list)

    @property
    def total(self) -> float:
        return sum(b.total_cost for b in self.breakdown)

    def print_table(self) -> None:
        print(f"\n{'='*70}")
        print(f"  Cost Estimate — {self.experiment_id}")
        print(f"{'='*70}")
        print(f"  {'Model':<25} {'Runs':>6} {'$/run':>8} {'Total':>10}")
        print(f"  {'-'*55}")
        for b in self.breakdown:
            print(
                f"  {b.model_id:<25} {b.num_runs:>6} "
                f"${b.cost_per_run:>7.4f} ${b.total_cost:>9.2f}"
            )
        print(f"  {'─'*55}")
        print(f"  {'TOTAL':<25} {'':>6} {'':>8} ${self.total:>9.2f}")
        print(f"{'='*70}\n")


def estimate_cost(
    config: ExperimentConfig,
    registry: ModelRegistry,
    avg_input_tokens_per_run: int = 55000,
    avg_output_tokens_per_run: int = 2000,
) -> CostEstimate:
    """
    Estimate API cost for an experiment configuration.

    Args:
        config: Experiment configuration.
        registry: Model registry with pricing info.
        avg_input_tokens_per_run: Expected average input tokens per API call.
        avg_output_tokens_per_run: Expected average output tokens per API call.

    Returns:
        CostEstimate with per-model breakdown.
    """
    context_sizes = config.context_sizes or [avg_input_tokens_per_run]
    num_seeds = len(config.seeds)
    estimate = CostEstimate(experiment_id=config.experiment_id)

    for model_id in config.models:
        try:
            model_cfg = registry.get(model_id)
        except KeyError:
            continue

        if model_cfg.provider == "mock":
            continue

        # Cap context sizes at model's max
        valid_sizes = [s for s in context_sizes if s <= model_cfg.max_context_tokens]
        num_context_sizes = len(valid_sizes) if valid_sizes else 1
        num_runs = num_context_sizes * num_seeds

        # For positional experiments, each position = 1 run
        if config.positions:
            num_runs = len(config.positions) * num_seeds

        avg_input = avg_input_tokens_per_run
        avg_output = avg_output_tokens_per_run
        cost_per_run = (avg_input / 1000 * model_cfg.cost_per_1k_input) + (
            avg_output / 1000 * model_cfg.cost_per_1k_output
        )
        total = cost_per_run * num_runs

        estimate.breakdown.append(
            CostBreakdown(
                model_id=model_id,
                num_runs=num_runs,
                avg_input_tokens=avg_input,
                avg_output_tokens=avg_output,
                cost_per_run=cost_per_run,
                total_cost=total,
            )
        )

    return estimate


def confirm_before_run(estimate: CostEstimate, threshold: float = 500.0) -> bool:
    """
    Print cost estimate and ask for confirmation if it exceeds threshold.

    Returns True if the run should proceed, False otherwise.
    """
    estimate.print_table()

    if estimate.total <= threshold:
        print(f"  Estimated cost ${estimate.total:.2f} is under threshold ${threshold:.2f}. Proceeding.\n")
        return True

    print(
        f"  WARNING: Estimated cost ${estimate.total:.2f} exceeds "
        f"threshold ${threshold:.2f}.\n"
        f"  Type 'yes' to proceed, or anything else to abort: ",
        end="",
        flush=True,
    )
    response = input().strip().lower()
    if response == "yes":
        print("  Confirmed. Starting run.\n")
        return True
    print("  Aborted.\n")
    return False
