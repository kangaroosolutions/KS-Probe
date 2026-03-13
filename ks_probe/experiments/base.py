"""
BaseExperiment — abstract base class for all KS-Probe experiments.

Each experiment subclass must implement:
  - experiment_id: str  (class attribute)
  - build_run_specs() -> List[RunSpec]
  - execute_run(spec: RunSpec) -> dict

The run() method handles cost estimation, confirmation, and dispatching
through AsyncOrchestrator.
"""
from __future__ import annotations

import asyncio
import json
import logging
import time
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, List, Optional

from ks_probe.adapters.registry import AdapterRegistry, get_registry
from ks_probe.core.config import ExperimentConfig, ModelRegistry, load_model_registry, get_models_yaml_path
from ks_probe.core.cost import CostEstimate, estimate_cost, confirm_before_run
from ks_probe.core.seed import SeedManager
from ks_probe.corpus.builder import ContextBuilder
from ks_probe.db.engine import init_db
from ks_probe.db.queries import RunData, insert_result
from ks_probe.orchestrator.checkpoint import CheckpointManager
from ks_probe.orchestrator.planner import RunPlanner, RunSpec
from ks_probe.orchestrator.rate_limiter import TokenBucketRateLimiter
from ks_probe.orchestrator.runner import AsyncOrchestrator
from ks_probe.probes.loader import load_probe_pool
from ks_probe.probes.types import ProbePool

logger = logging.getLogger(__name__)

# Default RPM per provider (conservative — override via models.yaml extra fields)
_DEFAULT_RPM: Dict[str, float] = {
    "openai": 10,
    "anthropic": 10,   # RPM alone insufficient for Anthropic; paired with TPM below
    "google": 20,
    "xai": 20,
    "deepseek": 10,
    "vllm": 60,
    "mock": 1000,
}

# Default input-tokens-per-minute budget per provider.
# Providers not listed here (or set to 0) use RPM-only limiting.
_DEFAULT_TPM: Dict[str, int] = {
    "anthropic": 25000,  # Tier-1 cap: 30K input tokens/min — 25K with safety buffer
}


class BaseExperiment(ABC):
    """Abstract base for all KS-Probe experiments."""

    experiment_id: str  # must be set in subclass

    def __init__(
        self,
        config: ExperimentConfig,
        seed_manager: Optional[SeedManager] = None,
        model_registry: Optional[ModelRegistry] = None,
        models_yaml: Optional[Path] = None,
        db_url: Optional[str] = None,
    ) -> None:
        self.config = config
        self.seed_manager = seed_manager or SeedManager()
        self.db_url = db_url or (config.checkpoint_db or None)

        # Load model registry
        reg_path = models_yaml or get_models_yaml_path()
        self.model_registry: ModelRegistry = model_registry or (
            load_model_registry(reg_path) if reg_path.exists() else ModelRegistry(models=[])
        )

        # Adapter registry
        self._adapters: AdapterRegistry = get_registry()

        # Probe pool (shared, cached)
        self.probe_pool: ProbePool = load_probe_pool()

        # Output directory
        self.output_dir = Path(config.output_dir) / self.experiment_id
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Ensure DB tables exist
        init_db(self.db_url)

        # Checkpoint manager
        self._checkpoint = CheckpointManager(db_url=self.db_url)

        # Planner
        self._planner = RunPlanner(self.seed_manager)

    # ── Abstract interface ────────────────────────────────────────────────────

    @abstractmethod
    def build_run_specs(self) -> List[RunSpec]:
        """Return a deterministic list of RunSpecs for this experiment."""

    @abstractmethod
    async def execute_run(self, spec: RunSpec) -> Dict[str, Any]:
        """Execute one RunSpec. Returns a result dict."""

    # ── Public entrypoint ─────────────────────────────────────────────────────

    def run(self) -> List[Dict[str, Any]]:
        """Build specs, estimate cost, confirm, then dispatch."""
        specs = self.build_run_specs()
        logger.info("[%s] %d run specs generated.", self.experiment_id, len(specs))

        if self.config.dry_run:
            logger.info("[%s] dry_run=True — exiting without executing.", self.experiment_id)
            self._save_run_log(specs=specs, results=[], runtime=0.0)
            return []

        if not self.config.mock_mode:
            est = estimate_cost(
                self.config, self.model_registry,
                avg_input_tokens_per_run=self._avg_input_tokens(),
            )
            if not confirm_before_run(est, threshold=self.config.cost_confirm_threshold):
                logger.info("[%s] Aborted by user.", self.experiment_id)
                return []

        rate_limiters = self._build_rate_limiters()
        orchestrator = AsyncOrchestrator(
            run_fn=self.execute_run,
            rate_limiters=rate_limiters,
            max_concurrent=self.config.max_concurrent,
            checkpoint_manager=self._checkpoint,
            mock_mode=self.config.mock_mode,
        )

        t0 = time.perf_counter()
        results = asyncio.run(orchestrator.run_all(specs))
        runtime = time.perf_counter() - t0

        logger.info(
            "[%s] Completed %d/%d runs in %.1fs.",
            self.experiment_id, len(results), len(specs), runtime,
        )
        self._save_run_log(specs=specs, results=results, runtime=runtime)
        return results

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _get_adapter(self, model_id: str):
        return self._adapters.get(model_id)

    def _sample_probes(self, n: int, seed: int):
        import random
        rng = random.Random(seed)
        return self.probe_pool.sample(n=n, type_mix=self.config.probe_type_mix, rng=rng)

    def _avg_input_tokens(self) -> int:
        """Rough average input token estimate for cost calculation."""
        if self.config.context_sizes:
            return int(sum(self.config.context_sizes) / len(self.config.context_sizes))
        return 50_000

    def _build_rate_limiters(self) -> Dict[str, TokenBucketRateLimiter]:
        limiters: Dict[str, TokenBucketRateLimiter] = {}
        for model_id in self.config.models:
            try:
                mcfg = self.model_registry.get(model_id)
                provider = mcfg.provider
            except KeyError:
                provider = "mock"
            rpm = _DEFAULT_RPM.get(provider, 10)
            tpm = _DEFAULT_TPM.get(provider, 0)
            limiters[model_id] = TokenBucketRateLimiter(
                requests_per_minute=rpm,
                tokens_per_minute=tpm,
            )
        return limiters

    def _save_run_log(
        self,
        specs: List[RunSpec],
        results: List[Dict],
        runtime: float,
    ) -> None:
        log_path = self.output_dir / "run_log.json"
        data = {
            "experiment_id": self.experiment_id,
            "n_specs": len(specs),
            "n_results": len(results),
            "runtime_seconds": round(runtime, 2),
            "errors": [r for r in results if "error" in r],
        }
        with log_path.open("w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)

    def _write_result(self, run_data: RunData) -> int:
        """Convenience wrapper to insert a result into the DB."""
        return insert_result(run_data, db_url=self.db_url)
