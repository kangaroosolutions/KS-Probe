#!/usr/bin/env python3
"""
KS-Probe CLI entrypoint.

Usage:
    python scripts/run_experiment.py --config configs/exp1_context_fidelity.yaml
    python scripts/run_experiment.py --config configs/exp1_context_fidelity.yaml --mock-mode
    python scripts/run_experiment.py --config configs/exp1_context_fidelity.yaml --dry-run
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

# Ensure project root is on the path when run directly
_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

# Load .env file automatically (OPENAI_API_KEY, ANTHROPIC_API_KEY, etc.)
try:
    from dotenv import load_dotenv
    load_dotenv(_ROOT / ".env", encoding="utf-8", override=True)
except ImportError:
    pass  # python-dotenv not installed — keys must be set in the environment manually

from ks_probe.core.config import load_config, load_model_registry, get_models_yaml_path
from ks_probe.core.seed import SeedManager

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("ks_probe.cli")

# Map experiment_id → experiment class
_EXPERIMENT_REGISTRY = {
    "exp1_context_fidelity":    "ks_probe.experiments.exp1_context_fidelity.Exp1ContextFidelity",
    "exp2_positional_recall":   "ks_probe.experiments.exp2_positional_recall.Exp2PositionalRecall",
    "exp3_multiturn_degradation": "ks_probe.experiments.exp3_multiturn.Exp3MultiTurn",
    "exp4_silent_truncation":   "ks_probe.experiments.exp4_truncation.Exp4SilentTruncation",
    "exp5_tokenizer_divergence": "ks_probe.experiments.exp5_tokenizer.Exp5TokenizerDivergence",
}


def _import_experiment(dotted_path: str):
    module_path, class_name = dotted_path.rsplit(".", 1)
    import importlib
    mod = importlib.import_module(module_path)
    return getattr(mod, class_name)


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="ks-probe",
        description="KS-Probe: LLM context retention benchmark runner",
    )
    parser.add_argument(
        "--config", "-c", required=True,
        help="Path to experiment YAML config file",
    )
    parser.add_argument(
        "--models-yaml", default=None,
        help="Path to models.yaml (default: configs/models.yaml)",
    )
    parser.add_argument(
        "--mock-mode", action="store_true",
        help="Override: use MockAdapter for all models (no API keys needed)",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Build RunSpecs and print cost estimate, then exit without running",
    )
    parser.add_argument(
        "--db-url", default=None,
        help="Database URL (default: sqlite:///ks_probe_results.db)",
    )
    return parser.parse_args(argv)


def main(argv=None) -> None:
    args = parse_args(argv)

    config_path = Path(args.config)
    if not config_path.exists():
        logger.error("Config file not found: %s", config_path)
        sys.exit(1)

    config = load_config(config_path)

    # CLI overrides
    if args.mock_mode:
        config = config.model_copy(update={"mock_mode": True})
    if args.dry_run:
        config = config.model_copy(update={"dry_run": True})

    experiment_id = config.experiment_id
    if experiment_id not in _EXPERIMENT_REGISTRY:
        logger.error(
            "Unknown experiment_id '%s'. Known: %s",
            experiment_id, list(_EXPERIMENT_REGISTRY.keys()),
        )
        sys.exit(1)

    ExperimentClass = _import_experiment(_EXPERIMENT_REGISTRY[experiment_id])

    # Load model registry
    models_yaml = Path(args.models_yaml) if args.models_yaml else get_models_yaml_path()
    model_registry = None
    if models_yaml.exists():
        model_registry = load_model_registry(models_yaml)

    seed_manager = SeedManager()
    experiment = ExperimentClass(
        config=config,
        seed_manager=seed_manager,
        model_registry=model_registry,
        models_yaml=models_yaml,
        db_url=args.db_url,
    )

    logger.info("Starting experiment: %s", experiment_id)
    results = experiment.run()
    logger.info("Done. %d results returned.", len(results))


if __name__ == "__main__":
    main()
