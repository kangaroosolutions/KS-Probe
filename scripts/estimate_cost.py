#!/usr/bin/env python3
"""
Estimate API cost for a KS-Probe experiment without running it.

Usage:
    python scripts/estimate_cost.py --config configs/exp1_context_fidelity.yaml
    python scripts/estimate_cost.py --config configs/exp1_context_fidelity.yaml --all-experiments
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from ks_probe.core.config import load_config, load_model_registry, get_models_yaml_path
from ks_probe.core.cost import estimate_cost


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Estimate KS-Probe experiment cost")
    parser.add_argument("--config", "-c", help="Path to experiment YAML config")
    parser.add_argument(
        "--all-experiments", action="store_true",
        help="Estimate cost for all configs in configs/ directory",
    )
    parser.add_argument(
        "--models-yaml", default=None,
        help="Path to models.yaml (default: configs/models.yaml)",
    )
    return parser.parse_args(argv)


def estimate_one(config_path: Path, models_yaml: Path) -> None:
    config = load_config(config_path)
    model_registry = load_model_registry(models_yaml) if models_yaml.exists() else None
    if model_registry is None:
        print(f"  Warning: models.yaml not found at {models_yaml}. Skipping cost estimate.")
        return

    est = estimate_cost(config, model_registry)
    est.print_table()


def main(argv=None):
    args = parse_args(argv)
    models_yaml = Path(args.models_yaml) if args.models_yaml else get_models_yaml_path()

    if args.all_experiments:
        configs_dir = _ROOT / "configs"
        configs = sorted(configs_dir.glob("exp*.yaml"))
        if not configs:
            print(f"No experiment configs found in {configs_dir}")
            sys.exit(1)
        for cfg_path in configs:
            print(f"\n{'='*60}")
            print(f"  {cfg_path.name}")
            estimate_one(cfg_path, models_yaml)
    elif args.config:
        estimate_one(Path(args.config), models_yaml)
    else:
        print("Provide --config or --all-experiments")
        sys.exit(1)


if __name__ == "__main__":
    main()
