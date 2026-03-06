#!/usr/bin/env python3
"""
Export KS-Probe experiment results from the database to CSV.

Usage:
    python scripts/export_results.py --experiment exp1_context_fidelity
    python scripts/export_results.py --all --output results/
    python scripts/export_results.py --experiment exp1_context_fidelity --model claude-sonnet-4-6
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from ks_probe.db.queries import query_runs, get_summary


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export KS-Probe results to CSV")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--experiment", "-e", help="Experiment ID to export")
    group.add_argument("--all", action="store_true", help="Export all experiments")
    parser.add_argument("--model", "-m", default=None, help="Filter by model ID")
    parser.add_argument("--output", "-o", default="outputs", help="Output directory")
    parser.add_argument("--db-url", default=None, help="Database URL")
    parser.add_argument("--summary", "-s", action="store_true",
                        help="Print summary statistics after export")
    return parser.parse_args(argv)


def export_runs(experiment_name, model, output_dir, db_url, print_summary=False):
    runs = query_runs(experiment_name=experiment_name, model=model, db_url=db_url)
    if not runs:
        print(f"No runs found for experiment='{experiment_name}' model='{model}'.")
        return

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    fname = f"{experiment_name or 'all'}.csv"
    csv_path = out / fname

    fieldnames = [
        "run_id", "experiment_name", "model", "model_version",
        "seed", "threshold_tokens", "condition", "probe_position",
        "pra_score", "hallucination_count",
        "response_latency_ms", "input_tokens", "output_tokens",
        "created_at",
    ]

    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for run in runs:
            writer.writerow({
                "run_id": run.run_id,
                "experiment_name": run.experiment_name,
                "model": run.model,
                "model_version": run.model_version,
                "seed": run.seed,
                "threshold_tokens": run.threshold_tokens,
                "condition": run.condition,
                "probe_position": run.probe_position,
                "pra_score": run.pra_score,
                "hallucination_count": run.hallucination_count,
                "response_latency_ms": run.response_latency_ms,
                "input_tokens": run.input_tokens,
                "output_tokens": run.output_tokens,
                "created_at": run.created_at,
            })

    print(f"Exported {len(runs)} runs -> {csv_path}")

    if print_summary and experiment_name:
        summary = get_summary(experiment_name, db_url=db_url)
        print("\n-- Summary ------------------------------------------")
        for k, v in summary.items():
            print(f"  {k}: {v}")
        print("-----------------------------------------------------\n")


def main(argv=None):
    args = parse_args(argv)
    db_url = args.db_url

    if args.all:
        runs = query_runs(db_url=db_url)
        experiments = {r.experiment_name for r in runs}
        for exp in sorted(experiments):
            export_runs(exp, args.model, args.output, db_url, args.summary)
    else:
        export_runs(args.experiment, args.model, args.output, db_url, args.summary)


if __name__ == "__main__":
    main()
