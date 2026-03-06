"""
Database query helpers for KS-Probe experiment logging and checkpointing.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from sqlalchemy.orm import Session

from ks_probe.db.engine import get_session
from ks_probe.db.schema import ExperimentRun


@dataclass
class RunData:
    experiment_name: str
    model: str
    seed: int
    threshold_tokens: Optional[int] = None
    model_version: Optional[str] = None
    condition: Optional[str] = None
    probe_position: Optional[float] = None
    pra_score: Optional[float] = None
    hallucination_count: int = 0
    response_latency_ms: Optional[int] = None
    input_tokens: Optional[int] = None
    output_tokens: Optional[int] = None
    raw_prompt: Optional[str] = None
    raw_response: Optional[str] = None
    per_question_scores: Optional[Dict[str, Any]] = None


def checkpoint_exists(
    experiment_name: str,
    model: str,
    seed: int,
    threshold_tokens: Optional[int] = None,
    probe_position: Optional[float] = None,
    db_url: Optional[str] = None,
) -> bool:
    """
    Return True if a completed run already exists for this combination.
    Used to skip already-done work on restart.
    """
    with get_session(db_url) as session:
        q = session.query(ExperimentRun).filter(
            ExperimentRun.experiment_name == experiment_name,
            ExperimentRun.model == model,
            ExperimentRun.seed == seed,
            ExperimentRun.pra_score.isnot(None),  # only fully scored runs count
        )
        if threshold_tokens is not None:
            q = q.filter(ExperimentRun.threshold_tokens == threshold_tokens)
        if probe_position is not None:
            q = q.filter(ExperimentRun.probe_position == probe_position)
        return q.count() > 0


def insert_result(run_data: RunData, db_url: Optional[str] = None) -> int:
    """Insert a completed run result. Returns the new run_id."""
    with get_session(db_url) as session:
        row = ExperimentRun(
            experiment_name=run_data.experiment_name,
            model=run_data.model,
            model_version=run_data.model_version,
            threshold_tokens=run_data.threshold_tokens,
            seed=run_data.seed,
            condition=run_data.condition,
            probe_position=run_data.probe_position,
            pra_score=run_data.pra_score,
            hallucination_count=run_data.hallucination_count,
            response_latency_ms=run_data.response_latency_ms,
            input_tokens=run_data.input_tokens,
            output_tokens=run_data.output_tokens,
            raw_prompt=run_data.raw_prompt,
            raw_response=run_data.raw_response,
            per_question_scores=(
                json.dumps(run_data.per_question_scores)
                if run_data.per_question_scores
                else None
            ),
        )
        session.add(row)
        session.commit()
        return row.run_id


def query_runs(
    experiment_name: Optional[str] = None,
    model: Optional[str] = None,
    db_url: Optional[str] = None,
) -> List[ExperimentRun]:
    """Query completed runs with optional filters."""
    with get_session(db_url) as session:
        q = session.query(ExperimentRun)
        if experiment_name:
            q = q.filter(ExperimentRun.experiment_name == experiment_name)
        if model:
            q = q.filter(ExperimentRun.model == model)
        return q.order_by(ExperimentRun.created_at).all()


def get_summary(experiment_name: str, db_url: Optional[str] = None) -> Dict[str, Any]:
    """Return summary statistics for a completed experiment."""
    import numpy as np

    runs = query_runs(experiment_name=experiment_name, db_url=db_url)
    if not runs:
        return {}

    scores = [r.pra_score for r in runs if r.pra_score is not None]
    hallucinations = [r.hallucination_count for r in runs if r.hallucination_count is not None]

    return {
        "experiment_name": experiment_name,
        "total_runs": len(runs),
        "mean_pra": float(np.mean(scores)) if scores else None,
        "std_pra": float(np.std(scores)) if scores else None,
        "min_pra": float(np.min(scores)) if scores else None,
        "max_pra": float(np.max(scores)) if scores else None,
        "total_hallucinations": sum(hallucinations),
        "models_tested": list({r.model for r in runs}),
    }
