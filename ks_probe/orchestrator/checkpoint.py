"""
CheckpointManager — wraps the DB queries to skip already-completed runs.

Uses the existing ks_probe.db.queries.checkpoint_exists() function which checks
whether a run with matching (experiment_name, model, seed, threshold_tokens,
probe_position) already has a pra_score in the database.
"""
from __future__ import annotations

from typing import Optional

from ks_probe.db.queries import checkpoint_exists as _db_checkpoint_exists


class CheckpointManager:
    """
    Stateless wrapper around the DB checkpoint query.
    Provides a consistent interface for the orchestrator to check
    whether a given RunSpec has already been completed.
    """

    def __init__(self, db_url: Optional[str] = None) -> None:
        self._db_url = db_url

    def is_done(
        self,
        experiment_name: str,
        model_id: str,
        seed: int,
        threshold_tokens: Optional[int] = None,
        probe_position: Optional[float] = None,
    ) -> bool:
        """Return True if this run combination already has a result in the DB."""
        return _db_checkpoint_exists(
            experiment_name=experiment_name,
            model=model_id,
            seed=seed,
            threshold_tokens=threshold_tokens,
            probe_position=probe_position,
            db_url=self._db_url,
        )
