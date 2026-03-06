"""
Reproducibility seed management for KS-Probe experiments.

Mirrors the set_seed(seed, deterministic=True) pattern used in
QFM_Research/train_cifar.py, extended for LLM-experiment reproducibility.
"""
from __future__ import annotations

import hashlib
import json
import random
from pathlib import Path
from typing import Optional

import numpy as np

MASTER_SEED_PATH = Path(__file__).resolve().parents[2] / "master_seed.json"


def set_seed(seed: int, deterministic: bool = True) -> None:
    """Set random seeds for Python, NumPy, and PyTorch (if available)."""
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        if deterministic:
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
    except ImportError:
        pass  # torch not required for LLM API experiments


def load_master_seed(path: Path = MASTER_SEED_PATH) -> dict:
    """Load the master seed file."""
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def derive_seed(master: int, experiment_id: str, model_id: str, replicate: int) -> int:
    """
    Deterministically derive a per-run seed.

    Uses SHA-256 so the same (master, experiment_id, model_id, replicate)
    combination always produces the same integer seed, regardless of call order.
    """
    key = f"{master}:{experiment_id}:{model_id}:{replicate}"
    digest = hashlib.sha256(key.encode("utf-8")).hexdigest()
    return int(digest[:8], 16) % (2**31)


class SeedManager:
    """Central seed manager for an experiment session."""

    def __init__(self, master_seed_path: Optional[Path] = None) -> None:
        path = master_seed_path or MASTER_SEED_PATH
        self._master = load_master_seed(path)
        self.global_seed: int = self._master["global_seed"]

    def get_seed(self, experiment_id: str, model_id: str, replicate: int) -> int:
        """Return the deterministic seed for a specific run combination."""
        return derive_seed(self.global_seed, experiment_id, model_id, replicate)

    def set_for_run(
        self, experiment_id: str, model_id: str, replicate: int, deterministic: bool = True
    ) -> int:
        """Compute and immediately apply the seed. Returns the seed used."""
        seed = self.get_seed(experiment_id, model_id, replicate)
        set_seed(seed, deterministic=deterministic)
        return seed
