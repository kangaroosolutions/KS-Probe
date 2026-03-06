"""
Configuration loading and validation for KS-Probe experiments.
Uses Pydantic v2 for schema enforcement.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional

import yaml
from pydantic import BaseModel, Field, field_validator


# ── Model registry entry ──────────────────────────────────────────────────────

class ModelConfig(BaseModel):
    id: str
    provider: Literal["openai", "anthropic", "google", "xai", "deepseek", "perplexity", "vllm", "mock"]
    max_context_tokens: int
    cost_per_1k_input: float = 0.0
    cost_per_1k_output: float = 0.0
    api_base: Optional[str] = None
    api_version: Optional[str] = None
    notes: Optional[str] = None


# ── Experiment configuration ──────────────────────────────────────────────────

class ExperimentConfig(BaseModel):
    experiment_id: str
    description: str = ""
    models: List[str]
    seeds: List[int] = Field(default_factory=lambda: [1, 2, 3, 4, 5])

    # Context-length experiments (exp1, exp2, exp4)
    context_sizes: Optional[List[int]] = None

    # Positional experiment (exp2)
    positions: Optional[List[float]] = None  # 0.0 – 1.0

    # Multi-turn experiment (exp3)
    turns: Optional[List[int]] = None

    # Probe settings
    probes_per_run: int = 10
    probe_type_mix: List[float] = Field(default_factory=lambda: [0.4, 0.35, 0.25])
    domain_mix: str = "equal"

    # Execution settings
    max_output_tokens: int = 2000
    max_concurrent: int = 4
    temperature: float = 0.0
    cost_confirm_threshold: float = 500.0

    # Run modes
    dry_run: bool = False
    mock_mode: bool = False

    # Storage
    output_dir: str = "outputs"
    checkpoint_db: str = ""  # empty = use DATABASE_URL env or SQLite default

    extra: Dict[str, Any] = Field(default_factory=dict)

    @field_validator("probe_type_mix")
    @classmethod
    def validate_mix(cls, v: List[float]) -> List[float]:
        if abs(sum(v) - 1.0) > 1e-6:
            raise ValueError(f"probe_type_mix must sum to 1.0, got {sum(v)}")
        return v


# ── Model registry ────────────────────────────────────────────────────────────

class ModelRegistry(BaseModel):
    models: List[ModelConfig]

    def get(self, model_id: str) -> ModelConfig:
        for m in self.models:
            if m.id == model_id:
                return m
        raise KeyError(f"Model '{model_id}' not found in registry")

    def all_ids(self) -> List[str]:
        return [m.id for m in self.models]


# ── Loaders ───────────────────────────────────────────────────────────────────

def load_config(path: str | Path) -> ExperimentConfig:
    """Load and validate an experiment YAML config."""
    with Path(path).open("r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)
    return ExperimentConfig(**raw)


def load_model_registry(path: str | Path) -> ModelRegistry:
    """Load and validate models.yaml."""
    with Path(path).open("r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)
    return ModelRegistry(models=[ModelConfig(**m) for m in raw.get("models", [])])


def get_models_yaml_path() -> Path:
    """Return the default models.yaml path relative to this package."""
    return Path(__file__).resolve().parents[2] / "configs" / "models.yaml"


def get_db_url() -> str:
    """Return the database URL, defaulting to a local SQLite file."""
    url = os.getenv("DATABASE_URL", "")
    if url:
        return url
    db_path = Path(__file__).resolve().parents[2] / "ks_probe_results.db"
    return f"sqlite:///{db_path}"
