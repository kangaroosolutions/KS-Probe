"""
SQLAlchemy ORM models for KS-Probe experiment logging.

Schema matches the document spec exactly:
  experiment_runs(run_id, experiment_name, model, threshold_tokens, seed,
                  condition, probe_position, pra_score, hallucination_count,
                  response_latency_ms, input_tokens, output_tokens,
                  raw_prompt, raw_response, created_at)
"""
from __future__ import annotations

from datetime import datetime

from sqlalchemy import (
    Column, DateTime, Float, Integer, String, Text, func,
)
from sqlalchemy.orm import DeclarativeBase


class Base(DeclarativeBase):
    pass


class ExperimentRun(Base):
    __tablename__ = "experiment_runs"

    run_id = Column(Integer, primary_key=True, autoincrement=True)
    experiment_name = Column(String, nullable=False)   # 'exp1_context_fidelity', etc.
    model = Column(String, nullable=False)             # model ID string
    model_version = Column(String, nullable=True)      # exact version from API response
    threshold_tokens = Column(Integer, nullable=True)  # context length tested
    seed = Column(Integer, nullable=False)
    condition = Column(String, nullable=True)          # 'single_prompt', 'conversation', etc.
    probe_position = Column(Float, nullable=True)      # 0.0–1.0 for positional experiments
    pra_score = Column(Float, nullable=True)           # Probe Recall Accuracy (0–1)
    hallucination_count = Column(Integer, default=0)
    response_latency_ms = Column(Integer, nullable=True)
    input_tokens = Column(Integer, nullable=True)
    output_tokens = Column(Integer, nullable=True)
    raw_prompt = Column(Text, nullable=True)
    raw_response = Column(Text, nullable=True)
    per_question_scores = Column(Text, nullable=True)  # JSON string
    created_at = Column(DateTime, default=datetime.utcnow, server_default=func.now())

    def __repr__(self) -> str:
        return (
            f"<ExperimentRun id={self.run_id} exp={self.experiment_name} "
            f"model={self.model} pra={self.pra_score:.3f}>"
        )
