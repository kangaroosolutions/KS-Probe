"""Abstract base class for all scorers."""
from __future__ import annotations

import abc
from dataclasses import dataclass
from typing import List, Optional


@dataclass
class ScorerResult:
    """Aggregated scoring result for a single probe question."""
    exact_match: Optional[bool]
    keyword_score: Optional[float]   # 0.0–1.0
    constraint_score: Optional[float]
    hallucination_flag: bool
    pra_score: float                 # final Probe Recall Accuracy [0, 1]


class BaseScorer(abc.ABC):
    """Abstract base for individual scorers."""

    @abc.abstractmethod
    def score(self, response: str, rubric: dict) -> float:
        """
        Score a model response against a rubric entry.

        Args:
            response: The model's text response.
            rubric: A scoring_rubric entry dict with keys:
                    gold, scoring, keywords, required_conclusion.

        Returns:
            Float score in [0, 1].
        """
