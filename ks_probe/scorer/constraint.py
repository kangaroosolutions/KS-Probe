"""Constraint compliance scorer — checks "must contain X" style constraints."""
from __future__ import annotations

import re
import unicodedata
from typing import List

from ks_probe.scorer.base import BaseScorer


def _normalize(text: str) -> str:
    text = unicodedata.normalize("NFKD", text).lower()
    text = re.sub(r"[^\w\s]", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def _extract_required_term(constraint: str) -> str:
    """
    Parse constraint strings like:
      "must contain X"  → "X"
      "must mention X"  → "X"
      "must include X"  → "X"
      "requires X"      → "X"
    Falls back to using the whole constraint string if no pattern matches.
    """
    patterns = [
        r"must (?:contain|mention|include|reference|state)\s+(.+)",
        r"requires?\s+(.+)",
        r"should (?:contain|mention|include)\s+(.+)",
    ]
    constraint_lower = constraint.lower().strip()
    for pat in patterns:
        m = re.match(pat, constraint_lower)
        if m:
            return m.group(1).strip().strip("\"'")
    return constraint_lower


class ConstraintComplianceScorer(BaseScorer):
    """
    Scores whether the response satisfies all constraints.
    Score = (# constraints satisfied) / (# total constraints).

    Falls back to keyword scoring against `required_conclusion` if no
    explicit constraints are provided.
    """

    def score(self, response: str, rubric: dict) -> float:
        constraints: List[str] = rubric.get("constraints") or []
        required_conclusion: str = rubric.get("required_conclusion", "")

        if not constraints and not required_conclusion:
            return 1.0  # Nothing to check — vacuously satisfied

        norm_resp = _normalize(response)

        if constraints:
            satisfied = sum(
                1 for c in constraints
                if _normalize(_extract_required_term(c)) in norm_resp
            )
            return satisfied / len(constraints)

        # Fallback: check required_conclusion as a keyword
        norm_conclusion = _normalize(required_conclusion)
        return 1.0 if norm_conclusion and norm_conclusion in norm_resp else 0.0


def constraint_score(response: str, constraints: List[str], required_conclusion: str = "") -> float:
    """Convenience wrapper."""
    return ConstraintComplianceScorer().score(
        response,
        {"constraints": constraints, "required_conclusion": required_conclusion},
    )
