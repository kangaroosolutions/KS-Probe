"""Pydantic data models for probe facts and questions."""
from __future__ import annotations

from typing import List, Literal, Optional
from pydantic import BaseModel, field_validator


class ProbeQuestion(BaseModel):
    q: str
    gold: str
    scoring: Literal["exact_match", "keyword_rule", "constraint_compliance"]
    keywords: List[str] = []
    required_conclusion: str = ""


class ProbeFact(BaseModel):
    id: str   # PF-001 … PF-100
    type: Literal["explicit_recall", "grounded_reasoning", "implicit_context"]
    statement: str
    questions: List[ProbeQuestion]

    @field_validator("questions")
    @classmethod
    def at_least_one_question(cls, v: List[ProbeQuestion]) -> List[ProbeQuestion]:
        if not v:
            raise ValueError("Each probe fact must have at least one question.")
        return v


class ProbePool(BaseModel):
    probe_facts: List[ProbeFact]

    def get(self, probe_id: str) -> ProbeFact:
        for pf in self.probe_facts:
            if pf.id == probe_id:
                return pf
        raise KeyError(f"Probe fact '{probe_id}' not found in pool.")

    def by_type(self, fact_type: str) -> List[ProbeFact]:
        return [pf for pf in self.probe_facts if pf.type == fact_type]

    def sample(
        self,
        n: int,
        type_mix: Optional[List[float]] = None,
        rng: Optional[object] = None,
    ) -> List[ProbeFact]:
        """
        Sample n probe facts according to type_mix ratios
        [explicit_recall, grounded_reasoning, implicit_context].
        """
        import random as _random
        import math

        rng = rng or _random

        types = ["explicit_recall", "grounded_reasoning", "implicit_context"]
        if type_mix is None:
            type_mix = [0.4, 0.35, 0.25]

        selected: List[ProbeFact] = []
        for t, ratio in zip(types, type_mix):
            pool = self.by_type(t)
            count = math.floor(n * ratio)
            selected.extend(rng.sample(pool, min(count, len(pool))))

        # Fill remainder (rounding)
        remaining = n - len(selected)
        if remaining > 0:
            all_ids = {pf.id for pf in selected}
            leftover = [pf for pf in self.probe_facts if pf.id not in all_ids]
            selected.extend(rng.sample(leftover, min(remaining, len(leftover))))

        return selected[:n]
