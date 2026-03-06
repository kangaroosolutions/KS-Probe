"""Hallucination detector — flags responses that confidently assert wrong facts."""
from __future__ import annotations

import re
import unicodedata
from typing import List

from ks_probe.scorer.base import BaseScorer

# Phrases that signal a definitive (non-hedged) claim
_DEFINITE_CLAIM_PATTERNS = [
    r"\bis\b", r"\bwas\b", r"\bare\b", r"\bwere\b",
    r"\bequals\b", r"\bthe answer is\b", r"\bit is\b",
]

# Phrases that signal uncertainty / refusal (safe — not a hallucination)
_HEDGING_PATTERNS = [
    r"i (don't|do not|cannot|can't) (know|find|recall|remember)",
    r"i('m| am) not sure",
    r"not (enough|sufficient) information",
    r"i (cannot|can't) answer",
    r"no information",
    r"unclear",
]


def _normalize(text: str) -> str:
    text = unicodedata.normalize("NFKD", text).lower()
    text = re.sub(r"[^\w\s]", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def _is_hedged(response_lower: str) -> bool:
    return any(re.search(p, response_lower) for p in _HEDGING_PATTERNS)


def _contains_gold(norm_resp: str, norm_gold: str) -> bool:
    return bool(norm_gold) and norm_gold in norm_resp


class HallucinationDetector(BaseScorer):
    """
    Flags a response as hallucination when ALL of:
      1. The gold answer is NOT in the response.
      2. The response is not hedged / refuses to answer.
      3. The response makes at least one definitive claim.

    Hallucination voids the PRA score (returns 0.0 from CompositeScorer).
    """

    def score(self, response: str, rubric: dict) -> float:
        """Returns 1.0 if hallucination detected, 0.0 otherwise."""
        return float(self.detect(response, rubric.get("gold", "")))

    def detect(self, response: str, gold: str) -> bool:
        norm_resp = _normalize(response)
        norm_gold = _normalize(gold)

        if _contains_gold(norm_resp, norm_gold):
            return False  # Correct answer — no hallucination

        if _is_hedged(norm_resp):
            return False  # Safe refusal

        # Check if the model makes a definite-sounding claim
        makes_claim = any(
            re.search(p, norm_resp) for p in _DEFINITE_CLAIM_PATTERNS
        )
        return makes_claim


def is_hallucination(response: str, gold: str) -> bool:
    """Convenience wrapper."""
    return HallucinationDetector().detect(response, gold)
