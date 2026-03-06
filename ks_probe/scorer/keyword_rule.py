"""Keyword rule scorer — fraction of required keywords found in response."""
from __future__ import annotations

import re
import unicodedata
from typing import List

from ks_probe.scorer.base import BaseScorer


def _normalize(text: str) -> str:
    text = unicodedata.normalize("NFKD", text).lower()
    text = re.sub(r"[^\w\s]", " ", text)
    return re.sub(r"\s+", " ", text).strip()


class KeywordRuleScorer(BaseScorer):
    """
    Returns fraction of keywords found in the normalized response.
    Score = (# keywords present) / (# total keywords).
    Returns 0.0 if no keywords provided.
    """

    def score(self, response: str, rubric: dict) -> float:
        keywords: List[str] = rubric.get("keywords", [])
        if not keywords:
            return 0.0
        norm_resp = _normalize(response)
        hits = sum(1 for kw in keywords if _normalize(kw) in norm_resp)
        return hits / len(keywords)


def keyword_score(response: str, keywords: List[str]) -> float:
    """Convenience wrapper."""
    return KeywordRuleScorer().score(response, {"keywords": keywords})
