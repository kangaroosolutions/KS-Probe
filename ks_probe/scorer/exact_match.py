"""Exact match scorer — normalized substring match."""
from __future__ import annotations

import re
import unicodedata


def _normalize(text: str) -> str:
    """Lowercase, strip punctuation, collapse whitespace."""
    text = unicodedata.normalize("NFKD", text).lower()
    text = re.sub(r"[^\w\s]", " ", text)  # remove punctuation
    text = re.sub(r"\s+", " ", text).strip()
    # Remove common articles for English robustness
    text = re.sub(r"\b(a|an|the)\b", "", text)
    return text.strip()


def exact_match_score(response: str, gold: str) -> float:
    """
    Return 1.0 if the normalized gold answer appears anywhere in the normalized response.
    Return 0.0 otherwise.
    """
    if not gold.strip():
        return 1.0
    norm_gold = _normalize(gold)
    norm_resp = _normalize(response)
    return 1.0 if norm_gold in norm_resp else 0.0
