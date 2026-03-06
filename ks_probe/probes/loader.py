"""Load and validate the probe fact pool from JSON."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

from ks_probe.probes.types import ProbePool

_DEFAULT_PATH = Path(__file__).resolve().parents[2] / "data" / "probes" / "probe_facts.json"
_SCHEMA_PATH = Path(__file__).resolve().parents[2] / "data" / "probes" / "probe_schema.json"

_pool_cache: Optional[ProbePool] = None


def load_probe_pool(path: Optional[Path] = None, use_cache: bool = True) -> ProbePool:
    """Load and validate the probe fact pool. Results are cached."""
    global _pool_cache
    if use_cache and _pool_cache is not None:
        return _pool_cache

    p = path or _DEFAULT_PATH
    with p.open("r", encoding="utf-8") as f:
        raw = json.load(f)

    pool = ProbePool(**raw)
    if use_cache:
        _pool_cache = pool
    return pool


def validate_probe_file(path: Optional[Path] = None) -> dict:
    """
    Validate the probe facts file and return a summary.

    Returns:
        dict with keys: total, by_type, missing_fields, warnings
    """
    pool = load_probe_pool(path, use_cache=False)
    ids = [pf.id for pf in pool.probe_facts]
    duplicates = [x for x in ids if ids.count(x) > 1]

    summary = {
        "total": len(pool.probe_facts),
        "by_type": {
            "explicit_recall": len(pool.by_type("explicit_recall")),
            "grounded_reasoning": len(pool.by_type("grounded_reasoning")),
            "implicit_context": len(pool.by_type("implicit_context")),
        },
        "total_questions": sum(len(pf.questions) for pf in pool.probe_facts),
        "duplicates": duplicates,
        "warnings": [],
    }

    if duplicates:
        summary["warnings"].append(f"Duplicate IDs found: {duplicates}")

    for pf in pool.probe_facts:
        if len(pf.questions) != 5:
            summary["warnings"].append(f"{pf.id} has {len(pf.questions)} questions (expected 5)")
        if pf.type == "grounded_reasoning":
            for q in pf.questions:
                if q.scoring == "keyword_rule" and not q.keywords:
                    summary["warnings"].append(f"{pf.id} keyword_rule question missing keywords")

    return summary
