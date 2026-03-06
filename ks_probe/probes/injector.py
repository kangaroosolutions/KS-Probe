"""
ProbeInjector: insert probe facts into filler text at specified positions.

Each probe fact replaces a chunk of filler so total length stays constant.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Tuple

from ks_probe.probes.types import ProbeFact


@dataclass
class InjectedContext:
    """Result of probe injection — ready to send to a model."""
    prompt_text: str                              # Full prompt text
    probe_map: Dict[str, Dict[str, Any]]          # probe_id → {position_frac, char_start, char_end}
    questions: List[Dict[str, str]]               # [{"probe_id": ..., "q": ..., "gold": ..., "scoring": ...}]
    scoring_rubric: List[Dict[str, Any]]          # Full scoring metadata


def inject_probes(
    filler_text: str,
    probe_facts: List[ProbeFact],
    positions: List[float],           # fractional positions 0.0–1.0
    question_header: str = "\n\n--- QUESTIONS ---\n\n",
    count_tokens: Callable[[str], int] = lambda t: len(t.split()),
) -> InjectedContext:
    """
    Insert probe facts into filler text at the given fractional positions.

    Each probe fact replaces an equal-length chunk of filler so the total
    character count (and approximate token count) stays constant.

    Args:
        filler_text: The base filler context.
        probe_facts: Facts to embed.
        positions: Fractional positions (0.0=start, 1.0=end) for each fact.
        question_header: Separator before the Q&A block.
        count_tokens: Token counting function for the target model.

    Returns:
        InjectedContext with the assembled prompt and scoring metadata.
    """
    if len(probe_facts) != len(positions):
        raise ValueError(
            f"Must have equal number of probe_facts ({len(probe_facts)}) "
            f"and positions ({len(positions)})"
        )

    # Sort probes by position so we can insert back-to-front (avoid offset shifts)
    items = sorted(zip(positions, probe_facts), key=lambda x: x[0])
    n = len(filler_text)

    probe_map: Dict[str, Dict[str, Any]] = {}
    text = filler_text

    # Insert probes back-to-front to preserve character offsets
    for pos_frac, pf in reversed(items):
        char_pos = int(pos_frac * n)
        # Find nearest word boundary
        while char_pos < len(text) - 1 and text[char_pos] not in (" ", "\n"):
            char_pos += 1

        insertion = f"\n\n[PROBE:{pf.id}] {pf.statement}\n\n"
        # Replace a same-length chunk of filler to keep total length roughly constant
        replace_len = min(len(insertion), len(text) - char_pos)
        text = text[:char_pos] + insertion + text[char_pos + replace_len:]

        probe_map[pf.id] = {
            "position_frac": pos_frac,
            "char_start": char_pos,
            "char_end": char_pos + len(insertion),
        }

    # Build question block
    questions: List[Dict[str, str]] = []
    scoring_rubric: List[Dict[str, Any]] = []
    q_block = question_header
    q_num = 1
    for _, pf in items:
        for q in pf.questions:
            q_block += f"Q{q_num}: {q.q}\nAnswer: "
            questions.append({"probe_id": pf.id, "q_num": q_num, "q": q.q,
                               "gold": q.gold, "scoring": q.scoring})
            scoring_rubric.append({
                "q_num": q_num,
                "probe_id": pf.id,
                "probe_type": pf.type,
                "gold": q.gold,
                "scoring": q.scoring,
                "keywords": q.keywords,
                "required_conclusion": q.required_conclusion,
            })
            q_num += 1

    prompt_text = text + q_block

    return InjectedContext(
        prompt_text=prompt_text,
        probe_map=probe_map,
        questions=questions,
        scoring_rubric=scoring_rubric,
    )
