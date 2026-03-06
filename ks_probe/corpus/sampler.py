"""
DomainSampler — loads and samples from domain-specific filler text files.

Domain directories (under data/corpus/) contain .txt files with filler paragraphs.
Each paragraph is separated by double newlines.
"""
from __future__ import annotations

import random
from pathlib import Path
from typing import Dict, List, Optional

DOMAINS = [
    "software_engineering",
    "biomedical",
    "legal",
    "financial",
    "creative_fiction",
    "conversational",
]

_DEFAULT_CORPUS_DIR = Path(__file__).resolve().parents[2] / "data" / "corpus"


class DomainSampler:
    def __init__(
        self,
        corpus_dir: Optional[Path] = None,
        seed: int = 42,
    ) -> None:
        self.corpus_dir = corpus_dir or _DEFAULT_CORPUS_DIR
        self._rng = random.Random(seed)
        self._paragraphs: Dict[str, List[str]] = {}

    def _load_domain(self, domain: str) -> List[str]:
        if domain in self._paragraphs:
            return self._paragraphs[domain]

        domain_dir = self.corpus_dir / domain
        paragraphs: List[str] = []

        if domain_dir.exists():
            for txt_file in sorted(domain_dir.glob("*.txt")):
                text = txt_file.read_text(encoding="utf-8", errors="ignore")
                paragraphs.extend(
                    p.strip() for p in text.split("\n\n") if len(p.strip()) > 50
                )

        if not paragraphs:
            # Fallback: generate minimal synthetic filler
            paragraphs = _synthetic_filler(domain, n=200)

        self._rng.shuffle(paragraphs)
        self._paragraphs[domain] = paragraphs
        return paragraphs

    def sample_text(
        self,
        target_chars: int,
        domain_mix: Optional[Dict[str, float]] = None,
    ) -> str:
        """
        Sample filler text reaching approximately target_chars characters.

        domain_mix: {domain_name: weight} (default = equal weights for all 6 domains)
        """
        if domain_mix is None:
            domain_mix = {d: 1.0 / len(DOMAINS) for d in DOMAINS}

        # Normalise weights
        total_w = sum(domain_mix.values())
        norm = {d: w / total_w for d, w in domain_mix.items()}

        chunks: List[str] = []
        total = 0

        while total < target_chars:
            # Pick domain by weight
            domain = self._rng.choices(list(norm.keys()), weights=list(norm.values()))[0]
            pool = self._load_domain(domain)
            if not pool:
                continue
            para = self._rng.choice(pool)
            chunks.append(para)
            total += len(para) + 2  # +2 for '\n\n'

        return "\n\n".join(chunks)


def _synthetic_filler(domain: str, n: int = 200) -> List[str]:
    """Generate n minimal synthetic paragraphs for a domain (used as fallback)."""
    templates = {
        "software_engineering": [
            "The team reviewed the pull request and identified several areas for improvement in the code structure.",
            "The architecture discussion focused on the trade-offs between monolithic and microservices designs.",
            "Code coverage metrics improved after the refactoring effort, reaching 87% on the core modules.",
            "The deployment pipeline was updated to include automated security scanning at the build stage.",
        ],
        "biomedical": [
            "The study examined the correlation between inflammatory markers and disease progression rates.",
            "Patient enrollment criteria were updated to include participants with comorbid conditions.",
            "Laboratory analysis revealed elevated cytokine levels in the treatment cohort.",
            "The protocol amendment was submitted to the IRB for expedited review.",
        ],
        "legal": [
            "The parties agreed to extend the negotiation period by thirty calendar days.",
            "Clause 7.2 was amended to reflect the updated indemnification thresholds.",
            "The regulatory filing requires notarized documentation from all principal signatories.",
            "Dispute resolution shall be governed by the laws of the specified jurisdiction.",
        ],
        "financial": [
            "The quarterly earnings report showed a 12% increase in operating margin year-over-year.",
            "Portfolio rebalancing was triggered by the asset allocation drifting beyond the 5% threshold.",
            "Risk-adjusted returns for the emerging markets fund outperformed the benchmark index.",
            "The credit committee reviewed the loan application and requested additional collateral documentation.",
        ],
        "creative_fiction": [
            "The old lighthouse keeper watched the storm approach from his window, counting the seconds between lightning and thunder.",
            "She had not expected to find a letter tucked beneath the floorboards, written in a hand she recognized.",
            "The market was filled with voices in a dozen languages, all haggling over things that might or might not be real.",
            "He stood at the crossroads, knowing that whichever path he chose, it would define the rest of his years.",
        ],
        "conversational": [
            "Sure, let's sync up after the standup to go over the requirements in more detail.",
            "I'll push a draft PR by end of day so we can start the review process tomorrow morning.",
            "Does anyone have context on why the original design chose this approach over the alternative?",
            "Happy to take a look at the failing tests — can you share the error logs from the last run?",
        ],
    }
    base = templates.get(domain, ["This is a filler paragraph for domain: " + domain])
    return [base[i % len(base)] for i in range(n)]
