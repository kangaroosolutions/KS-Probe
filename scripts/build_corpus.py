#!/usr/bin/env python3
"""
Build domain filler text corpus for KS-Probe experiments.

Generates synthetic filler paragraphs for each domain and saves them
to data/corpus/<domain>/filler.txt. These files are used by DomainSampler.

For richer corpora, you can replace the synthetic content with real-world
text (Project Gutenberg, arXiv abstracts, legal documents, etc.) following
the same directory structure.

Usage:
    python scripts/build_corpus.py
    python scripts/build_corpus.py --paragraphs 2000 --output data/corpus
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from ks_probe.corpus.sampler import DOMAINS, _synthetic_filler

# Richer filler templates for better diversity
_EXTENDED_TEMPLATES = {
    "software_engineering": [
        "The team held a retrospective to review the sprint velocity and identified three process bottlenecks.",
        "A race condition in the async message queue was resolved by introducing a mutex around the shared state.",
        "The API gateway was refactored to support OAuth 2.0 PKCE flows for enhanced security posture.",
        "Static analysis flagged 14 unused imports and two potential null dereferences in the payment service.",
        "The database schema migration was tested against a production snapshot before being applied.",
        "Integration tests now cover the edge case where the external vendor API returns a 204 No Content.",
        "The feature flag was enabled for 5% of traffic and then gradually rolled to 100% over 72 hours.",
        "Memory profiling revealed the image resizer was allocating 3x more than expected for large JPEGs.",
        "The CI pipeline was updated to cache pip dependencies, reducing build time from 8 minutes to 2.",
        "Code review guidelines were updated to require a performance impact analysis for all DB changes.",
    ],
    "biomedical": [
        "The randomized controlled trial enrolled 847 participants across six clinical sites in three countries.",
        "Elevated TNF-alpha levels were observed in the treatment group compared to placebo at week 12.",
        "The primary endpoint of a 30% reduction in CDAI score was achieved with statistical significance.",
        "Pharmacokinetic modeling suggested a twice-daily dosing interval would maintain therapeutic plasma levels.",
        "The safety board reviewed adverse event data and recommended continuing the study without modification.",
        "Histopathological analysis of biopsy samples confirmed the presence of inflammatory infiltrates.",
        "Next-generation sequencing identified a novel variant in the BRCA2 gene with uncertain significance.",
        "The protein folding simulation required 48 hours of GPU compute time to converge to a stable conformation.",
        "Animal model data showed dose-dependent hepatotoxicity at concentrations exceeding 50 mg/kg.",
        "The meta-analysis pooled data from 23 studies and calculated a pooled odds ratio of 1.73.",
    ],
    "legal": [
        "The arbitration clause stipulates that all disputes must be resolved under ICC rules in Geneva.",
        "Force majeure events as defined in Schedule C do not excuse payment obligations under Article 12.",
        "The indemnification cap is limited to the total fees paid in the twelve months preceding the claim.",
        "Counsel for the respondent filed a motion to dismiss for lack of subject matter jurisdiction.",
        "The licensing agreement grants a non-exclusive, non-transferable right to use the software.",
        "Representations and warranties in Section 8 survive the closing date for a period of two years.",
        "The court granted summary judgment on the trademark infringement claim, finding a likelihood of confusion.",
        "Confidential information must be returned or destroyed within 30 days of the agreement's termination.",
        "The merger requires regulatory clearance from antitrust authorities in five jurisdictions.",
        "A notice of default was issued pursuant to Section 15.3 of the credit agreement.",
    ],
    "financial": [
        "The fund's Sharpe ratio improved from 0.82 to 1.14 following the portfolio rebalancing.",
        "Duration risk was reduced by rotating from long-dated treasuries into short-term instruments.",
        "The stress test assumed a 300 basis point interest rate shock and a 25% equity drawdown.",
        "Counterparty exposure to the top five banks represents 38% of the total gross notional outstanding.",
        "The impairment review identified goodwill of $4.2M that must be written down in the current period.",
        "Revenue recognition follows ASC 606, with variable consideration constrained under the expected value method.",
        "The company's leverage ratio improved to 2.1x EBITDA from 3.4x following the debt repayment.",
        "Hedge accounting was elected for the cross-currency swap under IAS 39 fair value hedge criteria.",
        "The credit rating agency placed the issuer on negative watch citing weakening free cash flow generation.",
        "Derivatives are carried at fair value with changes recognized through other comprehensive income.",
    ],
    "creative_fiction": [
        "The map showed three possible routes, each marked with a different symbol she did not recognize.",
        "He had not spoken to his brother in seven years, and now here was a letter with his handwriting.",
        "The tavern was empty except for two men playing cards in the corner and a barkeeper polishing glasses.",
        "She pressed her ear against the wooden door and heard nothing — which was worse than any sound.",
        "The machine had been running for thirty years without interruption, and no one remembered what it was for.",
        "A crow landed on the fence post and stared at her with what felt like recognition.",
        "The city looked different in winter, all its familiar landmarks rearranged by snow and silence.",
        "They had an agreement: whatever happened in the valley, they would not speak of it when they returned.",
        "The child drew the same house every day — four walls, one window, no door.",
        "An envelope arrived addressed in her own handwriting, postmarked six months in the future.",
    ],
    "conversational": [
        "Sounds good — I'll loop in the design team once we have the wireframes reviewed.",
        "Can you send me the Confluence link? I want to double-check the acceptance criteria before we close the ticket.",
        "The on-call rotation was updated last week, so check PagerDuty if you're not sure who's primary.",
        "I'll draft the postmortem by end of week and share it with the broader team for comments.",
        "We should sync with legal before we move forward on the data retention policy changes.",
        "Has anyone looked at the dashboard? The error rate spiked around 14:30 UTC and I'm trying to find root cause.",
        "The vendor confirmed they can support our SLA requirements, but we'll need a signed addendum.",
        "Let's table the architectural discussion until we have more concrete load test numbers.",
        "Just merged the hotfix to main — can someone from QA verify in staging before we push to prod?",
        "I'll take the action item to document the decision and circulate it to the stakeholders by Thursday.",
    ],
}


def build_corpus(n_paragraphs: int, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    for domain in DOMAINS:
        domain_dir = output_dir / domain
        domain_dir.mkdir(exist_ok=True)
        out_file = domain_dir / "filler.txt"

        templates = _EXTENDED_TEMPLATES.get(domain, [f"Filler text for domain {domain}."])
        paragraphs = [templates[i % len(templates)] for i in range(n_paragraphs)]

        with out_file.open("w", encoding="utf-8") as f:
            f.write("\n\n".join(paragraphs))

        print(f"  Wrote {len(paragraphs):,} paragraphs → {out_file}")


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build KS-Probe domain corpus")
    parser.add_argument("--paragraphs", "-n", type=int, default=1000,
                        help="Number of paragraphs per domain (default: 1000)")
    parser.add_argument("--output", "-o", default=str(_ROOT / "data" / "corpus"),
                        help="Output directory (default: data/corpus)")
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    output_dir = Path(args.output)
    print(f"Building corpus: {args.paragraphs} paragraphs × {len(DOMAINS)} domains → {output_dir}")
    build_corpus(args.paragraphs, output_dir)
    print("Done.")


if __name__ == "__main__":
    main()
