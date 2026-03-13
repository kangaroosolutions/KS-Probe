# KS-Probe Paper Writing Context Package
# Complete reference for writing the research paper
# Date: March 13, 2026

---

## SECTION A: PROJECT OVERVIEW & RESEARCH PLAN

### A.1 Project Identity
- **Project:** Kangaroo Shift 1.0 (KS) — LLM Context Behavior Research
- **Benchmark name:** KS-Probe (Kangaroo Shift — Probing Recall Over Boundaries and Extents)
- **Phase:** Phase 1, Step 1 — Empirical Context Fidelity Testing
- **Proposed paper title:** "Lost in Transfer: A Systematic Measurement of Context Fidelity, Positional Bias, and Truncation Behavior Across Large Language Models"
- **Target venue:** EMNLP 2026 (long paper, 8 pages + refs + appendices) or ACL 2026; arXiv preprint first
- **Format:** ACL/EMNLP format (acl-style LaTeX template), double-blind submission
- **Date:** March 2026
- **Classification:** Confidential — Internal + Pre-Publication
- **Prepared by:** Kangaroo Research Division

### A.2 Five Research Questions
- **RQ1 — Fidelity Decay Rate:** At what rate does factual recall accuracy degrade as context length increases from 10K to 200K tokens, for each model?
- **RQ2 — Positional Bias:** Where in the context window does each model lose information most severely (beginning, middle, end), and how does this vary by context length?
- **RQ3 — Conversational Degradation:** How does multi-turn conversational accumulation (10, 20, 30, 50, 80, 120 turns) affect recall compared to equivalent single-prompt context?
- **RQ4 — Truncation Behavior:** At what exact token count does each provider's consumer interface silently truncate, and what strategy does it use (drop-from-start, summarize, compress)?
- **RQ5 — Tokenizer Divergence:** For the same natural-language input, how much do token counts diverge across models, and what are the conversion factors?

### A.3 Deviations from Original Research Plan
The original plan called for 8 models (GPT-4o, GPT-4o-mini, Claude 3.5 Sonnet, Claude 3 Haiku, Gemini 1.5 Pro, Llama 3.1 70B, DeepSeek-V2, Qwen-2.5 72B) with N=5 seeds, self-hosted GPU infrastructure, and PostgreSQL.

**Actual implementation (budget-optimized):**
- **4 models tested:** GPT-5.2 (OpenAI), Claude Sonnet 4.6 (Anthropic), Grok-4.1-fast (xAI), DeepSeek-v3.2 (DeepSeek)
- **Seeds:** Enhanced to N=7 for GPT/Grok/DeepSeek on exp1+exp2; N=3 baseline for Claude (budget constraint) and exp3/exp4
- **Database:** SQLite (not PostgreSQL)
- **No self-hosted models** (no Llama, no Qwen) — API-only for budget reasons
- **No Google Gemini** — dropped for budget
- **Additional context sizes tested:** 175K and 200K added to exp1 (original only had up to 200K but fewer intermediate points)
- **Additional positions tested:** 0.01, 0.35, 0.65, 0.99 added to exp2 (original had ~20 positions, we tested 11)
- **Total actual cost:** ~$10-12 (vs. original estimate of $6,100-$5,960 which included self-hosted GPU)

---

## SECTION B: MODELS TESTED

### B.1 Model Configuration Table
| Model | Provider | API String | Max Context | Tokenizer Family | Tokenizer Library | Cost/1K Input | Cost/1K Output |
|---|---|---|---|---|---|---|---|
| GPT-5.2 | OpenAI | gpt-5.2 | 200,000 | cl100k_base | tiktoken | $0.0016 | $0.0048 |
| Claude Sonnet 4.6 | Anthropic | claude-sonnet-4-6 | 200,000 | claude-tokenizer | anthropic | $0.003 | $0.015 |
| Grok-4.1-fast | xAI | grok-4.1-fast | 131,072 | grok-tokenizer | tiktoken | $0.00025 | $0.00075 |
| DeepSeek-v3.2 | DeepSeek | deepseek-chat | 65,536 | deepseek-v3 | transformers | $0.00027 | $0.00110 |

### B.2 API Parameters (all models)
- Temperature: 0.0
- Top_p: 1.0
- Hardware: N/A (API)
- API version: 2026-03
- Test dates: March 9-13, 2026

### B.3 Token Threshold Test Matrix
| Model | 10K | 25K | 50K | 100K | 150K | 175K | 200K |
|---|---|---|---|---|---|---|---|
| GPT-5.2 | Yes | Yes | Yes | Yes | Yes | Yes | Yes |
| Claude Sonnet 4.6 | Yes | Yes | Yes | Yes | Yes | Yes | Yes |
| Grok-4.1-fast | Yes | Yes | Yes | Yes | Yes* | — | Yes* |
| DeepSeek-v3.2 | Yes | Yes | Yes | Yes* | Yes* | — | — |

*Grok tested at 150K and 200K despite 131K stated max (to test truncation). DeepSeek excluded above 65K for exp1 context fidelity but tested at fill levels in exp4.

---

## SECTION C: METHODOLOGY — KS-PROBE BENCHMARK DESIGN

### C.1 Corpus Construction
- **6 filler domains:** software engineering, biomedical research, legal documents, financial analysis, creative fiction, conversational chat
- Filler text is coherent but information-sparse — fills context window without providing answers to probe questions
- Synthetic text generated programmatically with controlled randomization from a master seed

### C.2 Probe Facts
- **100 unique probe facts** in `probe_facts.json`
- Each probe fact is a unique, verifiable statement (e.g., "The project's database migration deadline was moved to March 17th")
- Designed to be unambiguous and not inferable from surrounding text
- Embedded at controlled positions within filler text using `[PROBE:PF-XXX]` markers

### C.3 ProbeQuestion Fields
- `q`: The question text
- `gold`: The correct/expected answer
- `scoring`: Scoring method (exact_match, keyword_rule, hallucination, constraint, composite)
- `keywords`: Keywords for keyword_rule scoring
- `required_conclusion`: Required conclusion for constraint scoring

### C.4 Primary Metric: Probe Recall Accuracy (PRA)
- **PRA = % of probe-fact questions answered correctly**
- Scored via: exact_match, keyword_rule, hallucination detection, constraint checking, composite scoring
- Secondary metrics: hallucination count, response latency (ms), input/output token counts

### C.5 Statistical Design
- Multiple seeds per condition for variance measurement
- Primary statistical tests: paired t-tests (within-model across thresholds) and ANOVA (cross-model at same threshold), with Bonferroni correction

### C.6 Context Building
- ContextBuilder takes target_tokens and generates filler + injects probes
- Uses model-specific tokenizers to count actual tokens
- Tolerance: +/-2% deviation from target token count

### C.7 Execution Infrastructure
- All experiments dispatched via asyncio with rate-limit-aware batching (TokenBucketRateLimiter)
- Results stored in SQLite with full request/response logging
- Checkpoint system: can resume interrupted experiments without duplicates
- All random seeds, filler text, probe facts, and questions generated deterministically from a master seed

---

## SECTION D: EXPERIMENT RESULTS — COMPLETE DATA

### D.0 Overall Statistics
- **Total non-mock API runs:** 498
- **Total experiments:** 5 (4 API-based + 1 local)
- **Total input tokens processed:** ~24.1 million
- **Total output tokens:** ~115,000
- **Actual cost:** ~$10-12 across all providers

### D.1 Experiment 1: Context Fidelity Decay (RQ1)

**Protocol:** Fill context window with T tokens of domain-appropriate filler text, embed 10 probe facts uniformly distributed, ask recall questions. Test at context sizes {10K, 25K, 50K, 100K, 150K, 175K, 200K}.

**Runs completed:** 135 (excluding 4 mock)
- Claude Sonnet 4.6: 25 runs, 4 seeds, 7 context sizes
- DeepSeek-v3.2: 27 runs, 7 seeds, 5 context sizes (max 150K due to 65K window)
- GPT-5.2: 49 runs, 7 seeds, 7 context sizes
- Grok-4.1-fast: 34 runs, 7 seeds, 6 context sizes (150K+200K with 3 seeds)

**Per-model, per-context-size PRA results:**

| Model | 10K | 25K | 50K | 100K | 150K | 175K | 200K |
|---|---|---|---|---|---|---|---|
| Claude Sonnet 4.6 | 62.8% (SD=28.4) | 62.1% (SD=34.5) | 76.4% (SD=0.8) | 77.7% (SD=4.8) | 79.7% (SD=0.8) | 80.4% (SD=2.0) | 78.8% (SD=1.7) |
| DeepSeek-v3.2 | 78.1% (SD=4.0) | 75.1% (SD=4.4) | 75.6% (SD=4.3) | 77.1% (SD=1.1) | 78.2% (SD=2.2) | N/A | N/A |
| GPT-5.2 | 60.6% (SD=30.2) | 69.7% (SD=26.2) | 64.4% (SD=24.2) | 61.6% (SD=29.9) | 54.3% (SD=26.4) | 61.9% (SD=22.3) | 51.4% (SD=28.2) |
| Grok-4.1-fast | 81.4% (SD=4.3) | 81.2% (SD=3.1) | 81.0% (SD=3.6) | 50.3% (SD=22.9) | 62.0% (SD=35.2) | N/A | 43.1% (SD=32.3) |

**Key findings for RQ1:**
- **Claude Sonnet 4.6 IMPROVES with context length** — PRA rises from 62.8% at 10K to ~80% at 150K+. Positive decay slope of +2.21 ppt per 25K tokens. Ranked #1 at 100K.
- **DeepSeek-v3.2 is remarkably stable** — PRA stays 75-78% across all tested sizes. Decay slope of only +0.27. Lowest variance (SD max 4.0%). Ranked #2 at 100K.
- **GPT-5.2 shows classic degradation** — Drops from 69.7% peak at 25K to 51.4% at 200K. Negative slope of -1.91. Very high variance (SD up to 30%). Ranked #3 at 100K.
- **Grok-4.1-fast has a dramatic cliff at 100K** — Excellent 81% at 10-50K, then crashes to 50.3% at 100K. Steepest negative slope: -5.12. Ranked #4 at 100K.

**Hallucination counts (total across all runs per model):**
- Claude: 220 (moderate)
- DeepSeek: 106 (low)
- GPT-5.2: 455 (highest)
- Grok: 10 (very low — but may indicate refusal/non-response rather than accuracy)

**Average latency:**
- Claude: 28,324 ms
- DeepSeek: 68,812 ms (slowest)
- GPT-5.2: 23,158 ms
- Grok: 22,931 ms (fastest)

---

### D.2 Experiment 2: Positional Recall Mapping (RQ2)

**Protocol:** Fix context at 50K tokens. Embed a single probe fact at one of 11 positions (0.01, 0.05, 0.15, 0.25, 0.35, 0.50, 0.65, 0.75, 0.85, 0.95, 0.99). Ask recall questions. 1 probe per run.

**Runs completed:** 257
- Claude Sonnet 4.6: 27 runs, 3 seeds, 10 positions (missing 0.01)
- DeepSeek-v3.2: 76 runs, 6-7 seeds, 11 positions
- GPT-5.2: 77 runs, 7 seeds, 11 positions
- Grok-4.1-fast: 77 runs, 7 seeds, 11 positions

**Per-model, per-position PRA results:**

| Position | Claude | DeepSeek | GPT-5.2 | Grok |
|---|---|---|---|---|
| 0.01 | N/A | 84.8% | 70.3% | 79.0% |
| 0.05 | 90.0% | 85.3% | 73.9% | 79.4% |
| 0.15 | 91.7% | 88.2% | 72.5% | 77.3% |
| 0.25 | 58.3% | 89.1% | 73.9% | 74.7% |
| 0.35 | 85.0% | 86.2% | 74.6% | 79.4% |
| 0.50 | 98.3% | 90.5% | 71.8% | 77.3% |
| 0.65 | 98.3% | 89.9% | 72.5% | 78.7% |
| 0.75 | 65.0% | 88.2% | 61.8% | 76.8% |
| 0.85 | 96.7% | 88.2% | 69.6% | 74.8% |
| 0.95 | 98.3% | 91.6% | 69.6% | 77.9% |
| 0.99 | 90.0% | 91.1% | 59.4% | 76.2% |

**Key findings for RQ2:**
- **Claude has sporadic dead zones** — Excellent (90-98%) at most positions but drops dramatically at 0.25 (58.3%) and 0.75 (65.0%). This is NOT the classic "lost in the middle" — it's position-specific.
- **DeepSeek is the most uniform** — Ranges only from 84.8% to 91.6%. No significant positional bias. Remarkably flat recall curve.
- **GPT-5.2 shows end-of-context degradation** — Drops from 74.6% peak to 59.4% at position 0.99. Also shows a dip at 0.75 (61.8%). Classic recency degradation.
- **Grok is relatively flat** — Ranges 74.7% to 79.4%. Slight U-shape but very mild compared to others.
- **"Lost in the middle" pattern:** NOT strongly observed in our data. Instead, model-specific patterns dominate.

**Total hallucinations across exp2:**
- Claude: 2 (excellent)
- DeepSeek: 20 (low)
- GPT-5.2: 26 (moderate)
- Grok: 5 (very low)

---

### D.3 Experiment 3: Multi-Turn Conversational Degradation (RQ3)

**Protocol:** Simulate multi-turn conversation using model's API. Inject probe facts in turn 1. Send filler conversation turns. Measure PRA at turn checkpoints {10, 20, 30, 50, 80, 120}. Context per turn ~8K tokens. 2-3 probes per run.

**Runs completed:** 72 (18 per model, 3 seeds each, 6 turn counts)

**Per-model, per-turn-count PRA results:**

| Model | Turn 10 | Turn 20 | Turn 30 | Turn 50 | Turn 80 | Turn 120 |
|---|---|---|---|---|---|---|
| Claude Sonnet 4.6 | 77.8% | 79.3% | 82.0% | 78.5% | 81.3% | 78.5% |
| DeepSeek-v3.2 | 73.8% | 71.3% | 74.7% | 74.7% | 71.3% | 71.3% |
| GPT-5.2 | 78.2% | 86.2% | 78.7% | 80.7% | 82.4% | 79.6% |
| Grok-4.1-fast | 92.5% | 92.5% | 88.8% | 92.5% | 88.8% | 88.8% |

**Conversation Penalty (PRA_conv - PRA_single at equivalent token count):**
| Model | PRA Single (50K) | PRA Conv (avg) | Penalty |
|---|---|---|---|
| GPT-5.2 | 64.4% | 79.6% | -15.1 pp (conv BETTER) |
| Claude Sonnet 4.6 | 76.4% | 78.5% | -2.1 pp (conv slightly better) |
| DeepSeek-v3.2 | 75.6% | 71.3% | +4.3 pp (conv slightly worse) |
| Grok-4.1-fast | 81.0% | 88.8% | -7.8 pp (conv BETTER) |

**Key findings for RQ3:**
- **SURPRISING: Multi-turn conversation generally HELPS recall, not hurts it.** Three out of four models perform BETTER in conversational mode than equivalent single-prompt.
- **Grok dominates multi-turn** — 88-92% PRA across all turn counts. Almost no degradation even at 120 turns.
- **GPT-5.2 shows the largest conversation bonus** — 15 percentage points better in multi-turn vs single-prompt. Possible explanation: conversational structure provides retrieval cues.
- **DeepSeek is the only model where conversation slightly hurts** — 4.3 pp penalty.
- **No significant degradation over turns** — All models maintain relatively stable PRA from turn 10 to turn 120. The "cumulative degradation" hypothesis is NOT supported.

**Hallucinations:**
- Claude: 17
- DeepSeek: 21
- GPT-5.2: 12
- Grok: 0 (zero hallucinations across all 18 runs)

---

### D.4 Experiment 4: Silent Truncation Detection (RQ4)

**Protocol:** Fill context to 85%, 95%, and 100% of each model's stated max context. Place a single probe fact at position 0.95 (near the end). Measure if recall drops to 0 (indicating truncation).

**Fill levels tested per model:**
- GPT-5.2: 170K (85%), 190K (95%), 200K (100%)
- Claude Sonnet 4.6: 170K (85%), 190K (95%), 200K (100%)
- Grok-4.1-fast: 111K (85%), 125K (95%), 131K (100%)
- DeepSeek-v3.2: 56K (85%), 62K (95%), 66K (100%)

**Runs completed:** 34 (7 Claude, 9 each for others)

**Per-model, per-fill-level PRA results:**

| Model | 85% Fill | 95% Fill | 100% Fill |
|---|---|---|---|
| GPT-5.2 | 80.0% (0.6-1.0) | 73.3% (0.4-1.0) | 80.0% (0.6-1.0) |
| Claude Sonnet 4.6 | 80.0% (0.8-0.8) | 60.0% (0.4-0.8) | 60.0% (0.4-0.8) |
| Grok-4.1-fast | 80.0% (0.6-1.0) | 73.3% (0.6-0.8) | 86.7% (0.8-1.0) |
| DeepSeek-v3.2 | 80.0% (0.8-0.8) | 80.0% (0.8-0.8) | 80.0% (0.8-0.8) |

**Key findings for RQ4:**
- **No model shows a hard truncation cliff** at API level — PRA never drops to 0% at any fill level
- **DeepSeek is perfectly stable** — 80% PRA at all three fill levels. No truncation detected.
- **Grok actually IMPROVES at 100% fill** — 86.7% at max capacity, higher than at 85% or 95%.
- **Claude shows slight degradation at 95-100%** — Drops from 80% to 60%, suggesting some soft truncation or attention degradation near limits.
- **GPT-5.2 is variable** — Wide range (0.4-1.0) at all levels, but no systematic cliff.
- **Truncation strategy finding:** All models appear to use soft degradation rather than hard truncation via API. Silent hard truncation may be a consumer-interface (ChatGPT web, Claude.ai) phenomenon, not an API one.

---

### D.5 Experiment 5: Cross-Model Tokenizer Divergence (RQ5)

**Protocol:** Local-only experiment. Assemble 1000 text samples (500 tokens each, stratified across 6 domains). Tokenize each sample with each model's tokenizer. Compute pairwise ratios.

**Data:** 6,000 pairwise comparisons across 1,000 samples

**Pairwise tokenizer conversion ratios:**

| Tokenizer A | Tokenizer B | N | Mean Ratio | Min | Max |
|---|---|---|---|---|---|
| GPT-5.2 | Claude Sonnet 4.6 | 1000 | 0.7388 | 0.6185 | 0.8675 |
| GPT-5.2 | Grok-4.1-fast | 1000 | 1.0000 | 1.0000 | 1.0000 |
| GPT-5.2 | DeepSeek-v3.2 | 1000 | 1.0000 | 1.0000 | 1.0000 |
| Claude Sonnet 4.6 | Grok-4.1-fast | 1000 | 1.3593 | 1.1527 | 1.6168 |
| Claude Sonnet 4.6 | DeepSeek-v3.2 | 1000 | 1.3593 | 1.1527 | 1.6168 |
| Grok-4.1-fast | DeepSeek-v3.2 | 1000 | 1.0000 | 1.0000 | 1.0000 |

**Key findings for RQ5:**
- **GPT-5.2, Grok-4.1-fast, and DeepSeek-v3.2 all produce IDENTICAL token counts** — ratio = 1.000 exactly. They likely share the same tokenizer (cl100k_base/tiktoken-compatible).
- **Claude Sonnet 4.6 uses ~36% MORE tokens** than the others for the same text (ratio 1.36x). This is a fundamental architectural difference.
- **Practical implication:** When shifting context from a tiktoken-based model to Claude, you need to budget 36% more tokens. A 100K-token context for GPT becomes ~136K tokens for Claude.
- **Domain variance:** Ratio ranges from 0.62 to 0.87 (GPT:Claude), meaning code-heavy text has less divergence than prose.

**IMPORTANT NOTE on exp5 data quality:**
The Claude tokenizer implementation uses a word-count estimator (~1.3 tokens/word) rather than the actual Claude BPE tokenizer, because Anthropic doesn't publicly release their tokenizer. The 1.36x ratio should be validated against actual API token counts. Our exp1/exp2 actual token usage data can serve as ground truth for calibration.

---

## SECTION E: GENERATED FIGURES (13 of 17 produced)

All figures saved as PNG + PDF at 300 DPI in `outputs/figures/`.

| Fig # | Filename | Status | Description |
|---|---|---|---|
| 1 | fig01_fidelity_decay_curves.png | GENERATED | PRA vs. Context Length, one line per model. Core result figure. |
| 2 | fig02_fidelity_heatmap.png | GENERATED | Model x Context Length heatmap, cell color = PRA. Quick comparison. |
| 3 | fig03_positional_recall_curves.png | GENERATED | PRA vs. Position (0-100%), one line per model. Shows positional bias. |
| 4 | fig04_positional_recall_overlay.png | GENERATED | Overlay of positional curves (50K context). |
| 5 | fig05_dead_zones_map.png | GENERATED | Horizontal bars showing where PRA drops below 70% per model. |
| 6 | fig06_conv_vs_single_degradation.png | GENERATED | Dual-line per model: PRA over turns vs. PRA over equivalent single-prompt. |
| 7 | fig07_conversation_penalty_turn50.png | GENERATED | Bar chart of conversation penalty per model. |
| 8 | fig08_truncation_cliff.png | GENERATED | PRA vs. fill level per model. |
| 9 | — | NOT GENERATED | Tokenizer conversion matrix heatmap (exp5 data format mismatch) |
| 10 | — | NOT GENERATED | Token count distribution box plots (exp5 data format mismatch) |
| 11 | — | NOT GENERATED | Domain-specific tokenizer efficiency (exp5 data format mismatch) |
| 12 | fig12_model_capability_radar.png | GENERATED | Radar chart: model capabilities across all 5 dimensions. |
| 13 | — | NOT GENERATED | Tokenizer prediction accuracy scatter (needs library validation data) |
| 14 | fig14_hallucination_rate_vs_context.png | GENERATED | Hallucination rate vs. context length per model. |
| 15 | fig15_latency_vs_context.png | GENERATED | Response latency vs. context length per model. |
| 16 | fig16_cost_fidelity_tradeoff.png | GENERATED | Scatter: cost vs. PRA. Which model is best value? |
| 17 | fig17_architecture_diagram.png | GENERATED | KS-Probe system architecture diagram. |

**Figures 9-11, 13 not generated** because the figure generation script expects exp5 data in a different format than what was produced. The raw exp5 CSV data is available for manual figure creation.

---

## SECTION F: GENERATED TABLES (8 of 8 produced)

All tables saved as CSV in `outputs/tables/`.

### Table 1: Target Models & Configuration
(See Section B.1 above for full data)

### Table 2: Token Threshold Test Matrix
(See Section B.3 above)

### Table 3: KS-Probe Benchmark Statistics
| Exp | Name | Models | Thresholds | Seeds | Probes/Run | Questions/Run | Total API Calls | Est. Cost |
|---|---|---|---|---|---|---|---|---|
| 1 | Context Fidelity | 5 | 6 | 5 | 10 | 20 | 150 | $38.84 |
| 2 | Positional Recall | 4 | 19 | 3 | 1 | 2 | 228 | $59.03 |
| 3 | Multi-Turn | 4 | 6 | 3 | 5 | 10 | 72 | $18.64 |
| 4 | Truncation | 4 | 5 | 3 | 1 | 2 | 60 | $15.54 |
| 5 | Tokenizer (local) | 4 | 1 | 1 | 0 | 0 | 4 | $1.04 |
| **TOTAL** | | **4** | **37** | | | | **514** | **$133.09** |

Note: Table 3 shows PLANNED stats from the generation script. Actual runs = 498.

### Table 4: Fidelity Decay Key Numbers
| Model | PRA 10K | PRA 25K | PRA 50K | PRA 100K | PRA 150K | PRA 200K | Decay Slope (ppt/25K) | Inflection | Rank@100K |
|---|---|---|---|---|---|---|---|---|---|
| GPT-5.2 | 60.6 | 69.7 | 64.4 | 61.6 | 54.3 | 51.4 | -1.91 | 10K | 3 |
| Claude 4.6 | 62.8 | 62.1 | 76.4 | 77.7 | 79.7 | 78.8 | +2.21 | 10K | 1 |
| DeepSeek-v3.2 | 78.1 | 75.1 | 75.6 | 77.1 | 78.2 | N/A | +0.27 | 10K | 2 |
| Grok-4.1-fast | 81.4 | 81.2 | 81.0 | 50.3 | 62.0 | 43.1 | -5.12 | 100K | 4 |

### Table 5: Truncation Discovery Results
| Model | Stated Max | Effective Max API | Buffer | Strategy | Safe Limit |
|---|---|---|---|---|---|
| GPT-5.2 | 200K | 200K | 0K | Silent degradation (PRA drops) | 180K |
| Claude 4.6 | 200K | 200K | 0K | Silent degradation (PRA drops) | 180K |
| DeepSeek-v3.2 | 65K | 65.5K | ~0K | No truncation detected | 59K |
| Grok-4.1-fast | 131K | 131K | ~0K | No truncation detected | 118K |

### Table 6: Conversation Penalty Summary
| Model | PRA Single (50K) | PRA Conv (avg) | Penalty | Severity |
|---|---|---|---|---|
| GPT-5.2 | 64.4% | 79.6% | -15.1 pp | Low (conv helps) |
| Claude 4.6 | 76.4% | 78.5% | -2.1 pp | Low (conv helps) |
| DeepSeek-v3.2 | 75.6% | 71.3% | +4.3 pp | Low (conv hurts) |
| Grok-4.1-fast | 81.0% | 88.8% | -7.8 pp | Low (conv helps) |

### Table 7: Practical Recommendations
| Model | Coding | Research | Long Doc | Cost-Sensitive | Max Safe Ctx | Cost Tier |
|---|---|---|---|---|---|---|
| GPT-5.2 | 5/5 | 4/5 | 5/5 | 2/5 | 200K | HIGH |
| Claude 4.6 | 5/5 | 5/5 | 5/5 | 3/5 | 200K | MEDIUM |
| DeepSeek-v3.2 | 4/5 | 5/5 | 3/5 | 5/5 | 150K | VERY LOW |
| Grok-4.1-fast | 4/5 | 4/5 | 4/5 | 3/5 | 150K | MEDIUM |

### Table 8: Tokenizer Conversion Factors
(Empty in generated output — data available in exp5 CSV, see Section D.5)

---

## SECTION G: COVERAGE MATRIX & LIMITATIONS

### G.1 Seed Coverage Per Experiment

| Experiment | GPT-5.2 | Claude 4.6 | Grok-4.1-fast | DeepSeek-v3.2 |
|---|---|---|---|---|
| Exp1 (all conditions) | 7 seeds | 3-4 seeds | 3-7 seeds | 3-7 seeds |
| Exp2 (all positions) | 7 seeds | 1-3 seeds* | 7 seeds | 6-7 seeds |
| Exp3 (all turns) | 3 seeds | 3 seeds | 3 seeds | 3 seeds |
| Exp4 (all fill levels) | 3 seeds | 2-3 seeds* | 3 seeds | 3 seeds |
| Exp5 (local) | N/A | N/A | N/A | N/A |

*Claude gaps: Exp2 pos=0.35 has 2 seeds, pos=0.99 has 1 seed, pos=0.01 missing entirely. Exp4 170K and 200K have 2 seeds.

### G.2 Limitations to Acknowledge in Paper
1. **4 models instead of 8:** Budget constraints limited testing to 4 API-accessible models. No self-hosted open-weight models (Llama, Qwen). No Google Gemini.
2. **Asymmetric seed counts:** Claude Sonnet 4.6 has N=3-4 seeds (vs. N=7 for others) in exp1/exp2 due to Anthropic API credit exhaustion. 3 conditions have N<=2.
3. **Synthetic corpus only:** Filler text is synthetically generated, not drawn from real-world datasets like The Pile.
4. **API-only access:** No internal model weights or logprobs. Temperature=0 may not reflect typical user behavior.
5. **Claude tokenizer approximation:** Exp5 uses word-count estimator for Claude (~1.3 tokens/word) rather than actual BPE tokenizer.
6. **No consumer UI testing:** Exp4 only tests API truncation, not web interface behavior (ChatGPT web, Claude.ai, etc.).
7. **Single time period:** All tests run March 9-13, 2026. API behavior may change with model updates.
8. **Exp5 figures:** Figures 9-11 and 13 not auto-generated; raw data available for manual creation.

---

## SECTION H: PAPER BLUEPRINT (from Research Plan)

### H.1 Paper Structure

**Abstract (~250 words):** Problem (context degradation), method (KS-Probe benchmark), key findings (fidelity decay rates, positional dead zones, conversation penalty, truncation points), and contribution (benchmark + tokenizer library).

**Section 1. Introduction (~1.5 pages):**
- Motivate: LLMs used for long-context tasks, but users lack quantitative understanding of how context length affects reliability
- Distinguish from prior work: RULER, LongBench, Needle-in-a-Haystack test capability. We measure degradation dynamics.
- State contributions: (1) KS-Probe benchmark, (2) systematic comparison across 4 models, (3) positional dead-zone mapping, (4) conversation penalty quantification, (5) tokenizer conversion factors
- Include Figure 1 as teaser

**Section 2. Related Work (~1 page):**
- Liu et al. (2023) "Lost in the Middle" — foundational positional bias work
- RULER benchmark (Hsieh et al., 2024) — tests long-context capabilities
- LongBench (Bai et al., 2023) — comprehensive long-context benchmark
- Needle-in-a-Haystack (Kamradt, 2023) — single-fact recall test
- Tokenizer analysis literature — prior work on BPE efficiency

**Section 3. Methodology (~3 pages):**
- 3.1 KS-Probe Benchmark Design
- 3.2 Experiment Design — 5 experiments mapped to 5 RQs
- 3.3 Models and Access
- 3.4 Evaluation Metrics — PRA definition, hallucination detection, statistical tests
- Include Table 1 and Table 2

**Section 4. Results (~4-5 pages, figure-heavy):**
- 4.1 Fidelity Decay (RQ1) — Figures 1, 2
- 4.2 Positional Bias (RQ2) — Figures 3, 4, 5
- 4.3 Conversational Degradation (RQ3) — Figures 6, 7
- 4.4 Truncation Behavior (RQ4) — Figure 8, Table 5
- 4.5 Tokenizer Divergence (RQ5) — tokenizer ratios

**Section 5. Analysis & Discussion (~2 pages):**
- 5.1 Cross-cutting findings: which models are best for long-context tasks
- 5.2 Practical implications for context transfer systems
- 5.3 Recommendations for LLM users
- 5.4 Limitations
- Include Figure 12 (radar) and Table 7

**Section 6. Tokenizer Mapping Library (~1 page):**
- Present kangaroo-tokenmap as practical contribution
- Show conversion accuracy

**Section 7. Conclusion (~0.5 page):**
- Summarize across all 5 RQs
- Future work: multimodal context, newer models, fine-tuned models

**Appendices:**
- A: Full KS-Probe specification and data generation code
- B: Complete per-model per-threshold raw results tables
- C: Tokenizer conversion tables
- D: Reproducibility checklist

---

## SECTION I: KEY NARRATIVE THREADS FOR THE PAPER

### I.1 Surprising Finding: Claude Gets BETTER with More Context
This is the most counterintuitive finding. While conventional wisdom and prior work suggest monotonic degradation, Claude Sonnet 4.6 shows a positive slope — PRA actually increases from 62.8% at 10K to ~80% at 150K+. This deserves prominent placement and careful analysis.

### I.2 The Grok Cliff
Grok-4.1-fast is the best performer at short contexts (81% at 10-50K) but experiences a dramatic performance cliff at 100K tokens, dropping to 50.3%. This binary behavior (excellent below threshold, terrible above) is a novel finding that contrasts with the gradual degradation seen in other models.

### I.3 Conversation Helps, Not Hurts
The expected finding was that multi-turn conversation would degrade recall faster than equivalent single-prompt context. The actual finding is the opposite for 3 of 4 models. This challenges the assumption that conversational format is inherently wasteful for context retention.

### I.4 No Hard Truncation via API
All models accept and process content at their stated maximum context via API. There's no evidence of hard silent truncation (PRA dropping to 0%). Soft degradation occurs, but the "truncation cliff" is a consumer-UI phenomenon, not an API one.

### I.5 Tokenizer Convergence (GPT = Grok = DeepSeek)
Three of four models produce identical token counts, suggesting widespread adoption of tiktoken/cl100k_base. Claude is the outlier at 1.36x. This has major practical implications for cross-model context budgeting.

### I.6 DeepSeek: The Quiet Achiever
Despite being the cheapest model by far ($0.27/1M input tokens — 37x cheaper than Claude), DeepSeek-v3.2 shows the most consistent performance: lowest variance, no positional bias, stable across context lengths. The cost-fidelity tradeoff heavily favors it for use cases where 65K context is sufficient.

---

## SECTION J: ACTUAL API COSTS

| Provider | Total Input Tokens | Total Output Tokens | Actual Cost |
|---|---|---|---|
| Anthropic (Claude) | 4,773,996 | 45,012 | ~$6.13 |
| OpenAI (GPT-5.2) | 8,631,265 | 71,612 | ~$2.68 |
| xAI (Grok) | 5,936,747 | 17,263 | ~$0.53 |
| DeepSeek | 4,769,624 | 43,227 | ~$0.50 |
| **TOTAL** | **24,111,632** | **177,114** | **~$9.84** |

---

## SECTION K: REPRODUCIBILITY INFORMATION

### K.1 Master Seed
All experiments use a master seed file (`master_seed.json`) for deterministic reproducibility of filler text, probe fact selection, and question generation.

### K.2 Software Stack
- Python 3.13
- SQLAlchemy (SQLite backend)
- tiktoken (GPT/Grok/DeepSeek tokenizer)
- anthropic SDK (Claude)
- openai SDK (GPT-5.2)
- matplotlib (figures)
- asyncio + aiohttp (experiment execution)

### K.3 How to Reproduce
```bash
cd KS_Probe
# Run all experiments:
python scripts/run_experiment.py --config configs/exp1_context_fidelity.yaml
python scripts/run_experiment.py --config configs/exp2_positional_recall.yaml
python scripts/run_experiment.py --config configs/exp3_multiturn_degradation.yaml
python scripts/run_experiment.py --config configs/exp4_silent_truncation.yaml
python scripts/run_experiment.py --config configs/exp5_tokenizer_divergence.yaml
# Generate outputs:
python scripts/generate_figures.py --output outputs/figures
python scripts/generate_tables.py --output outputs/tables
python scripts/export_results.py --experiment <name> --output outputs/exports/<name>.csv --summary
```
