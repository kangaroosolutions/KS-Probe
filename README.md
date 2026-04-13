# KS-Probe: Kangaroo Shift LLM Context Retention Benchmark

A systematic benchmark for measuring how well large language models retain and recall information embedded in long contexts. KS-Probe injects synthetic "probe facts" into filler documents and measures **Probe Recall Accuracy (PRA)** across five experimental conditions.

---

## Research Questions

| # | Question | Experiment |
|---|---|---|
| RQ1 | Does PRA degrade as context length grows from 10K to 200K tokens? | `exp1_context_fidelity` |
| RQ2 | Does probe position within the context affect recall? (primacy/recency bias) | `exp2_positional_recall` |
| RQ3 | Does PRA degrade as multi-turn conversation length grows? | `exp3_multiturn_degradation` |
| RQ4 | Can we detect silent truncation by placing probes near the context limit? | `exp4_silent_truncation` |
| RQ5 | Do tokenizer differences explain PRA divergence across models? | `exp5_tokenizer_divergence` |

---
## Read our research paper here:
https://kangaroo.solutions/#research 

## Models Tested

| Model | Provider | Context Window |
|---|---|---|
| `claude-opus-4-6` | Anthropic | 200K |
| `claude-sonnet-4-6` | Anthropic | 200K |
| `gpt-4.1` | OpenAI | 128K |
| `gpt-5.2` | OpenAI | 200K |
| `grok-4.1-fast` | xAI | 131K |
| `deepseek-v3.2` | DeepSeek | 65K |

---

## Project Structure

```
KS_Probe/
├── ks_probe/
│   ├── adapters/        # One adapter per LLM provider (OpenAI-compatible interface)
│   ├── corpus/          # Filler text generation and context assembly
│   ├── probes/          # Probe fact bank, injector, and question types
│   ├── scorer/          # Scoring pipeline: exact match, keyword, hallucination, composite PRA
│   ├── orchestrator/    # Async runner, rate limiter, checkpointing, run planner
│   ├── experiments/     # exp1–exp5 experiment classes
│   ├── db/              # SQLite schema, queries, engine (SQLAlchemy)
│   └── core/            # Config loader, cost estimator, seed manager
├── configs/
│   ├── models.yaml      # Model registry with costs and context sizes
│   ├── exp1_context_fidelity.yaml
│   ├── exp2_positional_recall.yaml
│   ├── exp3_multiturn_degradation.yaml
│   ├── exp4_silent_truncation.yaml
│   ├── exp5_tokenizer_divergence.yaml
│   └── mock_test.yaml   # Smoke test — no API keys needed
├── scripts/
│   ├── run_experiment.py    # Main entry point
│   ├── export_results.py    # Export DB results to CSV
│   ├── build_corpus.py      # Generate filler text corpus
│   └── estimate_cost.py     # Estimate API cost before running
├── data/
│   ├── probes/probe_facts.json   # 100 synthetic probe facts
│   └── corpus/                   # Generated filler text (gitignored)
├── outputs/             # Experiment run logs (gitignored)
├── tests/
├── analysis/
├── requirements.txt
└── .env.example         # Copy to .env and fill in your API keys
```

---

## Probe Types

- **explicit_recall** — Direct factual questions ("What is the capital of Zorvath?")
- **grounded_reasoning** — Multi-hop inference over injected facts
- **implicit_context** — Constraint satisfaction requiring background knowledge

---

## Quick Start

### 1. Install dependencies
```bash
pip install openai anthropic tiktoken aiohttp httpx numpy pandas \
            pydantic pyyaml python-dotenv sqlalchemy tqdm rich \
            scipy matplotlib seaborn pytest pytest-asyncio
```

### 2. Set API keys
```bash
cp .env.example .env
# Edit .env and fill in your keys
```

Required keys:
```
OPENAI_API_KEY=sk-...
ANTHROPIC_API_KEY=sk-ant-...
XAI_API_KEY=xai-...
DEEPSEEK_API_KEY=sk-...
```

### 3. Build the filler corpus
```bash
python scripts/build_corpus.py
```

### 4. Smoke test (no API keys needed)
```bash
python scripts/run_experiment.py --config configs/mock_test.yaml --mock-mode
```

### 5. Estimate cost before running
```bash
python scripts/estimate_cost.py --config configs/exp1_context_fidelity.yaml
```

### 6. Run an experiment
```bash
python scripts/run_experiment.py --config configs/exp1_context_fidelity.yaml
```

### 7. Export results to CSV
```bash
python scripts/export_results.py --experiment exp1_context_fidelity --summary
```

---

## Running All Experiments

```bash
python scripts/run_experiment.py --config configs/exp1_context_fidelity.yaml
python scripts/run_experiment.py --config configs/exp2_positional_recall.yaml
python scripts/run_experiment.py --config configs/exp3_multiturn_degradation.yaml
python scripts/run_experiment.py --config configs/exp4_silent_truncation.yaml
python scripts/run_experiment.py --config configs/exp5_tokenizer_divergence.yaml
```

Runs are **checkpointed** — if interrupted, rerunning the same command will skip already-completed runs and resume from where it left off.

---

## Scoring

Each probe question receives a **PRA score** (0.0–1.0) computed as a composite of:

| Component | Description |
|---|---|
| Exact match | Gold answer string found in response |
| Keyword score | Fraction of required keywords present |
| Constraint compliance | Structured constraints satisfied |
| Hallucination penalty | Confident wrong claims penalize score |

---

## Results

Results are stored in `ks_probe_results.db` (SQLite). Export with:

```bash
# Summary table per model/experiment
python scripts/export_results.py --all --summary

# Full run-level data for one experiment
python scripts/export_results.py --experiment exp1_context_fidelity --output results_exp1.csv
```

---

## Adding a New Model

1. Add model entry to `configs/models.yaml`
2. Register model ID → provider in `ks_probe/adapters/registry.py`
3. If new provider: create `ks_probe/adapters/<provider>_adapter.py`
4. Add model to desired experiment YAML configs
5. Add API key to `.env`

---

## License

MIT
