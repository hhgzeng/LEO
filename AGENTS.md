# AGENTS.md

> Project instructions and operating guidelines for AI Coding Agents working in this repository.

## 1. Project Overview

**LEO (Large Language Model-Based Evolutionary Optimizer)** is a zero-shot black-box optimization framework based on the paper:
_Large language model-based evolutionary optimizer: Reasoning with elitism_ (Neurocomputing 622, 2025).

### Core Mechanism

- **Dual-Population Decoupling:** Maintains separate `Explore Pool` ("explore far away regions") and `Exploit Pool` ("exploit close by regions"). Does **not** adjust LLM temperature to balance exploration and exploitation.
- **Elitism Guardrails (`PortFilter`):** In each iteration, copies the top $N_{\text{port}}$ candidates from the explore pool into the exploit pool (overwriting its worst candidates), and retains the top $N_{\text{pop}}$ in each pool.
- **Jitter Perturbation:** Injects minor noise ($\sim 10^{-4}$ to $10^{-3}$) into prompt numbers to avoid token hallucination and autoregressive mode collapse.
- **Persona:** Enforces the role `"You are an optimization researcher tasked to minimize the value of loss."`

---

## 2. Environment & Toolchain (`uv`)

This project strictly uses **`uv`** for dependency and environment management.

### Essential Commands

```bash
# Setup & Sync environment
uv sync

# Add dependencies
uv add <package>
uv add --dev pytest ruff mypy

# Run scripts / modules
uv run python -m leo.cli --help
uv run python -m leo.cli optimize --problem sphere2d --mock

# Run test suite
uv run pytest

# Code quality
uv run ruff check .
uv run ruff format .
```

### Environment Variables (`.env`)

```bash
OPENAI_API_KEY=sk-...                      # LLM API key
OPENAI_BASE_URL=https://api.openai.com/v1 # OpenAI / compatible endpoint (Ollama/vLLM/DeepSeek)
LEO_MODEL_NAME=gpt-3.5-turbo-0613         # Target model
LEO_MOCK_LLM=true                         # Set true for offline unit tests / CI
LEO_CACHE_DIR=.leo_cache                  # Persistent disk cache for LLM queries
```

---

## 3. Project Structure

```text
LEO/
├── AGENTS.md                 # This file (AI instructions)
├── LEO.md                    # Paper transcription & reference equations
├── pyproject.toml            # Project metadata & uv dependencies
├── src/
│   └── leo/
│       ├── core/             # Optimization loop & population management
│       │   ├── algorithm.py  # Main LEO execution loop (Algorithm 1)
│       │   ├── population.py # Population container & PortFilter logic
│       │   └── jitter.py     # Decimal jitter perturbation generator
│       ├── llm/              # LLM client, prompt templates, parser
│       │   ├── client.py     # Async OpenAI client with retry & disk caching
│       │   ├── prompts.py    # Explore, Exploit, Modular prompt builders
│       │   ├── parser.py     # Regex float extractor & CSV table parser
│       │   └── mock.py       # Deterministic mock LLM for offline tests
│       ├── problems/         # Benchmark objective functions
│       │   ├── base.py       # BaseProblem interface
│       │   ├── benchmarks_2d.py # Sphere, Himmelblau, Rosenbrock, Beale, Goldstein-Price
│       │   ├── rosenbrock_nd.py # Shifted N-dim Rosenbrock (a = 0.2913)
│       │   ├── multi_objective.py # ZDT1, ZDT3
│       │   └── engineering/  # Nozzle, heat transfer, windfarm layout
│       ├── modular/          # LEO integration into NSGA-II
│       ├── baselines/        # Baseline optimizers (CMA-ES, L-BFGS-B, LEO-Rnd)
│       └── cli.py            # CLI entry point (Typer)
├── tests/                    # Pytest test suite
└── experiments/              # Paper reproduction scripts & artifact logs
```

---

## 4. Key Algorithmic Steps (Algorithm 1)

1. **Initialization:** Sample $N_{\text{pop}}$ random candidate vectors within variable bounds for both `x_explore` and `x_exploit`. Evaluate objective values `y_init`.
2. **Jitter Injection:** Add small perturbation $x_{\text{jitter}} = x + \epsilon$ before injecting coordinates into prompt context.
3. **LLM Query:**
   - Call LLM with `Explore Prompt` $\to \tilde{x}_{\text{explore}}$ ($N_{\text{pop}}$ new scattered candidates).
   - Call LLM with `Exploit Prompt` $\to \tilde{x}_{\text{exploit}}$ ($N_{\text{pop}}$ new localized candidates).
4. **Evaluation:** Evaluate $f(\tilde{x})$ on both pools. Append to pools ($2 N_{\text{pop}}$ candidates each).
5. **Port & Filter:**
   - Sort explore pool; take top $N_{\text{port}}$ solutions.
   - Sort exploit pool; replace worst $N_{\text{port}}$ solutions with explore top solutions.
6. **Selection:** Sort and truncate both pools back to size $N_{\text{pop}}$.
7. **Termination:** Repeat for `maxIters` (typically 30 for 2D, 100 for high-dim/engineering).

---

## 5. Development & Agent Working Rules

1. **UV Execution Only:** Always run commands via `uv run` (e.g. `uv run pytest`). Never call global `pip` or system `python`.
2. **Offline-First Testing:** Unit tests in `tests/` must default to `LEO_MOCK_LLM=true` so tests run instantly without external network or API keys.
3. **Cost & Rate Limit Protection:** External LLM calls must pass through a local cache (`client.py`) with exponential backoff on HTTP 429.
4. **Strict Response Parsing:**
   - LLMs output floating-point values in CSV format (`%.6f`).
   - Parser must strip code fences (`csv ...`), regex-extract numbers, reshape to $(N_{\text{candidates}}, N_{\text{vars}})$, clamp to domain bounds, and handle retries/fallbacks cleanly.
5. **Statistical Rigor:** When benchmarking optimization performance, evaluate over multiple seeds (at least 10–30) and report median and mean $\pm$ standard deviation.
6. **Code Standards:** Type annotations on all functions, Pydantic for configuration, `ruff` for formatting. Keep modules decoupled and easily testable.
