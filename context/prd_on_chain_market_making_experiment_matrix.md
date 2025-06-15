# Product Requirements Document (PRD)

## 1. Title

**On‑Chain Market‑Making Experiment Matrix System (4090‑Optimised)**

---

## 2. Background & Goal

We need a *single‑node* Python system that can spin up **1 000+ parameterised experiments** to evaluate inventory‑aware market‑making strategies on high‑liquidity crypto assets across Base, Arbitrum, and Solana.\
Key drivers:

- Optimise PnL per **\$1 000** of notional risk.
- Reuse the mature **ETHPredict** ML pipeline (feature engineering, hierarchical models, CV) to reduce dev time.
- Run end‑to‑end in minutes on a **RTX 4090 / Ubuntu 22.04** workstation.

---

## 3. Success Metrics

| Metric                                    | Target                              | Notes                                         |
| ----------------------------------------- | ----------------------------------- | --------------------------------------------- |
| Median wall‑clock runtime for 1000 trials | < **2 h**                           | incl. model training & simulation             |
| GPU utilisation during training           | > **70 %**                          | via `gpu_hist` for XGBoost & RAPIDS pipelines |
| Peak RAM                                  | ≤ **24 GB**                         | allows parallel CPU sims                      |
| Code re‑use of ETHPredict functions       | ≥ **70 % lines** untouched          | measured via `cloc` diff                      |
| CLI runner UX                             | edit **one YAML** + `python run.py` | no code edits for new grids                   |

---

## 4. In‑Scope Functionality

1. **Experiment Orchestrator** – grid/random/Bayesian sweep executor with multiprocessing & GPU semaphore.
2. **Config‑as‑Code** – single YAML schema defining:
   - data sources & date ranges
   - bar construction params
   - feature blocks to toggle
   - model hyper‑params (all levels)
   - GLFT/MM γ, bribe strategy, hedging flags
3. **CLI Runner** – `runner.py` with sub‑commands:
   - `run` – execute YAML once
   - `grid` – explode YAML ranges & run matrix
   - `resume` – continue crashed batch
   - `report` – aggregate CSV → HTML dashboard
4. **Adaptor Layer to ETHPredict** – thin wrappers exposing
   - `load_data()`, `make_features()`, `train_models()`, `walk_forward()`
5. **DEX Execution Simulator** – deterministic back‑test engine supporting:
   - AMM (constant‑product) swaps for Aerodrome, PancakeSwap
   - Hyperliquid LOB matching (price‑time FIFO)
   - Bribe inclusion probability model (logistic) with seedable RNG
6. **Logging & Artefact Store** – write metrics to `results/{exp_id}/` (CSV, parquet, plots) + optional MLflow.

---

## 5. Out‑of‑Scope

- Real‑money execution or private‑key management
- Multi‑node cluster scheduling
- UI dashboards beyond static HTML/PNG

---

## 6. High‑Level Architecture

```text
┌────────────┐   YAML      ┌────────────────┐     GPU       ┌────────────────┐
│ CLI Runner │───────────▶│ Orchestrator   │─────────────▶│ Model Trainer  │
└────────────┘            │  (multiproc)   │               └────────────────┘
        ▲                 │    ▼           │                       ▲
        │        bars     │ Simulator/Exec │ metrics              │
        │                 └────────────────┘───────────results───┘
        │                         ▲
        │      adapt              │
┌───────┴───────┐   DF/np  ┌──────┴───────┐
│ ETHPredict    │<────────▶│ Feature Eng. │
└───────────────┘          └──────────────┘
```

- **Single process per trial**; GPU‑heavy parts guarded by `cuda_lock`.
- Shared memory cache for pre‑computed bars & features.

---

## 7. Detailed Module → Task Breakdown

| #    | Module             | Task                                                   | Complexity | Key Acceptance Criteria                |
| ---- | ------------------ | ------------------------------------------------------ | ---------- | -------------------------------------- |
| 7.1  | **config/**        | Define `schema.yaml` with JSON‑Schema annotations      | **S**      | `yamale` lint passes                   |
| 7.2  | **cli/**           | Build `runner.py` using `typer` (auto‑docs)            | **M**      | `--help` shows commands                |
| 7.3  | **orchestrator/**  | Multiprocessing pool, GPU semaphore                    | **H**      | 1000 trials run with no GPU collisions |
| 7.4  | **data/**          | Port `scripts/data_setup.py` from ETHPredict           | **S**      | Identical output hashes                |
| 7.5  | **bars/**          | Implement Tick/Volume/Dollar bar constructor (Numba)   | **M**      | 10× faster than pandas baseline        |
| 7.6  | **features/**      | Wrap ETHPredict feature funcs; add volatility‑adj flow | **M**      | Feature matrix matches legacy pipeline |
| 7.7  | **models/**        | Adapt ETHPredict `train.py` to accept injected params  | **M**      | GPU training < 2 min on sample set     |
| 7.8  | **market\_maker/** | GLFT quote calculator + inventory book                 | **M**      | Unit tests: skew vs q matches formula  |
| 7.9  | **bribe/**         | Logistic inclusion model + optimiser                   | **S**      | Finds arg‑max within 1 ms              |
| 7.10 | **sim/**           | Back‑test engine (AMM & LOB modes)                     | **H**      | PnL deterministic given seed           |
| 7.11 | **risk/**          | VaR, drawdown, inventory drift monitors                | **S**      | Alerts logged when thresholds crossed  |
| 7.12 | **logging/**       | MLflow / CSV writers, HTML report generator            | **M**      | Generates `index.html` with charts     |
| 7.13 | **tests/**         | PyTest suite + GitHub Actions (CPU)                    | **S**      | >90 % coverage                         |
| 7.14 | **docs/**          | Auto‑generated docs via MkDocs                         | **S**      | `mkdocs serve` runs                    |

Legend: **S** = ≤1 dev‑day, **M** = 1‑3 dev‑days, **H** = >3 dev‑days

---

## 8. CLI Runner Specification

### 8.1 File Layout

```text
repo/
├── run.py          # entrypoint
├── configs/
│   └── base.yml    # user‑editable template
└── src/…           # modules above
```

### 8.2 Command Surface

```
python run.py run    --config configs/base.yml --tag trial‑001
python run.py grid   --config configs/grid.yml --max-par 8
python run.py report --input results/ --out summary.html
```

### 8.3 YAML Schema (excerpt)

```yaml
experiment:
  id: exp_{{timestamp}}
  seed: 42
  trials: 1000

bars:
  type: dollar
  threshold_usd: [50000, 100000]   # range expands in grid mode

features:
  frac_diff_order: 0.5
  include:
    - vol_adj_flow
    - rsi

model:
  level0:
    algo: xgboost
    params:
      max_depth: 6
      eta: 0.05
      tree_method: gpu_hist
  meta_labeling: true

market_maker:
  gamma: [0.1, 0.5, 1.0]
  hedge: true

bribe:
  mode: percentile
  percentile: 95
```

---

## 9. ETHPredict Re‑use Map

| ETHPredict Module       | New Layer               | Adaptation Notes                                    |
| ----------------------- | ----------------------- | --------------------------------------------------- |
| `preprocess.py`         | **bars/data**           | Replace time bars with injected bar DF              |
| `label.py`              | **features/labels**     | Keep triple‑barrier unchanged                       |
| `model.py`              | **models/**             | Import PriceLSTM, MetaMLP, ConfidenceGRU as drop‑in |
| `train.py`              | **orchestrator/models** | Expose `train_models(cfg, X, y)`                    |
| `ensemble.py`           | **models/ensemble**     | Optional; flag in YAML                              |
| `scripts/data_setup.py` | **data/**               | Retain for data download                            |

---

## 10. Non‑Functional Requirements

- **Performance** – End‑to‑end ≥5× faster than pure‑CPU baseline; leverage cuDF + XGBoost GPU.
- **Reproducibility** – Each trial runs with iso‑timestamped seed; outputs hash‑named.
- **Portability** – Works on Ubuntu 22.04; CUDA 12; Python 3.11; install via `pip -e .`.
- **Observability** – Progress bar per trial; global ETA; error logs per failed trial.
- **CI/CD** – Lint (ruff), type‑check (mypy), unit tests on every PR.

---

## 11. Milestones (Optimistic Single‑Dev Timeline)

| Week | Deliverable                                                        |
| ---- | ------------------------------------------------------------------ |
| 1    | Repo skeleton, YAML schema, runner `run.py`, data re‑use adapter   |
| 2    | Bar constructor, feature layer port, GPU model trainer hooked      |
| 3    | Simulator v1 (AMM), GLFT module, bribe optimiser                   |
| 4    | LOB sim, risk monitors, HTML report, CI pipeline                   |
| 5    | Performance tuning on 4090, docs, internal dry‑run of 1 000 trials |
| 6    | Buffer: bug fixes, polish, handoff                                 |

---

## 12. Risks & Mitigations

- **GPU contention** – use semaphore + queueing; consider partitioning model training vs sims.
- **DEX APIs breaking** – cache raw data snapshots; fail‑soft & mark trial invalid.
- **Overfitting to history** – enforce walk‑forward CV; out‑of‑sample test window locked.
- **Sim realism gap** – validate against small real‑money trades before full deploy.

---

## 13. Appendices

- Appendix A – GLFT formulae reference
- Appendix B – Inclusion probability logistic model details
- Appendix C – YAML full schema JSON‑Schema draft

---

**Document Owner:** Pranay Kundu\
**Last Updated:** 15 Jun 2025

