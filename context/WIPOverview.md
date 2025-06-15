# Work in Progress Overview

## Implemented Functionality (from ETHPredict)

### Core ML Pipeline
- Multi-source data integration (Binance, DeFiLlama, Santiment)
- Advanced feature engineering (fractional differentiation, entropy, structural breaks, volatility regimes, etc.)
- Hierarchical model architecture (PriceLSTM, MetaMLP, ConfidenceGRU)
- Triple-barrier labeling and meta-labeling
- Purged time-series cross-validation with embargo
- Sample uniqueness weighting
- Baseline and ensemble models
- Comprehensive validation and deliverable generation (datasource matrix, signal design, prototype script)
- Visualization and analysis scripts (data overview, feature correlation, label analysis, model performance, etc.)
- Automated data collection and validation scripts
- Modular pipeline with `preprocess.py`, `label.py`, `model.py`, `train.py`, `ensemble.py`

### CLI and Runner (Basic)
- Basic CLI for pipeline execution (setup.py, results/prototype.py)
- Typer CLI stub for experiment runner (run.py) with placeholder commands
- Configurable via YAML for basic pipeline runs

### Results and Reporting
- Generation of core deliverables and visualizations in `results/`
- Logging and output of validation/test results

---

## Not Yet Implemented (from PRD)

### 1. Experiment Orchestrator
- Grid/random/Bayesian sweep executor for large-scale experiment matrix
- Multiprocessing and GPU semaphore for parallel trial execution
- Resume/continue crashed batch functionality
- Progress tracking and error recovery for 1000+ trials

### 2. Config-as-Code System
- Unified YAML schema for all experiment parameters (data, bars, features, models, MM params, bribe, hedging, etc.)
- Full support for grid expansion and parameter sweeps
- JSON-schema validation and linting

### 3. Advanced CLI Runner
- `run.py` with full sub-commands:
  - `run` (single experiment)
  - `grid` (parameter sweep)
  - `resume` (continue batch)
  - `report` (aggregate results to HTML dashboard)
- No-code UX for new experiment grids (edit YAML only)

### 4. DEX Execution Simulator
- Deterministic back-test engine for:
  - AMM (constant-product) swaps (Aerodrome, PancakeSwap)
  - Hyperliquid LOB matching (price-time FIFO)
  - Bribe inclusion probability model (logistic, seedable RNG)
- Inventory-aware market making logic

### 5. Market Making Components
- GLFT quote calculator and inventory book
- Bribe optimization and hedging strategy modules
- Integration with ETHPredict feature/model pipeline

### 6. Risk Management
- VaR, drawdown, and inventory drift monitors
- Alerting/logging for risk threshold breaches

### 7. Logging & Artifact Store
- Structured logging for all experiment runs
- Results storage in `results/{exp_id}/` (CSV, parquet, plots)
- Optional MLflow integration for experiment tracking

### 8. Performance Optimizations
- GPU utilization optimization for >70% usage
- Memory management for ≤24GB peak RAM
- Shared memory cache for pre-computed bars/features
- Parallel CPU simulation support

### 9. Documentation & CI/CD
- Auto-generated docs via MkDocs
- API documentation and usage tutorials
- Performance benchmarks and reproducibility notes
- Linting (ruff), type checking (mypy), and unit tests on every PR
- >90% test coverage

---

## Implementation Priority (PRD Timeline)

- **Week 1:** Repo skeleton, YAML schema, runner, data re-use adapter
- **Week 2:** Bar constructor, feature layer port, GPU model trainer
- **Week 3:** Simulator v1 (AMM), GLFT, bribe optimiser
- **Week 4:** LOB sim, risk monitors, HTML report, CI pipeline
- **Week 5:** Performance tuning, docs, dry-run of 1,000 trials
- **Week 6:** Bug fixes, polish, handoff

---

## Summary

The current repo implements the ETHPredict ML pipeline and core data/model/feature engineering, but the market-making experiment matrix system (orchestrator, simulator, advanced CLI, MM logic, risk, logging, and full config-as-code) is not yet implemented. The next steps should focus on building the orchestrator, simulator, and advanced experiment management as outlined in the PRD.
