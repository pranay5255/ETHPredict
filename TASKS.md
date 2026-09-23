# ETHPredict Lighter-Only Task List

## Product Goal and Source of Truth

ETHPredict is a local decision-support application for a discretionary Lighter ETH perpetual trader. It should show refreshed public market data, aggregated feature values, model outputs with evidence for interpreting them, checkpoint comparisons, and simulated backtest results. The trader places any real order separately in Lighter. The v2 artifact-backed pipeline in `configs/config.yml` is the primary path for future research and manually frozen models; `lighter_compare` and legacy staged GLFT runs remain exploratory history.

GitHub issue state is authoritative for the linked backlog below. This file was reconciled with GitHub on 2026-09-23; a closed experiment issue does not close its parent implementation issue.

## Local Decision-Support Delivery

- [ ] [#43 Collect refreshed Lighter ETH feature snapshots with provenance](https://github.com/pranay5255/ETHPredict/issues/43): refresh after each completed 5-minute candle, retain immutable data/feature snapshots, and show provenance, coverage, and stale status. Public read-only data only.
- [ ] [#44 Freeze reviewed v2 checkpoints and emit read-only current model outputs](https://github.com/pranay5255/ETHPredict/issues/44): keep retraining separate from 5-minute refresh; explicitly promote a complete model and preprocessing package; show predicted returns and direction probabilities by horizon with as-of times and calibration/error context. No automatic promotion or order generation.
- [ ] [#45 Build a local read-only Lighter research and current-state dashboard](https://github.com/pranay5255/ETHPredict/issues/45): show current feature/model outputs, a checkpoint performance summary linked to Trackio, run-level data/features/models/backtest stages with duration/resource/failure metrics, and AFML phase progress. Keep validation/test and simulated/real activity distinct; display unavailable evidence clearly.
- [ ] Complete [#23 Trackio research accounting](https://github.com/pranay5255/ETHPredict/issues/23) and [#24 artifact reproducibility](https://github.com/pranay5255/ETHPredict/issues/24) as shared prerequisites. Local Trackio delivery/readback and per-checkpoint model metrics are being hardened; the full issue acceptance criteria remain open.

Feature importance from [#17](https://github.com/pranay5255/ETHPredict/issues/17) and richer backtest statistics from [#21](https://github.com/pranay5255/ETHPredict/issues/21) enrich the dashboard when implemented. Their absence must be shown instead of inferred.

## Completed in Lighter-Only Refactor

- [x] Set `configs/config.yml` to `data.sources: [lighter]`.
- [x] Configure Lighter ETH perp mainnet market `market_id: 0` with `1h` resolution.
- [x] Collect raw Lighter OHLCV for `2026-03-28` through `2026-06-28`.
- [x] Validate raw Lighter OHLCV row count, timestamp ordering, uniqueness, and OHLC numeric quality.
- [x] Archive legacy Binance, DeFiLlama, and Santiment CSV inputs under `archive/legacy_data_sources/data`.
- [x] Archive legacy multi-source feature/config/docs/code snapshots under `archive/legacy_data_sources`.
- [x] Remove active bribe optimization, DEX simulation, and parameter optimization config sections.
- [x] Refactor `DataPreprocessor` to load only `data/raw/*-lighter-*.csv` and build OHLCV-only features.
- [x] Add tests proving legacy-looking raw files are ignored.
- [x] Verify focused tests and Lighter feature-build smoke command.

## Completed in UV GPU Experiment Setup

- [x] Import the current `requirements.txt` dependency set into the uv project metadata.
- [x] Keep `requirements.txt` unchanged for legacy setup while treating `pyproject.toml` and `uv.lock` as the active experiment dependency source.
- [x] Configure `torch>=2.11.0` to use the official PyTorch CUDA 12.8 wheel index on Linux/Windows.
- [x] Refresh `uv.lock` so CUDA 13 artifacts are replaced with CUDA 12.x packages for the 4090 path.
- [x] Run `uv sync` for the active environment.
- [x] Verify uv-managed Torch reports `2.11.0+cu128`, CUDA available, and `NVIDIA GeForce RTX 4090` on `cuda:0`.
- [x] Add shared CUDA device resolution that defaults neural training to `cuda:0` and requires `--allow-cpu` for CPU neural runs.
- [x] Add `src.experiments.lighter_compare` for Lighter-only neural trials, CPU ARIMA/SARIMAX baselines, and GLFT metric ranking.
- [x] Add `configs/lighter_experiments.yml` for triple-barrier and next-hour-return experiment targets.
- [x] Add small full-data staged configs for 5m `next_hour_return` and `triple_barrier` experiments.
- [x] Run exploratory small staged 5m experiments and confirm serial full-data execution succeeds after parallel CUDA OOM fallback.
- [x] Add CUDA environment, one-batch neural stack, CPU baseline, and end-to-end smoke tests.
- [x] Verify `uv run pytest`, `uv lock --check`, and the Lighter compare smoke command.

## Active AFML Review GitHub Issue Plan

Source critique: `context/afml_pipeline_review.md`.
GitHub label set: `afml-review`, `priority:*`, and `area:*`.

These issues turn the AFML critique into the active research-hardening roadmap. Work
through them in dependency order. Do not use final test PnL as the discovery loop;
select features, thresholds, policies, and trial winners on purged validation evidence
before any final test evaluation.

### Phase 0: Make Runs Auditable Before Expanding Research

- [ ] [#24 Harden artifacts, manifests, and test-set reuse guards](https://github.com/pranay5255/ETHPredict/issues/24)
  - Tackle first because later issues need immutable run identity.
  - Add raw data hashes, resolved config hash, git commit, dirty state, critical untracked files, dependency metadata, device metadata, split manifest hashes, and trial counts to v2 run manifests.
  - Add a final-test reuse guard and counter so selected test evaluations are visible and cannot be silently repeated under the same research spec.
  - Add a frozen-rerun comparison command that checks manifests, split boundaries, trial counts, and metric variance.
  - Done when a changed raw file or changed config is detected by manifest comparison.

- [ ] [#23 Make Trackio experiment accounting mandatory for research runs](https://github.com/pranay5255/ETHPredict/issues/23)
  - Make Trackio mandatory for non-smoke runs, while keeping an explicit local-debug override.
  - Log all trial statuses: completed, failed, skipped, no-trade, low-trade, raw best, trade-qualified best, classification best, and calibration best.
  - Ensure local artifacts contain the same critical accounting fields as Trackio.
  - Build dashboard-ready failure-mode metrics: no-trade winners, poor calibration, high turnover, unstable ranking, high cost sensitivity, and concentrated PnL.
  - Done when every non-smoke run can answer how many trials were attempted and why the selected trial won.

### Phase 1: Fix Upstream Data Clock and Feature Validity

- [ ] [#8 Add activity-driven bars and validated side-data joins](https://github.com/pranay5255/ETHPredict/issues/8)
  - Keep current 5-minute time bars as the baseline.
  - Add volume-bar and dollar-bar construction where Lighter source coverage is adequate.
  - Add side-data joins for funding, mark basis, open interest, exchange volume, order-book, and trade-flow features only behind coverage and leakage gates.
  - Emit bar-clock diagnostics before model training: missingness, serial correlation, volatility distribution, label balance, and source coverage.
  - Done when a frozen downstream config can compare time bars against at least one validated activity-driven clock or fail with a clear coverage report.

- [ ] [#9 Fix feature reproducibility and fold-local preprocessing](https://github.com/pranay5255/ETHPredict/issues/9)
  - Make `features.include` control the active v2 feature matrix.
  - Respect configured fractional differentiation order and add fixed-width fracdiff as the reproducible AFML-oriented mode.
  - Move scaling into fold-local preprocessing so train windows fit scalers and validation/test windows only transform.
  - Add feature-family manifests and leakage tests.
  - Done when feature ablations are real, reproducible, and not using future distributional information.

- [ ] [#10 Add event-based sampling and candidate-time separation](https://github.com/pranay5255/ETHPredict/issues/10)
  - Separate ML observation sampling from later trade-candidate filtering.
  - Add dense, CUSUM, volatility-scaled CUSUM, and edge-triggered event modes.
  - Persist event metadata and report event rate, average uniqueness, effective sample size, and class balance.
  - Done when dense rolling samples and event-triggered samples can run under the same downstream labels, models, and policy.

### Phase 2: Build Label-Span Metadata, Weights, and Real Purging

- [x] [#14 Implement label-span uniqueness weights and effective sample diagnostics](https://github.com/pranay5255/ETHPredict/issues/14)
  - Add start time, horizon end, realized `t1`, and span indices for base observations and meta-label candidates.
  - Compute concurrency, average uniqueness, and effective sample size.
  - Add weighting modes: uniform, horizon-span uniqueness, triple-barrier `t1` uniqueness, return magnitude, and combined uniqueness-by-return.
  - Wire weights into base training, meta-label training, and relevant metrics.
  - Done when every labeled row has enough span metadata to prove how much it overlaps other rows.

- [ ] [#15 Purge and embargo by exact label spans, then add CPCV](https://github.com/pranay5255/ETHPredict/issues/15)
  - Replace fixed purge gaps with exact label-span overlap checks.
  - Keep the existing fixed-gap walk-forward mode as a baseline.
  - Add embargo modes based on max horizon, sequence length plus horizon, realized `t1`, and volatility-adaptive gaps.
  - Add purged k-fold plus CPCV or grouped scenario CV, then report validation-to-test rank stability.
  - Done when split manifests prove that train rows do not overlap validation or test label spans.

- [ ] [#13 Audit triple-barrier geometry and meta-label definitions](https://github.com/pranay5255/ETHPredict/issues/13)
  - Make profit/stop kappa, symmetric versus asymmetric barriers, vertical success mode, and ambiguous OHLC ordering configurable.
  - Persist label diagnostics for each candidate: side, entry, volatility, barriers, first touch, `t1`, realized return, net return, and ambiguity flag.
  - Compare positive-net-at-vertical success against profit-take-only success.
  - Report class balance, precision, recall, F1, calibration, trade coverage, and PnL.
  - Done when every meta label can be reconstructed from artifact metadata.

### Phase 3: Strengthen Forecasting, Training, and Meta Models

- [ ] [#11 Expand base forecast calibration and edge-ranking diagnostics](https://github.com/pranay5255/ETHPredict/issues/11)
  - Extend forecast reports beyond MAE/MSE/direction into calibration, rank correlation, and edge-bucket realized performance.
  - Add forecast-to-candidate mapping variants: raw sign, probability sign, expected edge after costs, volatility-scaled edge, and horizon agreement.
  - Keep zero-return and momentum baselines prominent.
  - Done when forecast quality can be judged separately from trading PnL.

- [ ] [#12 Add early stopping, uniqueness weighting, and ensemble diagnostics](https://github.com/pranay5255/ETHPredict/issues/12)
  - Add purged-validation early stopping.
  - Add seed ensembles with prediction dispersion and per-seed metrics.
  - Integrate uniqueness weights from #14.
  - Add simple non-neural baselines so LSTM complexity is justified by evidence.
  - Done when model instability is visible before using a larger neural setup.

- [ ] [#18 Calibrate and benchmark meta-labeler models](https://github.com/pranay5255/ETHPredict/issues/18)
  - Add constant, logistic regression, calibrated tree ensemble, and MLP meta-labeler options.
  - Add validation-only probability calibration and threshold sweep tables.
  - Report precision, recall, F1, trade count, coverage, validation net PnL, and calibration error by threshold.
  - Done when the selected meta model and threshold are chosen on validation evidence and are reproducible from artifacts.

- [ ] [#22 Deepen forecast benchmark diagnostics and TimesFM parity](https://github.com/pranay5255/ETHPredict/issues/22)
  - Route benchmark models through identical split, cost, candidate, meta-label, and backtest logic when trading comparison is enabled.
  - Add TimesFM context, scaling, horizon mapping, skip reason, and failure metadata.
  - Compare forecast-only ranking against trading ranking.
  - Done when zero, momentum, LSTM, TimesFM, and future benchmark models share one auditable schema.

### Phase 4: Stop Using Backtests as Feature and Trial Discovery

- [ ] [#17 Add purged feature-importance and ablation reports](https://github.com/pranay5255/ETHPredict/issues/17)
  - Add purged permutation importance for meta-labeler features using F1 and negative log loss.
  - Add single-feature importance, grouped feature-family ablations, and random-noise controls.
  - Report base forecaster and meta-labeler importance separately.
  - Done when feature decisions can be made from purged validation importance and ablations before final test PnL.

- [ ] [#16 Make hyperparameter search trial-aware and meta-label scored](https://github.com/pranay5255/ETHPredict/issues/16)
  - Add deterministic grid, shuffled grid, random search, and log-uniform search.
  - Add selectors for F1, negative log loss, calibration, edge-rank correlation, validation net PnL, and composite minimum-trade metrics.
  - Record full search space, executed subset, trial count, skipped runs, failed runs, no-trade runs, and selection role.
  - Done when trial selection can be audited without looking at final test PnL.

### Phase 5: Upgrade Policy, Execution Assumptions, and Backtest Evidence

- [ ] [#19 Add probability and concurrency-aware bet sizing](https://github.com/pranay5255/ETHPredict/issues/19)
  - Add fixed, probability-scaled, edge-scaled, probability-times-edge, and concurrency-budgeted sizing.
  - Track active bet concurrency across holding periods and horizons.
  - Report turnover, exposure, PnL per turnover, return on execution costs, drawdown, and concentration.
  - Done when fixed notional is only the baseline, not the final sizing assumption.

- [ ] [#20 Upgrade dynamic cost and execution modeling](https://github.com/pranay5255/ETHPredict/issues/20)
  - Add static, stressed, dynamic funding, dynamic spread, and dynamic slippage cost models.
  - Align historical funding to candidate holding periods where coverage is valid.
  - Add execution assumptions: taker, maker with fill haircut, hybrid, delayed entry, missed fills, and slippage stress.
  - Done when candidate filtering, labels, and PnL all consume the same explicit cost model.

- [ ] [#21 Add backtest statistics, PBO, PSR, DSR, and path dispersion](https://github.com/pranay5255/ETHPredict/issues/21)
  - Add holding period, bet frequency, independent-bet proxy, HHI concentration, time under water, PnL per turnover, return on execution costs, and drawdown duration.
  - Add PSR and DSR with explicit assumptions and trial-count inputs.
  - Add PBO once CPCV or grouped scenario paths exist.
  - Done when the backtest report distinguishes debugging evidence from alpha-claim-grade evidence.

### AFML Backlog Verification Commands

```bash
gh api --method GET repos/pranay5255/ETHPredict/issues -f state=open -f labels=afml-review -f per_page=100 --paginate --jq ".[] | \"#\(.number) \(.title)\""
uv run pytest tests/test_config.py tests/test_lighter_preprocessor.py tests/test_lighter_5m_labels.py tests/test_forecast_benchmark.py tests/test_staged_trial.py tests/test_trackio_logging.py
python runner.py configs/config.yml
```

## AFML Experiment Follow-Up Issues

Run these after their parent implementation issues are complete and after the required
backtest/accounting prerequisites listed in each experiment issue body exist. The goal
is to freeze everything except the implemented feature, run a controlled backtest, and
keep or reject the feature based on validation-first evidence and AFML backtest stats.

### Phase 0 Experiment Gate

- [x] After #24, run [#25 Validate manifest reproducibility and test-set guard value](https://github.com/pranay5255/ETHPredict/issues/25).
  - Proves frozen reruns, data-hash checks, config-hash checks, git dirty-state capture, and final-test reuse counters work before trusting later experiments.
  - Evidence: `context/afml_freeze_regen_20260802_report.md` and local artifacts under `artifacts/afml_freeze_regen_20260802`; identical reruns passed, the changed-config control failed config equality while raw data and split hashes stayed fixed, and the final-test reuse counter reached 2 for the frozen baseline.
- [x] After #23, run [#26 Audit Trackio trial accounting and failure-mode panels](https://github.com/pranay5255/ETHPredict/issues/26).
  - Proves every attempted, skipped, failed, no-trade, low-trade, and selected trial is visible in Trackio and local artifacts.
  - Evidence: `context/afml_freeze_regen_20260802_report.md` and `artifacts/afml_freeze_regen_20260802/26_accounting_matrix_20260802_20260802T145622Z/trial_accounting.json`; counts covered 3 completed, 1 failed, 1 skipped, 1 no-trade, 2 low-trade, and 1 trade-qualified trial.

### Phase 1 Data and Feature Experiments

- [x] After #8, run [#27 Compare time, volume, dollar bars, and side-data joins](https://github.com/pranay5255/ETHPredict/issues/27).
  - Freezes downstream settings and varies only bar clock or validated side-data groups.
  - Evidence: `context/afml_freeze_regen_20260802_report.md` and local run artifacts under `artifacts/afml_freeze_regen_20260802`; time, volume, and dollar bar clocks ran on GPU, while side-data joins remained disabled pending validated historical coverage.
- [x] After #9, run [#28 Measure feature-family, fracdiff, and scaler value](https://github.com/pranay5255/ETHPredict/issues/28).
  - Freezes model, labels, split, costs, and policy while varying feature families, fracdiff mode, and scaler discipline.
  - Evidence: `context/afml_freeze_regen_20260802_report.md` and local run artifacts under `artifacts/afml_freeze_regen_20260802`; OHLCV-only, full-no-fracdiff, and fixed-width-fracdiff variants ran with fold-local scaler fitting.
- [x] After #10, run [#29 Compare dense, CUSUM, volatility-CUSUM, and edge events](https://github.com/pranay5255/ETHPredict/issues/29).
  - Freezes labels and policy while testing whether event sampling improves uniqueness, precision, turnover, and net PnL per trade.
  - Evidence: `context/afml_freeze_regen_20260802_report.md` and local run artifacts under `artifacts/afml_freeze_regen_20260802`; dense, CUSUM, volatility-CUSUM, and edge-triggered variants ran on GPU and showed higher uniqueness for event sampling, but the frozen 0.55 meta-threshold policy still produced no trades.

### Phase 2 Label and Validation Experiments

- [ ] After #14, run [#30 Compare uniform, uniqueness, return, and combined weights](https://github.com/pranay5255/ETHPredict/issues/30).
  - Tests whether uniqueness and return-aware weights improve validation classification quality and OOS trading evidence.
- [ ] After #15, run [#31 Compare fixed-gap, exact-span, purged k-fold, and CPCV validation](https://github.com/pranay5255/ETHPredict/issues/31).
  - Tests whether exact-span purging and CPCV change trial ranking, path dispersion, or alpha credibility.
- [ ] After #13, run [#32 Compare triple-barrier geometry and success definitions](https://github.com/pranay5255/ETHPredict/issues/32).
  - Re-labels a frozen candidate set to compare barrier geometry, vertical success rules, and ambiguous OHLC handling.

### Phase 3 Modeling Experiments

- [ ] After #11, run [#33 Compare forecast calibration and candidate mapping value](https://github.com/pranay5255/ETHPredict/issues/33).
  - Tests whether forecast calibration and edge-rank mappings improve candidate quality after costs.
- [ ] After #12, run [#34 Compare early stopping, seed ensembles, weights, and baselines](https://github.com/pranay5255/ETHPredict/issues/34).
  - Tests whether training discipline and simpler baselines beat the current single LSTM under the same split and policy.
- [ ] After #18, run [#35 Compare meta-labeler models, calibration, and thresholds](https://github.com/pranay5255/ETHPredict/issues/35).
  - Tests constant, logistic, calibrated tree, and MLP meta-labelers with validation-only threshold selection.
- [ ] After #22, run [#36 Route benchmark forecasters through identical trading logic](https://github.com/pranay5255/ETHPredict/issues/36).
  - Tests zero, momentum, LSTM, TimesFM, and future benchmark forecasters under one trading schema.

### Phase 4 Research-Discipline Experiments

- [ ] After #17, run [#37 Validate feature importance with purged ablations and noise controls](https://github.com/pranay5255/ETHPredict/issues/37).
  - Confirms feature-importance output predicts ablation value and does not rank random noise as useful.
- [ ] After #16, run [#38 Compare search modes and selection metrics under fixed trial budgets](https://github.com/pranay5255/ETHPredict/issues/38).
  - Tests deterministic grid, shuffled grid, random search, log-uniform search, and classification-first selection metrics under equal budgets.

### Phase 5 Execution and Backtest Evidence Experiments

- [ ] After #19, run [#39 Compare fixed, probability, edge, and concurrency sizing](https://github.com/pranay5255/ETHPredict/issues/39).
  - Tests whether variable sizing improves capital efficiency without amplifying poor calibration.
- [ ] After #20, run [#40 Stress static, dynamic, maker, taker, and delayed execution costs](https://github.com/pranay5255/ETHPredict/issues/40).
  - Tests whether strategy value survives dynamic funding, spread, slippage, maker/taker, delayed-entry, and missed-fill assumptions.
- [ ] After #21, run [#41 Evaluate alpha claims with richer stats, DSR, PBO, and path dispersion](https://github.com/pranay5255/ETHPredict/issues/41).
  - Uses the richer backtest module to decide whether an apparently profitable strategy is debugging-only, validation-promising, or alpha-claim-grade.

### Experiment Backlog Verification Command

```bash
gh api --method GET repos/pranay5255/ETHPredict/issues -f state=open -f labels=afml-experiment -f per_page=100 --paginate --jq ".[] | \"#\(.number) \(.title)\""
```

## Active Near-Term Tasks

- [ ] Finish [#23](https://github.com/pranay5255/ETHPredict/issues/23) and [#24](https://github.com/pranay5255/ETHPredict/issues/24), then use their local artifacts as the dashboard's source of truth.
- [ ] Deliver [#43](https://github.com/pranay5255/ETHPredict/issues/43), [#44](https://github.com/pranay5255/ETHPredict/issues/44), and [#45](https://github.com/pranay5255/ETHPredict/issues/45) in that order, while continuing the AFML issue dependencies above.
- [ ] Complete feature validity [#8](https://github.com/pranay5255/ETHPredict/issues/8) and [#9](https://github.com/pranay5255/ETHPredict/issues/9) before displaying side-data or feature-family claims as validated.
- [ ] Complete forecast diagnostics [#11](https://github.com/pranay5255/ETHPredict/issues/11), feature importance [#17](https://github.com/pranay5255/ETHPredict/issues/17), and backtest evidence [#21](https://github.com/pranay5255/ETHPredict/issues/21) before interpreting a model as reliable for trading decisions.
- [ ] Verify the legacy `python runner.py configs/config.yml` path separately; the v2 research runner is the primary path.
- [ ] Decide whether to migrate legacy `requirements.txt` users fully to uv in a later cleanup.

## Deferred / Archived for Later

- [ ] Reintroduce Binance OHLCV only if a mixed-source experiment is explicitly reopened.
- [ ] Reintroduce DeFiLlama/Santiment joins only after defining target/feature semantics for non-market data.
- [ ] Reintroduce DEX simulation only after the price-data backtest path is stable.
- [ ] Treat GLFT parameter search as deferred alpha evaluation; use it only for passive execution research after directional alpha validation.
- [ ] Reintroduce bribe/MEV optimization only as a separate execution research track.
- [ ] Reintroduce parameter sweeps only after the base Lighter-only run is reproducible.
- [ ] Reconsider signed Lighter trading or live execution only as a separate future product decision; the decision-support dashboard does not submit orders.

## Useful Commands

```bash
python scripts/lighter_collect_data.py --config configs/config.yml
python -m src.data.features_all full_features --data-dir data --sequence-length 24 --out-dir /tmp/ethpredict-lighter-features
uv sync
uv run python -c "import torch; print(torch.__version__); print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0))"
uv run pytest
uv run python -m src.experiments.lighter_compare --config configs/lighter_experiments.yml --smoke
uv run python -m src.experiments.lighter_compare --config configs/lighter_experiments.yml
uv run python -m src.experiments.staged_trial --config configs/staged_trial_small_next_hour.yml --run-name staged_5m_small_next_hour
uv run python -m src.experiments.staged_trial --config configs/staged_trial_small_triple_barrier.yml --run-name staged_5m_small_triple_barrier
python runner.py configs/config.yml
```
