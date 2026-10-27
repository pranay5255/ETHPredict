# #12 / #11 / #18 implementation and verification

Implemented after PR #49, based on `main` at `858f2f3`, and merged in PR #50.
It does not execute performance experiments #33–#35 or promote a model.

## Training and validation contract

`training.early_stopping` controls an internal chronological holdout. Complete
forecast-anchor groups stay together; exact inclusive label-span purging and the
configured source-clock embargo protect the holdout. Side-data coverage is
rechecked for the retained partitions. Scalers and sample weights fit only the
retained training rows. Base LSTM and meta MLP restore the best validation-loss
state and record selected epoch, losses, fit/holdout IDs, and span audits.

Tiny partitions explicitly record `unavailable` and use the predeclared fixed
epoch budget. Audits distinguish attempted holdouts from the actual fit rows.
Tabular fits report stopping as not applicable. No outer validation or final-test
labels participate in stopping.

`training.ensemble_seeds` selects reproducible members. Return and direction
probabilities are averaged independently; logits are not averaged. Saved OOF
frames include member values and standard deviations. Fold and main trial
summaries include per-seed forecast metrics, metric variance, and mean prediction
variance. Checkpoints retain member scaler buffers. CUDA uses deterministic
cuDNN settings; byte-for-byte reproducibility across hardware/library versions is
not promised. Existing validation-to-test rank stability reports only actually
evaluated trials and is unavailable when too few results exist.

`training.bootstrap` supports `none`, `bootstrap`, and `sequential_bootstrap`.
Sequential draws use marginal span uniqueness on retained training events with
replacement. Uniform and the existing uniqueness/return/combined weight modes
remain configurable and audited. Each sequential draw is one vectorized pass
over events (prefix sums over span segments), so a 16,000-event fold takes about
3 seconds; a test pins the draws to a brute-force indicator-matrix reference.
Bootstrap draws are recorded only when resampling is enabled.

`model.base.type` supports `lstm`, `linear`, and `tree`. Tabular heads use the
latest selected feature vector plus rolling mean/std; regression and binary
direction heads share the split and prediction schema. Existing dotted search
paths control hidden size, layers, dropout, learning rate, weight decay, and
epochs. Randomized search/budget improvements remain #16.

## Forecast and candidate contract

`candidate_mapping.mode` selects `raw_sign`, `probability_sign`, `expected_edge`,
`volatility_scaled_edge`, or `horizon_agreement` from the same saved forecasts.
The default expected-edge policy preserves the prior cost filter. Raw/probability
sign expose directional proposals; the downstream alpha policy still applies its
own edge threshold. Probability-sign expected edge uses the chosen side, so a
probability/return disagreement can produce negative expected edge. Volatility
scaling divides cost-adjusted edge by causal per-bar realized volatility in bps.

Forecast artifacts include Brier/log-loss/ECE and reliability buckets, descriptive
return calibration, signed-return rank correlation, expected-edge versus realized
net-edge rank correlation, and gross/net edge buckets with coverage. These are
fixed-horizon forecast diagnostics; they do not substitute for path-dependent
trade results. Constant/insufficient ranks are explicitly unavailable. Zero-return
has zero realized exposure/cost; momentum remains an explicit comparison.

`forecast_diagnostics.json` is saved separately from alpha diagnostics and PnL.
Forecast and backtest namespaces stay separate in Trackio (see
`context/trackio_dashboard.md`). Trackio run configs replace row-level ID lists
(`*_indices`, `*_keys`, bootstrap `draws`) with counts; local manifests keep the
full audits. Benchmark summaries
always include zero return, even if only other models were requested. Linear and
tree models can be selected alongside LSTM and momentum. TimesFM success records
include checkpoint/package/backend, context, input transform, effective compile
scaling settings when available, horizon offsets, and probability calibration
behavior. Disabled/failed runs retain configuration and reasons.

With `research.final_test_guard.evaluate: false`, a benchmark must also use
`benchmark.route_to_meta_backtest: false`; it then evaluates validation forecasts
only and does not create final-test ledger entries. Broader downstream benchmark
parity and forecast-versus-trading rank reports remain #22.

## Meta-model and threshold contract

`model.meta_labeler.type`: `constant`, `logistic`, `tree`, or `mlp`.
`calibration.method`: `none`, `platt`, or `isotonic`. Constant fallbacks record
`empty_labeled_candidates`, `single_class_training_labels`, or a configured
constant reason. Logistic coefficients are in standardized feature units; tree
impurity importance is descriptive and does not replace #17's purged importance.

Optional `model.meta_labeler.selection` reserves a purged tail inside meta
training and compares predeclared `models` by Brier score or log loss. Optional
`alpha_backtest.meta_threshold_selection` uses that same selection population
and a predeclared threshold grid, maximizing F1 or simulated validation net PnL
subject to `min_trades`. Ties choose the higher threshold. If no selection is
available, the configured type/fixed threshold is retained with an explicit
reason. Class balance before and after candidate/label filtering is recorded.

When calibration is enabled, a **different earlier purged holdout** fits the
calibrator. MLP stopping uses a further holdout inside the remaining fit rows.
The direct fitting API accepts `dataset` to supply the original source clock;
timestamp embargo fails closed without that clock. Calibration rows do not fit
the model or its scaler. Final trained meta models now use chronological nested
base OOF forecasts from development, avoiding calibration on nonchronological
outer OOF predictions whose base models may have seen calibration labels.

Outer held-out validation candidates supply descriptive threshold tables with
precision, recall, F1, selected candidates, coverage, executed trade count,
policy-aware simulated PnL, and ECE. They are saved as JSON and
`threshold_sweep_validation.parquet`. These descriptive outer tables do not
change the fitted per-model threshold. Candidates/trades retain selected
threshold, source, model kind, and actual calibration method. Final-test data is
rejected by model fitting and threshold sweeping; only the frozen threshold is
applied when a later permitted test evaluation occurs.

Global accounting for expanded search/model-selection budgets remains #16. Do
not interpret current fixture reports as multiple-testing-adjusted alpha claims.

## Verification

New tests cover synthetic stopping curves and restored weights; train-only
scaling; exact-span holdouts and tiny-fold audits; deterministic bootstrap and
two-seed means/dispersion; linear/tree schema; mapping variants and immutable
saved forecasts; calibration/bucket arithmetic; all four meta types under all
three calibration modes; empty/single-class fallbacks; chronological calibration
isolation; monotone threshold coverage; model/threshold selection; and final-test
rejection.

A complete two-trial fixture runs two LSTM seeds with constant versus logistic
meta models, confirms both saved prediction members and aggregate diagnostics,
and keeps final test and the forecast benchmark disabled. A separate forecast-only
fixture runs tabular models and fake TimesFM, records its horizon/scaling audit,
and confirms zero test reads. No optional model downloads are needed for tests.

Final verification after review fixes: **201 passed, 37 warnings in about 100
seconds**, using the shared project virtual environment and
`OMP_NUM_THREADS=1 python -m pytest -q`. This includes the existing CUDA smoke
checks. The warnings come from scientific-library calculations on constant or
nearly constant synthetic series. Both smoke configs below also completed on the
committed Lighter 5m data (2m07s and 45s on the RTX 4090), and the training
smoke was delivered to a scratch Trackio project and read back with the
`forecast.*`, `forecast_baselines.*`, `model_instability.*`, and per-seed
`training_final.*` namespaces present.

Reusable local smoke configs (require an existing raw-data fixture):

```bash
uv run python -m src.experiments.staged_trial --config configs/training_discipline_smoke.yml --smoke
uv run python -m src.experiments.forecast_benchmark --config configs/forecast_diagnostics_smoke.yml --smoke
```

Both disable final test and external Trackio delivery. Use CPU only for explicit
fixture debugging by adding `--device cpu --allow-cpu`. Historical performance
experiments require their separately frozen specifications and normal Trackio
policy, and remain unrun. Remaining issue dependencies are in the repo-root
`experiment-readiness.md`.
