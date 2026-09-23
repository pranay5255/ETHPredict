# AFML Experiment Controls

Use `configs/config.yml` as the baseline v2 research spec. Freeze model, labels,
validation, costs, search budget, and alpha policy before varying one AFML control.

## Bar Clock

Keep the baseline time bars:

```yaml
bars:
  type: time
  time_interval: 5m
```

Compare against activity-driven bars by changing only the bar block:

```yaml
bars:
  type: volume
  threshold_volume: 50000
  min_rows: 256
```

or:

```yaml
bars:
  type: dollar
  threshold_usd: 100000000
  min_rows: 256
```

The run artifacts record raw file hashes, bar coverage, timestamp gaps,
missingness, serial correlation, volatility distribution, and label balance.

## Side Data

Side-data joins must stay disabled unless a join reports enough coverage under the
configured timestamp tolerance. Use `join_side_data_asof` and
`validate_side_data_coverage` from `src.data.features_all`; failed coverage should
block feature use instead of silently forward-filling sparse data.

## Feature Families And Fracdiff

`features.include` is the authoritative v2 feature-family selector. Vary only this
list for feature ablations. `features.frac_diff_mode: fixed_width` is the
reproducible AFML-oriented baseline; `auto`, `fixed`, and `none` are comparison
modes.

## Observation Sampling

Dense rolling samples remain the baseline:

```yaml
sampling:
  mode: dense
```

Event-sampled runs can use:

```yaml
sampling:
  mode: cusum
  cusum_threshold_bps: 10
```

or:

```yaml
sampling:
  mode: volatility_cusum
  volatility_multiplier: 1.0
  min_threshold_bps: 1.0
```

Edge-triggered observation sampling can use a feature or prediction column already
present in the feature frame:

```yaml
sampling:
  mode: edge_triggered
  edge_column: log_return
  edge_threshold_bps: 10
```

Candidate trade filtering remains a later policy step after forecasting.

## Frozen Rerun Comparison

After two frozen reruns, compare the run directories or their `manifest.json` files:

```bash
uv run python -m src.experiments.reproducibility artifacts/runs/run_a artifacts/runs/run_b
```

The comparison checks config hashes, raw data hashes, trial count, trial IDs, split
manifest hashes, and metric variance.

## Final-Test Reuse And Run Identity

Trial selection reads validation metrics only. `pipeline.selection_metric` paths
that contain `metrics.test` are rejected before any trial is trained. After
selection, the test split is backtested only for the roles in
`research.final_test_guard.evaluate_selection_roles` (default `raw_best` and
`trade_qualified_best`). Each of those evaluations appends one entry to the
final-test ledger. Unselected trials keep an empty `metrics.test` and do not
touch the ledger.

The ledger key is a hash of the research specification plus the split hash and
an evaluation scope (`meta_label_trial` or `forecast_benchmark`). The research
specification is data, bars, features, sampling, sample weights, targets,
labels, model, training, validation, costs, alpha policy, selection policy, and
seed. `pipeline.run_name`, `pipeline.artifact_root`, experiment labels, and
tracking settings are not part of the hash, so renaming a run is still reuse.
Set `research.final_test_guard.do_not_reuse_test: true` to refuse a second
non-smoke evaluation of the same specification. Pin `ledger_path` when runs
must share a ledger across different artifact directories. Smoke runs increment
the counter and do not block.

`research.raw_data_guard.on_change` controls what happens at run start when the
raw-file hashes differ from the last run of the same research specification:
`warn` (default), `fail`, or `ignore`. Smoke runs record a warning instead of
failing. The comparison result is stored on `run_identity.raw_data_guard`.

Every v2 `feature_manifest` includes `code_identity`, a hash of
`src/data/features_all.py`, `src/features/labeling.py`, and
`src/features/sample_weights.py`, plus a `family_identity_hash` of the configured
feature families, columns, and fracdiff mode. Git summaries list the commit,
dirty state, and critical untracked or modified paths. That critical set
includes `configs/`, `src/`, `tests/`, `context/`, `scripts/`, `TASKS.md`,
`pyproject.toml`, and `uv.lock`.

The forecast benchmark writes `split_manifest.json` with the purged
walk-forward boundaries it actually used (development, gap, test, and each
fold). Each benchmark model that reads the test split increments the same
final-test guard under the `forecast_benchmark` scope before its test metrics
are computed.
