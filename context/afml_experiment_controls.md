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
