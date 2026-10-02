# Trackio dashboard schema

Trackio mirrors local artifacts. Panels should read manifest fields and receipt paths, not recompute selection or failure modes. Publishing a Hugging Face Space is a later decision; this note only defines the fields and panels a local or future Space view would show.

## Run identity

Each non-smoke research run logs three Trackio runs in group `meta_label_mvp`:

- `meta_label/<run_id>/<trial_id>` for every completed, failed, and skipped trial
- `meta_label/<run_id>/summary` for the selected-trial report
- `meta_label/<run_id>/accounting` for the trial-accounting summary

Forecast benchmarks use group `forecast_benchmark` and the name `benchmark/<run_id>/<model_id>`. `run_forecast_benchmark` applies the same Trackio policy as the meta-label runner: non-smoke runs require `tracking.trackio.enabled: true` unless `tracking.trackio.allow_local_debug_without_trackio: true`.

Local files stay authoritative. After a log attempt, `trackio_receipt.json` sits next to the trial manifest, `tracking/summary.json` records the summary attempt, and `tracking/accounting.json` records the accounting attempt. `trial_accounting.json` and each trial manifest store those receipt paths.

## Trial fields

| Field | Source |
| --- | --- |
| `trial_id`, `trial_index`, `trial_count`, `trial_status` | trial manifest |
| `reason`, `skip_reason` | failed and skipped manifests; both keys are logged |
| `selection_status`, `selection_roles` | validation selection, stored on the accounting row |
| `checkpoint_id`, `model_path` | saved base-model checkpoint |
| `config_hash`, `split_manifest_hash` | run identity and split manifest |
| `feature_families`, `feature_manifest` | stage-0 feature manifest |
| `costs`, `alpha_policy` | resolved config |
| `failure_modes` | the object stored on the trial and on `trial_accounting.json` |

`failure_modes.flags` is computed once from the trial's first selection role and then logged unchanged. Dashboard code must not derive the flags again from `selection_status`.

Flag names: `no_trade_winner`, `low_trade_winner`, `poor_calibration`, `high_turnover`, `unstable_validation_test`, `high_cost_sensitivity`, `concentrated_pnl`.

Metrics are grouped so they stay distinct:

- `alpha.validation.*` and `alpha.test.*` are simulated backtest results
- `forecast_oof_by_fold.*` is out-of-sample forecast quality
- `training_final.*` is the final fit loss
- `trial_accounting.failure_modes.*` is the persisted flag object

## Panels

1. Trial inventory: count of completed, failed, skipped, no-trade, low-trade, and trade-qualified trials from the accounting summary.
2. Selection: raw-best id, trade-qualified id, classification-best id, calibration-best id, and the validation metric used.
3. Failure modes: one series per flag above, read from `trial_accounting.failure_modes`.
4. Checkpoint: content hash, model path, split hash, and config hash for the saved trial.
5. Backtest: validation and test net PnL, trade count, and coverage, shown as separate series.
6. Delivery: receipt path and `delivery` (`logged`, `disabled`, or `failed`). A missing local readback fails a non-smoke run unless the debug override is set.

## Hugging Face Spaces

Do not publish a Space as part of this control. A future Space would be a read-only view of the same fields, pointed at the Trackio project `ethpredict`, and would not train models or submit orders. Until that decision, use the local receipt files and `uv run trackio show --project ethpredict` when Trackio is enabled.
