# ETHPredict dependencies and runtime — 2026-10-17

`pyproject.toml` and `uv.lock` are the active dependency contract. Use `uv sync`
and `uv lock --check`; `requirements.txt` is retained for legacy setup. Python
requires 3.13 or newer. Linux neural research uses the configured CUDA 12.8
PyTorch wheel source and defaults to `cuda:0`; CPU neural runs require an explicit
override. ARIMA/SARIMAX and most tabular diagnostics run on CPU.

## Active components

- Public collection: `src/data/lighter_client.py`, `scripts/lighter_collect_data.py`
  and `scripts/lighter_backfill_lake.py`; requests, pandas, PyArrow and dotenv.
- Features/bars/side data: `src/data/features_all.py`, `src/data/side_data.py` and
  related modules; pandas, NumPy, statistical/indicator packages. Default inputs
  are Lighter ETH CSVs. The Parquet lake is collected but not generally wired in.
- Labels/weights: `src/features/labeling.py`, `src/features/sample_weights.py`.
  Splits: `src/validation/splits.py`; exact-span/purged validation and audits.
- Research orchestration: `src/experiments/staged_trial.py` dispatches v2 configs
  to `src/experiments/meta_labeling_mvp.py`; forecast benchmarks have a separate
  artifact lane. PyTorch and scikit-learn provide neural/tabular models;
  statsmodels/arch support statistical baselines. TimesFM is optional at runtime
  and must record its actual model/configuration or skip/failure status.
- Evidence: reproducibility/test-use manifests, Parquet prediction/trade tables,
  JSON/Markdown reports and mandatory non-smoke Trackio delivery/readback. Local
  artifacts remain authoritative; tracking is not just CSV logging.
- Notebook/EDA: Jupyter, nbformat/nbclient/ipykernel via the locked Jupyter stack;
  `notebooks/execute_research_pipeline.py` prints progress while executing the
  notebook. The custom `eda_frontend/` bundles Plotly and reads distilled JSON;
  `scripts/experiment_eda.py` provides inventory/build/serve and explicit cleanup.
- Trading: the official `lighter-sdk` source and `external/lighter-python`
  examples support a separate guarded signed-testnet path. Public lake collection
  neither reads private keys nor places orders.

## Implementation boundaries

Search/importance/execution enhancements are in open PR #51. Forecast transforms,
checkpoint reload enhancements, the bounded development runner and the economic
gate are in stacked draft PR #53. Inspect their branch dependencies before using
commands from those reports. These modules are not claimed to be on merged main.

No active vLLM/sglang, DEX/Foundry or Docker deployment requirement is established
by the current research path. Legacy planning notes are under
`archive/legacy_data_sources/`. Current coverage and the funding gap caveat are
in [lighter_data_lake.md](lighter_data_lake.md); current priorities are in
[PROJECT_STATUS.md](../PROJECT_STATUS.md).
