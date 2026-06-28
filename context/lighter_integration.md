# Lighter Integration Guide for ETHPredict

The active repository scope is public Lighter market data only. Legacy Binance,
DeFiLlama, Santiment, DEX simulation, and bribe-optimization planning notes are
archived under `archive/legacy_data_sources`.

## Active Data Path

1. Configure `configs/config.yml` with `data.sources: [lighter]` and the
   `data.lighter` market/date settings.
2. Collect data with:

```bash
python scripts/lighter_collect_data.py --config configs/config.yml
```

3. Keep the model OHLCV export in `data/raw/ETHUSDT-<resolution>-lighter-<start>-<end>.csv`.
4. Build OHLCV-only features with:

```bash
python -m src.data.features_all full_features --data-dir data --sequence-length 24
```

`DataPreprocessor` ignores non-Lighter raw CSV names. Lighter side-data files in
`data/lighter/` are preserved for later feature work, but the active model path
uses candles only.

## Out of Scope for This Pass

- Binance, DeFiLlama, and Santiment joins.
- DEX simulation configuration and code paths.
- Bribe or MEV optimization configuration.
- Signed Lighter trading, account state, nonce management, and `sendTx` flows.
