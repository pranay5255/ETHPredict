# ETH Price Prediction Signal Design Guide

## 1. Philosophy

We treat **ETH/USD** as a complex adaptive system driven by *on‑chain activity, macro liquidity, micro‑structure, and social attention*.  
Following López de Prado’s “meta–strategy” paradigm, every step—data curation, feature engineering, labeling, model selection, back‑testing, deployment—is an *independent assembly‑line task* that can be unit‑tested and monitored fileciteturn0file0.

> **Goal:** build a forecasting signal that materially exceeds a **random walk** or **rolling‑mean** baseline in out‑of‑sample (OOS) risk‑adjusted returns while obeying strict anti‑leakage and anti‑overfitting rules.

---

## 2. Data Pipeline

| Layer | Source | Granularity | Notes |
|-------|--------|-------------|-------|
| **Price / Volume** | Binance spot `ETHUSDT` | 1‑hour | Use *dollar* bars if equities–style micro‑data is later available to equalise information flow fileciteturn0file7 |
| **Chain TVL** | DeFiLlama | 1‑hour resample | Both chain TVL and top‑10 protocol TVLs |
| **On‑chain Fundamentals** | Santiment | 1‑hour resample | Active addresses, dev activity, network growth, social metrics |
| **Consensus / Macro** | Beacon chain slots, fed funds, DIXY | Daily | Add later to capture macro shocks |

### ETL Checklist
1. **Timestamp integrity** – keep UTC, forward‑fill only after resampling.  
2. **Versioned raw files** (`data/raw/…`), processed files (`data/processed/…`).  
3. **Unit tests**: row counts, non‑negativity, missing‑value ratio.

---

## 3. Feature Engineering

> Rule of thumb: **stationary ⟺ learnable**, **memory ⟺ predictive**.

### 3.1 Fractional Differentiation
Compute `fracdiff(series, d*)` for each non‑stationary series; choose the smallest *d* s.t. the ADF statistic crosses the 95 % stationarity threshold while keeping the correlation with the original series ≥ 0.97 fileciteturn0file7.

### 3.2 Traditional Signals  
* **Returns & Volatility:** log‑returns, rolling σ (1 h, 24 h, 7 d).  
* **Micro‑structure:** high‑low estimator, volume‑imbalances if L2 data arrives fileciteturn0file2.  
* **Liquidity Ratios:** price/TVL, mcap/TVL, volume/TVL.  
* **Growth Metrics:** ∆ active addresses, ∆ dev‑commits, ∆ social dominance.  
* **Entropy & Structural Breaks:** rolling Shannon entropy of returns; CUSUM & SADF explosive‑move flags fileciteturn0file2.  

### 3.3 Regime Features
* **Volatility Regime**: discretise realised σ into *low / mid / high*.  
* **Funding Rate Regime** (if perp data added): positive/neutral/negative.

All features are z‑scored _within_ training folds to avoid look‑ahead bias.

---

## 4. Labeling Strategy

We adopt the **(meta‑)triple‑barrier** method:

| Barrier | Definition |
|---------|------------|
| **Upper** | +κ × σ_{h} cumulative return |
| **Lower** | –κ × σ_{h} |
| **Timeout** | τ = 24 h |

*Primary label* `y_dir` ∈ {–1, 0, +1}.  
*Target regression* `y_ret` = next‑hour log‑return (scaled).

Meta‑labeling: train an auxiliary classifier to predict *whether the regression signal’s hit is trustworthy*, then weight positions by the meta‑probability fileciteturn0file5.

---

## 5. Sample Weighting

Because adjacent hourly bars overlap, we compute **sample uniqueness** weights `w_i = 1 / (# overlapping labels)` and further apply *volatility‑scaling* so that high‑σ periods do not dominate fileciteturn0file7.

---

## 6. Cross‑Validation & Hyper‑Tuning

| Step | Method |
|------|--------|
| **CV** | **Purged K‑Fold (K = 5)** with **embargo = 3 h** to prevent information leakage fileciteturn0file6 |
| **Combinatorial Purged CV** | For final evaluation of top models |
| **Optimiser** | Optuna ≥ 3.0, objective = out‑of‑fold **Information Ratio** |
| **Early Stopping** | 10 rounds for boosting nets; patience 5 for DL |

---

## 7. Model Stack

1. **Tree‑based Learners**  
   * Gradient Boosted Trees (LightGBM) – handles heteroskedasticity.  
   * ExtraTrees for variance reduction.
2. **Sequence DL**  
   * LSTM/Transformer Encoder fed with 24‑step tensors (see `ensemble.py`).  
3. **Stacking Ensemble**  
   * Blend tree predictions (`pred_*`) + raw features + conviction probability → LSTM regressor.

Regularise trees with *subsample_col* and *max_depth*; apply `dropout=0.2` in LSTM.

---

## 8. Baselines & Metrics

| Metric | Rationale |
|--------|-----------|
| **MAE / RMSE** | point forecast quality |
| **Directional Accuracy** | ≥ 55 % target |
| **Hit Ratio & Avg Hit : Miss** | trading intuition |
| **Information Ratio (IR)** | risk‑adj. performance |
| **Max Drawdown & Time‑Under‑Water** | capital efficiency checks fileciteturn0file4 |

*Baselines*:  
* **Random walk** – shuffled returns.  
* **Rolling mean** – 24 h SMA forecast.  
* **Vol‑scaled naïve drift** – μ_{n‑24:n}.

Success ⇒ IR_{OOS} > 0.5 and outperform baselines at 95 % confidence.

---

## 9. Back‑Testing Protocol

1. **Walk‑forward** on chronological splits: train past 60 d → test next 7 d.  
2. **Transaction Costs**: assume 2 bp per side.  
3. **Slippage**: model with linear impact `0.1 bp × (trade_vol / hourly_vol)`.  
4. **Position Sizing**: Kelly‑style bet sizing from predicted probability `p` → size = `2Φ(z)–1`, where `z = (p – 0.5) / √(p(1‑p))` fileciteturn0file5.  
5. **Capacity Test**: stress notional up to 10 × current volume.  
6. **Result Dashboard**: CAGR, Sharpe, Sortino, DD, TuW.

---

## 10. Risk Management

* **Hard stop** at –2 σ intraday.  
* **Shadow volatility target**: scale exposure so rolling 30 d σ_pnl ≤ 15 % annualised.  
* **Ensemble diversity**: monitor feature importance & correlation between learners to avoid crowded trades fileciteturn0file2.

---

## 11. HPC & Production

Large hyper‑parameter sweeps are vectorised, multi‑threaded and submitted to an HPC queue (or Ray on‑cluster) to exploit embarrassingly parallel workloads fileciteturn0file1.

Deployed pipeline:

```
prefect flow:
    extract → preprocess → CV/train → backtest → publish S3:/signals/eth.parquet
```

* Store model artefacts with `mlflow` + git hash of training code.  
* Real‑time inference containerised (Docker) & served via FastAPI; Kafka event triggers portfolio rebalance.

---

## 12. Implementation Roadmap

1. **ETL & Feature Store**  
   * Extend `preprocess.py` with fractional‑diff & entropy features.  
2. **Label Module** (`label.py`) with triple‑barrier.  
3. **SampleWeight Module** (`weights.py`).  
4. **Model Registry** – move `ensemble.py` into `/models/stack.py`.  
5. **Research Notebook**: sanity checks & diagnostic plots.  
6. **CI/CD** – unit tests on every PR; nightly walk‑forward backtest.

---

## 13. References

* López de Prado, *Advances in Financial Machine Learning*, lecture decks 2018 (Features, Data Analysis, Backtesting I&II, Modelling, Machine‑Learning Portfolio) fileciteturn0file2turn0file7turn0file5turn0file4turn0file6  
* Marcos López de Prado, *Fractionally Differentiated Features*, SSRN 2018.  
* Chan et al., *Machine Learning for Crypto Markets*, 2024.

---


---

## 14. Code‑base Integration Guide — “wiring” the new features & labels

Below is a concrete, file‑by‑file checklist that **must** be executed so the research ideas become working code.

| File | Change | Key Lines / Functions |
|------|--------|-----------------------|
| **`preprocess.py`** | 1. **Add imports**<br>`from statsmodels.tsa.stattools import adfuller`<br>`from scipy.signal import welch` (entropy)<br>2. **Fractional differentiation helper**<br>`def fracdiff(series, d): …`<br>3. In `prepare_features` and `get_base_dataset`:<br>&nbsp;&nbsp;• compute `fracdiff_close`, `return_entropy_24h`, `cusum_flag`, `sardf_flag`.<br>&nbsp;&nbsp;• Append them to `feature_cols`.<br>4. **Return labels** – do **not** touch here; labels live in `label.py`. | `prepare_features`, `get_base_dataset` |
| **`label.py`** *(new)* | Implements **triple‑barrier** and **meta‑labeling**:<br>`def triple_barrier(df, ub, lb, timeout): …`<br>Outputs: `y_dir`, `y_ret`, `meta_conf`. | new file |
| **`weights.py`** *(new)* | Computes **sample uniqueness** & **volatility weights**:<br>`get_sample_weights(events, price_series)` | new file |
| **`ensemble.py`** | 1. **Import new label & weight modules**.<br>2. Replace current `targets_df` with `y_ret` (regression) and add `y_dir` (classification).<br>3. Pass `sample_weights` to `fit` where supported (e.g. `lgb.train`).<br>4. Concatenate new engineered features (those added in `preprocess.py`) and meta‑labels (`conviction`, `meta_conf`). | top; dataset build section |
| **`model.py`** | If using Transformer instead of LSTM:<br>`class PriceTransformer(nn.Module): …` (optional). | create/extend |
| **`config.yaml`** *(new)* | Centralise hyper‑parameters: feature list, CV folds, κ, τ, embargo hours, transaction cost assumptions. | new file |
| **`tests/test_preprocess.py`** | Add unit tests for **stationarity check** (ADF), **shape invariants** after adding new columns, and **NaN absence**. | new file |
| **`notebooks/eda.ipynb`** | Visual sanity checks for new features: stationarity heat‑map, feature importances, label distribution histograms. | update |

> **Tip** : Tackle modifications in the order given; each downstream step expects the previous layer to be finished and unit‑tested.

---

*All subsequent CI runs should pass `pytest` and execute the nightly walk‑forward back‑test to ensure that the new features still beat the baseline.*



---

## 15. **Model Hierarchy (PyTorch‑only)**

The entire modelling stack is rewritten to rely **exclusively on PyTorch**; all `sklearn` components are deprecated.

| Level | Purpose | Architecture | Input Tensor |
|-------|---------|--------------|--------------|
| **Level‑0 (Base)** | Direct regression of next‑hour TVL (or price) | `PriceLSTM` (2 × LSTM → FC) | `[batch, 24, F]` where **F = len(feature_cols)** from **`preprocess.py`** |
| **Level‑1 (Meta‑forecast)** | Combine base predictions with raw features to improve calibration | `MetaMLP` (3 × FC, ReLU, Dropout) | `concat([pred_0, last_features]) → [batch, F+1]` |
| **Level‑2 (Meta‑label)** | Binary classifier for “confidence” of regression hit | `ConfidenceGRU` (1 × GRU → FC + sigmoid) | Same `[batch, 24, F]` |
| **Level‑3 (P&L Simulator)** | *Not a NN* – rule‑based allocation using confidence score & Kelly sizing |

Training sequence:

1. Train **Level‑0** on `(X_seq, y_ret)` with **MSELoss** + **Adam**.  
2. Freeze Level‑0, generate predictions; train **Level‑1** on residuals (**SmoothL1Loss**).  
3. Generate direction labels (`y_dir`) via **`label.py`**; train **Level‑2** with **BCEWithLogitsLoss**.  
4. During inference: `pred_0 → pred_1` gives refined point estimate; `conf = Level‑2(x)` gates the position size.

---

## 16. Updated Code‑base Integration (PyTorch‑centric)

| File | Change Summary |
|------|---------------|
| **`ensemble.py` → `train_stack.py`** | Remove every reference to `sklearn`.<br>Define `PriceLSTM`, `MetaMLP`, `ConfidenceGRU`.<br>Build training loops with `torch.utils.data.DataLoader`.<br>Save checkpoints `price_lstm.pt`, `meta_mlp.pt`, `conf_gru.pt`. |
| **`preprocess.py`** | Preserve the original `feature_cols`; append any new engineered features **after** those indices to keep shape stable. |
| **`label.py`** | Houses triple‑barrier logic; returns `y_ret` (regression) & `y_dir` (classification). |
| **`weights.py`** | Supplies `sample_weights` tensor for optional loss weighting. |
| **`config.yaml`** | Central list of `feature_cols`; training script should `assert` equality with `preprocess.feature_cols`. |

---

## 17. Sanity Checklist

* Feature column order invariant.  
* CI enforces **no `import sklearn`**.  
* `xb.shape == (batch, 24, F)` & `yb.shape == (batch, 1)` unit‑tested.  
* Inference latency target: **< 3 ms** on RTX 4090.

---


**End of file.**