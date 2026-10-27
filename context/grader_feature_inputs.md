# P3 engineered-feature verification

This contract is under review in P3; it is not yet frozen at `grader-v1`.
An engineered-feature declaration does not authorize a strategy to supply its
own importance values. The trusted grader measures actual policy changes with
the unchanged vendored `permutation_meta` and `summarize_importance` routines.

## Optional strategy methods required for the feature gate

Declare numeric input families in `engineered_features` before fitting. In
addition to the ordinary `fit` and `predict` methods, expose:

```python
def feature_packet(self, history, decision_times) -> FeaturePacket: ...
def predict_features(self, packet: FeaturePacket) -> pd.DataFrame: ...
```

Import `FeaturePacket` from `grader.feature_packets`. Its `frame` has UTC
`timestamp`, `market`, and finite floating numeric policy inputs, on the full
declared decision grid. `families` partitions the real input columns using the
predeclared family names. `noise` contains the two groups `diagnostic_noise_0`
and `diagnostic_noise_1`. Every numeric column belongs to exactly one group.
`noise_lags` gives each noise column's nonnegative lag in decision bars, within
the declared lookback. Sequence inputs may use separate columns for each lag.
Unknown metadata, extra columns, missing markets, duplicates, nonfinite inputs
and forged noise values fail closed. Parquet metadata bounds decoded size
before numeric arrays are read; timestamps are compared without rounding.

Use `ctx.noise(available_timestamps_ms, market=market)` in training and
`feature_noise(..., seed=self.seed, market=market)` in feature extraction.
Both return the same two float32 Gaussian controls, keyed by the grader seed,
market and absolute availability timestamp. Prefixes and shifted history
windows therefore receive identical controls for identical observations.
These synthetic controls never replace missing observations or labels.

The normal worker independently obtains the ordinary positions and the
input-only positions. Their full target paths must agree within `1e-9`.
Feature inputs themselves must also pass the 20 proper-prefix checks per
block and the repeated same-seed fit check. A strategy without the optional
methods can still run ordinary integrity checks, but its declared engineered
features cannot pass the credible-features gate.

## Trusted permutation evaluation

The five outer evaluation populations use the existing purged growing-window
splits. Fitted state is reloaded in a fresh isolated worker for every
counterfactual prediction. The worker receives numeric inputs and immutable
fitted state, with no evaluation labels, raw evaluation history or lake view.
Access attempts outside that reduced allowlist are denied and audited even
when candidate code catches the error. All calls count toward the same budget
as fitting, ordinary prediction and integrity reruns.

The vendored seeded permutation routine selects each family three times. The
adapter converts its random keys into within-market permutations, moving an
entire family's input window together and retaining every declared market.
Only signals that execute before the block ends contribute diagnostic labels;
the configured entry delay is applied and nonexecuted tail rows are recorded.
Labels are the sign of the actual next execution-interval return. The fixed
probability proxy is `(target_position + 1) / 2`; the original routine computes
F1 and clipped binary log loss. This measures whole-policy dependence on
numeric inputs, not a model's claimed explanation or calibrated probability.

The unchanged summary requires a stable positive real-family importance above
the noise controls and rejects noise winners. The economic gate consumes that
trusted summary. No manual importance override exists. This diagnostic alone
does not establish a trading edge or a causal relationship in the market.

## Baseline adaptations

Momentum exposes its causal return and both controls. Its analytic rule
ignores the controls, whose policy importance is consequently zero.

The LSTM wrapper retains original feature generation, training, nested OOF
meta-labeling, calibration and forecast mapping. Both controls enter base and
meta training with absolute-time keys. Numeric sequence lags retain the
original feature-family grouping. Additional inputs are realized volatility
and the original `CostModel`'s causal quotes for each side. The input-only
decoder serves those stored quotes to the original candidate mapping; it
cannot consult raw evaluation bars or future funding. Diagnostic truth fields
remain explicitly unknown and are never model inputs. Fixed-size inference
batches pad only with a duplicate known context, discard padding outputs, and
keep CUDA rounding independent of prefix length. No policy thresholds or
model hyperparameters were tuned.
