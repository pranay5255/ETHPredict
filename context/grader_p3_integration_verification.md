# P3 integration checkpoint — 2026-10-28

P3 remains in progress in [draft PR #63](https://github.com/pranay5255/ETHPredict/pull/63).
Code checkpoints: `f929810` (models, causal views and baseline adapters),
`a63ce5f` (grade/report CLI, trial accounting, bootstrap and red teams).
No `grader-v1` tag exists; P3 acceptance is not complete.

## Verified

- The complete synthetic flat grade runs two seeds, five blocks, 200 causal
  prefixes and repeat fitting: 225 worker calls. It scores zero and records
  every computed failed gate.
- All ten specified red teams pass through the shared evaluation/report path;
  both sealed-path variants are tested. Inputs are synthetic and reports use a
  fixture profile that is never loop-eligible. The sleeper has a shortened
  internal test deadline; the CLI exposes no budget override.
- All baseline adapters exist, including both momentum horizons. Their `.py.in`
  payloads cannot be imported as grader modules. Only the worker loads their
  materialized candidate code and editable-library imports. The original
  no-editable-import tests pass without exemptions. Source dependencies come
  from Git objects, pinned to `grader-v1` after freeze.
- The existing LSTM/meta recipe fits and passes 20 real causal prefixes on CPU
  and CUDA. Original training, nested OOF meta-labeling and calibration helpers
  are reused; time-valued 5m parameters retain their duration on the common
  clock. Both diagnostic noise controls use the grader seed.
- Pinned TimesFM 2.5 median-sign inference passes 20 CUDA causal prefixes. Its
  immutable backbone is referenced rather than duplicated in fitted pickles.
  Registry hashes, Apache-2.0 and conservative release labels are recorded.
- Worker provenance contains only the exact slice and its hash. Original
  monthly provenance stays in the trusted report; future partition metadata
  is not copied into an earlier slice.
- Synthetic two-lake shadow loading verifies public/private source hashes,
  retains the dev training prelude, rejects corruption, and never opens the
  synthetic August holdout canary.
- Strict trial events count crashes and interrupted attempts without inventing
  Sharpe values. All 1,000 stationary daily bootstrap replicates synchronize
  the seeds and retain each day's intraday cashflows.
- The production CLI rejects the untagged grader at zero before reading market
  data. Initial flat control and qualitative judge prompt exist; qualitative
  review cannot override the strict numeric keep rule.
- Engineered-feature evaluation now perturbs actual fitted-policy numeric inputs
  in fresh isolated workers without raw evaluation history or lake access. The
  unchanged vendored permutation/noise calculations feed the economic gate.
  Input-only targets must reproduce normal targets; feature packets themselves
  pass prefix causality and repeat-fit determinism. The momentum and LSTM
  wrappers expose the required inputs. See [the contract](grader_feature_inputs.md).

Full suite before the additional targeted checks below:

```text
519 passed, 1 skipped, 23 warnings in 579.04s (0:09:39)
```

The command is the full-suite command in [the earlier checkpoint](grader_p3_progress.md),
plus `ETHPREDICT_TIMESFM_MODEL_PATH` pointing at the pinned local 2.5 snapshot.
Real P2 OS-denial and Docker probes were enabled. The sole skip is P1's now-sealed
historical CSV overlap test; its exact 26,496-row pre-sealing receipt is preserved.
No package was added or upgraded.

Additional checks after that suite:

```text
two-lake source/hash/holdout-canary fixture: 1 passed in 1.58s
LSTM CUDA + post-exit peak-RSS enforcement: 2 passed in 38.90s
O_PATH/proc descriptor-alias denial:        1 passed in 0.07s
```

After the feature-input integration, targeted tests passed:

```text
pytest tests/test_grader_lstm_baseline.py tests/test_grader_baselines.py \
  tests/test_grader_checks.py tests/test_grader_score.py tests/test_grader_grade.py -xq
22 passed in 204.08s (0:03:24)
```

This includes the original LSTM on CPU/CUDA with 20 proper prefixes and an
additional input-only prediction reproducing its normal positions. A complete
synthetic engineered-feature grade runs both seeds and all five blocks, records
100 budgeted permutation calls, and supplies real summaries to the economic
gate. No policy threshold, gate, model hyperparameter or component map changed.
Timestamp storage units are normalized without rounding before packet grid
comparison; a regression fixture covers the pandas/Arrow mixed-unit issue.

Artifact/report checkpoint after that integration:

```text
pytest tests/test_grader_artifacts.py tests/test_grader_models.py \
  tests/test_grader_grade.py tests/test_grader_redteam.py -xq
29 passed in 300.54s (0:05:00)
pytest tests/test_grader_vendoring.py -xq
2 passed in 1.02s
```

Every report references a manifest of the trial's inputs, positions, traces,
opaque fitted checkpoints and other files. Hash verification rejects changed,
missing or unmanifested files; capture rejects symlinks, hard links, special
files and unreadable directories. The manifest/report exclusion avoids a hash
cycle and permits later statistical rescoring. Candidate checkpoints are never
interpreted in the scorer. Artifact/qualitative report fields reject unknown
keys, and JSON nonfinite values (including numeric overflow) fail at any depth.
Registered model provenance now separates each block's visible fitting window
from its actual next-open execution window, with release labels for each.
The full report schema and baseline-cache integration still need final review.

Baseline initialization is now wired to the production CLI. Its cache key
covers the grading fingerprint, execution-data identity, declared market set,
clock and tier. A key-specific OS lock serializes construction; each attempt
retains stable artifact paths and only a validated complete bank gets an atomic
index. Existing corrupt indexes/banks fail closed. The reader rederives the
winning baseline, per-seed thresholds and Sharpe history from all six reports,
and verifies their artifact manifests. Initialization runs after the candidate
worker releases its GPU reservation. Shadow bank/cache paths must stay under
the private service. No full real-data bank has been run yet.

```text
pytest tests/test_grader_baseline_bank.py tests/test_grader_baseline_cache.py \
  tests/test_grader_grade.py -xq
21 passed in 80.35s (0:01:20)
```

Cache/bank unit fixtures use synthetic trusted-report stubs; they do not stand
in for the full real-data baseline acceptance table. The grade tests use real
audited workers and verify that the provider runs after worker shutdown and its
bank snapshot is included in the trial's artifact manifest.

## Still required before freeze

1. Finish nested report validation and final integration review. Baseline-bank
   initialization and artifact persistence are wired and have targeted tests;
   the actual complete baseline run remains below.
2. Run all baselines on the full common dev blocks; commit per-block numbers and
   measured runtimes in `grader_v1_baselines.md`. No full real-data bank has run.
3. Calibrate only pre-gate component maps under the approved interpretation,
   preserve every gate and rejected score zero, run the final full suite, then
   push `grader-v1` and complete PR #63.
4. P4 private broker/driver and P5 untuned scaffolds follow P3 acceptance/merge.
   Private runtime provisioning still requires the human: ethgrader cannot
   traverse the human's 0750 home.

No real hidden evaluation, holdout evaluation, consumed-test rerun, strategy
search or edge claim occurred. Numerical model/grade tests here use synthetic
market inputs and the existing pinned public model cache.
