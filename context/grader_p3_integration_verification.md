# P3 acceptance verification — 2026-10-29

P3 acceptance is verified in [PR #63](https://github.com/pranay5255/ETHPredict/pull/63).
The pushed `grader-v1` tag pins commit
`58228cbad987ea59f1d172b461fba823a8de0e64`. The final full suite passed **583 tests**
with one intentional historical-parity skip. A separate checkout pinned to that
exact tag completed an actual production dev CLI grade: every integrity check
passed, and the flat control scored zero under the unchanged economic gates.
See [the production receipt](grader_v1_production_verification.json),
[final baseline measurements](grader_v1_baselines.md) and
[repeated-measurement comparison](grader_baseline_final_validation.json).

Earlier checkpoints below retain the supporting verification history. Their
pre-freeze status statements describe those checkpoints; the final acceptance
and production evidence above supersede them.

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
At that checkpoint the full report schema and baseline-cache integration still
needed final review; the later completion evidence follows below.

Baseline initialization is now wired to the production CLI. Its cache key
covers the grading fingerprint, execution-data identity, declared market set,
clock and tier. A key-specific OS lock serializes construction; each attempt
retains stable artifact paths and only a validated complete bank gets an atomic
index. Existing corrupt indexes/banks fail closed. The reader rederives the
winning baseline, per-seed thresholds and Sharpe history from all six reports,
and verifies their artifact manifests. Initialization runs after the candidate
worker releases its GPU reservation. Shadow bank/cache paths must stay under
the private service. The full real-data bank was still pending at that checkpoint;
its completed measurements are recorded below.

```text
pytest tests/test_grader_baseline_bank.py tests/test_grader_baseline_cache.py \
  tests/test_grader_grade.py -xq
21 passed in 80.35s (0:01:20)
```

Cache/bank unit fixtures use synthetic trusted-report stubs; they do not stand
in for the full real-data baseline acceptance table. The grade tests use real
audited workers and verify that the provider runs after worker shutdown and its
bank snapshot is included in the trial's artifact manifest.

## Full measurements and pre-freeze audit — October 29

Closed report schemas cover scored components, economic gates, feature
diagnostics, model metadata and resource receipts. Unknown nested fields are
rejected, while complete input provenance is separate from the common execution
identity. Targeted report checks: `5 passed in 84.32s (0:01:24)`.

The integrated full suite, including the pinned SDK, actual P2 denial/Docker
probes and pinned TimesFM CUDA fixture, passed:

```text
567 passed, 1 skipped, 23 warnings in 725.87s (0:12:05)
```

All six baselines then completed both seeds and all five full public ETH dev
blocks at 1h. See [per-block measurements](grader_v1_baselines.md). Flat is best
by net PnL. Its pre-gate score is 0.100; all pre-gate scores lie in 0.100–0.174.
Initial anchors, weights and every gate are retained, with calibration marked
complete. Every final baseline score is zero. All non-tag integrity checks pass;
the expected pre-freeze tag failure remains explicit.

Primary-source review found that the pinned TimesFM snapshot was published in
October after the September family announcement. The registry now uses an
October 3 UTC evidence boundary. The [provenance review](grader_checkpoint_provenance.md)
and [correction receipt](grader_baseline_metadata_correction.json) verify that
this metadata correction retained strategy sources, all numeric-score/resource
digests and every covered artifact. The corrected bank passes full hash checks
and a cache-hit probe with training deliberately disabled.

Trial-ledger validation also rejects malformed UTC timestamps, numeric overflow,
invalid scores/hashes, decisions before results and invalid keep decisions.
Qualitative ratings cannot override the strict numeric comparison. Crashes and
interrupted attempts remain counted without inventing missing Sharpe values.

```text
pytest tests/test_grader_history.py tests/test_grader_models.py \
  tests/test_grader_score.py tests/test_grader_baseline_bank.py \
  tests/test_grader_baseline_cache.py -xq
46 passed in 3.58s
```

## Final causality audit and freeze verification

The first final suite passed `582 passed, 1 skipped, 23 warnings in 742.90s`.
A subsequent adaptive fixture reproduced a real causality bypass: a look-ahead
policy could anticipate the policy-seeded probe locations and mask just those
rows. Audit locations now use 128-bit parent-only entropy, drawn only after the
reference positions are committed. Policy seeds, the 20-check count, tolerance,
scoring and every gate are unchanged. The nonce is never sent to a worker.
The regression reproduces the old bypass with controlled public locations and
rejects the same policy with independent audit locations. Checks and complete
grade integration: `11 passed in 119.38s (0:01:59)`.
All six adapters have now completed the repeat under the corrected guard. The
[final receipt](grader_baseline_final_validation.json) verifies identical strategy
code, performance values and all 60 position files against the first fixed-adapter
measurement. Every block passed its 20 independently drawn prefix probes. The
[baseline table](grader_v1_baselines.md) now contains the final measured runtimes:
77.891s flat, 78.276s buy-and-hold, 114.509s momentum 1h, 114.167s momentum 24h,
494.895s TimesFM and 1165.877s LSTM. No strategy parameter, rubric number or gate
changed. The final full suite after this guard passed:

```text
583 passed, 1 skipped, 23 warnings in 765.36s (0:12:45)
```

The pinned SDK, actual P2 PermissionError/Docker probes and TimesFM CUDA
fixture were enabled. The historical CSV overlap remains the sole intentional
skip. No grader or strategy code changed during this suite.

The annotated tag is pushed. From the separate pinned checkout, the real
`uv run --no-sync python -m grader.cli grade --strategy <candidate> --tier dev`
command used the public lake and the verified final baseline bank. The empty
smoke ledger counts six baselines plus this control. It completed both seeds,
all five blocks per seed and 225 audited worker calls. All 200 proper-prefix
comparisons and repeat fits passed. The report's grader hashes match the tag,
and its complete artifact manifest verifies. Output:

```text
score: 0.000000000000
status: rejected
```

Only economic/statistical checks reject the flat control; zero trades and
unavailable Sharpe are reported honestly. Exact report, manifest and grader-file
hashes, worker runtime and memory peaks are in the production receipt. No
`grader/` file has changed since the tag.

P4 private broker/driver and P5 untuned scaffolds follow P3's merge. Private
runtime provisioning requires the human: ethgrader cannot traverse the human's
0750 home. No home permission change or Docker group membership is proposed.

No real hidden evaluation, holdout evaluation, consumed-test rerun, strategy
search or edge claim occurred. Red-team tests use synthetic inputs; the baseline
measurements use the public dev lake and the existing pinned public model cache.
