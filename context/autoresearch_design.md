# Frozen research harness design

Status: P0 merged in #60; P1 merged in #61, 2026-10-27. P2 sealing/isolation is verified (human seal, 259 OS denials and a tested
dev-only agent tool worker) and merged in #62 at `86a9df0`. P3 is in progress;
P4–P5 have not started. This project builds a research environment, grader and bounded
driver. It does not search for an edge during construction, submit orders, sign
transactions, read credentials or promote a strategy. Existing bounded studies
lost money after costs. No strategy edge is claimed.

## Starting evidence and scope

`main` is `c365a81`; PRs #51 and #53 are merged; #16/#17/#20 are closed. #54's
funding repair is on main. The fresh structural audit is
[`autoresearch_start_audit.json`](autoresearch_start_audit.json), produced by
`scripts/audit_autoresearch_start.py` before sealing. P0 verifies all 25 dataset
directories, all 425 manifest hashes, no duplicate timestamps or internal funding
gaps, and unchanged historical input/artifact hashes. The audit is not a backtest.
The active training path still reads the ETH CSV. New work uses separate
`autoresearch/p<N>-<slug>` worktrees and one PR per phase; each acceptance gate
must pass before starting the next. Recheck concurrent PRs before pushes and
merges; merge only with `--match-head-commit`.

The existing test-use ledger
`artifacts/first_bounded_16_17_20/_final_test_reuse_ledger.json` records the
consumed last 20% of the March 28–June 28, 2026 ETH CSV experiment. Its count is
one. That test is no longer clean; its period may be development data in the
dev/shadow layout. Do not repeat it as a test or advertise its reuse as new
evidence. Holdout/promotion #41, variable sizing #19, the #43/#44/#45 current-state
chain and starting an open-ended overnight loop are outside this implementation.

## Frozen grader and editable strategy

The source of truth is a separate, clean worktree pinned to the pushed
`grader-v1` tag. All grader files, tier windows, model identities, cost settings,
rubric anchors, gates and driver parameters have recorded hashes. A changed
grader requires `grader-v2` and a new leaderboard. The runner starts with a clean
environment, removes candidate paths from the grader import path, and must not
import `src/`, `strategy/` or any candidate-editable module. Candidate code is
loaded only in an isolated child. A candidate edit to its own `grader/` has no
effect on evaluation and is rejected by the driver.

Planned files under `grader/` are `data.py`, `tiers.yaml`, `env.py`, `reward.py`,
`costs.py`, `stats.py`, `splits.py`, `feature_importance.py`, `runner.py`,
`checks.py`, `rubric.yaml`, `score.py`, `judge.md`, `baselines/`, and `cli.py`.
`strategy/strategy.py` and `strategy/README.md` are agent-editable, as are the
feature, preprocessing, model, training and policy libraries in `src/`.
`autoresearch/program.md` is human-maintained; `run_experiment.py` is trusted
driver code. Results/ledger/artifact bundles are untracked.

Vendor the existing `CostModel`/`dataset_cost_model`, statistics, validation,
selection and feature-importance logic, preserving original source hashes and
attribution. Vendor their necessary transitive dependencies too: cost coverage
errors/duration helpers, statistics' sample-weight helpers, and importance's
feature-family declarations. Do not let a lazy import reach editable `src/`.
Adapt only the boundary needed for standalone grader execution, document each
adaptation and test parity against the originals. Strategy baselines may use
existing training/checkpoint/TimesFM adapters inside the candidate process; those
imports must never enter the trusted scoring process.

All communication crosses files: grader-owned input Parquet slices and an exact
JSON request contract; candidate-owned positions Parquet and scratch artifacts.
No Python pickle or candidate-generated executable is read by the scorer.
Unknown request, position and report fields fail validation. No live-order script
is reachable from any grader, loop or seed entry point.

## Lake contract and units

Select explicit markets, dataset, resolution and half-open UTC `[start, end)`
windows on integer `timestamp_ms`. Allowed datasets are `candles_1m`,
`candles_5m`, `candles_1h`, `fundings_1h` and `mark_price_5m`; resolution must
agree with the dataset. Verify each selected partition against `_manifest.json`
before deserializing it. Reject missing/unlisted files, hash mismatches, schema
errors, invalid identities and conflicting same-timestamp records. Identical
duplicates may collapse. Unknown values are never zero-filled or forward-filled.

Return both data and provenance: market, market ID, dataset, resolution,
manifest identity, partition paths/hashes/counts, selected row counts, first/last
timestamps and explicit gaps. Multi-market loading preserves provenance per
market. A gap is evidence to reject a scoring path when its clock requires it;
it is not permission to silently synthesize a row. Funding completeness includes
the boundary payments required by the existing cost model.

Candle/mark timestamps identify bar opens; availability is
`timestamp_ms + resolution`. Funding is available at its payment timestamp.
Joins select only already available records and enforce freshness; they preserve
an explicit unavailable state for missing history. A decision exactly at a bar's
close may see that bar, never the following bar. Mark history begins August 25,
2025 (actual first observed 5m bar is 15:25 UTC), not midnight; earlier mark
features are unavailable. They must not become zero-valued price observations.

OHLCV comparisons use the existing CSV path's documented units and timestamps,
with no price/volume fitting or strategy-dependent conversions. P1 must compare
the complete common ETH 5m CSV/lake interval, record row counts and differences,
and test the conversions. A real disagreement requires a human decision before
proceeding. Do not change historical files to make a parity test pass.

## Tier windows and walk-forward blocks

- Dev: `[2025-06-04T00:00Z, 2026-05-01T00:00Z)`, full agent-visible reports.
- Shadow: `[2026-05-01T00:00Z, 2026-08-01T00:00Z)`, every trial; human-only
  report/score under `/srv/ethpredict-grader/`.
- Holdout: `[2026-08-01T00:00Z, 2026-10-16T00:00Z)`, sealed; only a human can
  trigger one ledgered evaluation for a promotion. No implementation run uses it.

These literal half-open endpoints are preserved. The lake contains observations
on October 16 after midnight, but the specified holdout endpoint excludes that
partial final day. No implicit extension to October 17 is allowed.

Use six contiguous chronological groups within dev: the first supplies initial
training, the next five are evaluation blocks. Their training population grows
from the start through the preceding group. Exact inclusive dependency spans
use the declared lookback plus maximum horizon as the purge width. Vendor
`purge_training_spans`/`embargo_width` logic and use a sequence-plus-horizon
embargo; persist the removed row IDs and prove no dependency overlap. Embargo
excludes future-side rows; pure walk-forward training has none, which must be
visible in its audit rather than claimed as an extra gap. Separately exercise
future-side embargo in tests. No training episode may escape the retained span.
Evaluation blocks themselves are contiguous and do not overlap; context-only
history is distinguished from scored decisions. All declared markets use the
same eligible clock, without silently dropping a difficult market or block.

### Pretrained-model contamination

The only v1 foundation checkpoint is
`google/timesfm-2.5-200m-pytorch`, Apache-2.0. The upstream
[release announcement](https://github.com/google-research/timesfm#update---sept-15-2025)
dates public release to **2025-09-15**. The
[model API](https://huggingface.co/api/models/google/timesfm-2.5-200m-pytorch)
records repository creation September 2, 2025; creation alone does not prove a
public weight-release date. Use the later announcement conservatively, and pin
the actual checkpoint revision/hash (observed revision
`1d952420fba87f3c6dee4f240de0f1a0fbc790e3`) in the model registry before grading.
Report any pre-release evidence as **possibly in pretraining**. Release timing
does not prove that every training source is known or that no contamination exists.

The specified August 2026 holdout begins after the 2.5 release. Eligibility still
requires verified model identity, untouched data and the human ledger gate.
Unknown release dates make pretrained holdout claims ineligible. The remedy is
forward-collected history beyond October 16, 2026, obtained by one read-only
collector at a time with at least 0.5s pacing; stop/back off on Human Verification.
TimesFM 3.0 is not added here. Its downloaded weights use a separate
[non-commercial license](https://github.com/google-research/timesfm#license-notice-for-pretrained-weights),
and its release is after the specified holdout start; adding it requires human
approval and a new eligibility decision, potentially forward data. Record license,
release evidence and checkpoint identity for every pretrained model a grader sees.

## Physical sealing and process isolation

P2 supplies `scripts/seal_grader_tiers.sh` with a read-only plan and explicit
operator execution mode. A human creates the dedicated `ethgrader` Unix user
and runs the reviewed move into `/srv/ethpredict-grader/`, owner `ethgrader`,
mode `0700`. The agent must never run sudo, create that user or move data out of
`data/lake/`. Print exact `! <command>` invocations at the handoff. Acceptance
requires a real agent-user `PermissionError`, not a mock permission check.

The sealing plan must cover alternative readable copies: funding recovery
backups/receipts, old CSVs and derived artifacts that overlap hidden periods,
other worktrees, caches and staged/exported inputs. Merely moving the monthly
Parquet files leaves a bypass. Preserve historical identities and ledgers; move
or restrict access under human control rather than rewriting historical bytes.
The original full manifests contain private partition metadata; replace public
manifests with dev-only manifests and keep originals in sealed storage. Maintain
an idempotent, verified journal so interruption cannot silently lose a partition.

Add `.claude/settings.json` deny rules for Edit/Write `grader/**`, Read
`/srv/ethpredict-grader/**`, and Read `autoresearch/shadow*`. They are defense
in depth. Unix permissions and candidate sandboxing supply the actual boundary.
Privileged grading runs as `sudo -u ethgrader`, but **candidate code must not
inherit that user's unrestricted access**. The trusted process creates only
causal slices, launches the child with restricted filesystem access and no
network/credentials, captures attempted opens, and reads only validated outputs.
An inaccessible raw private path must both deny access and fail the isolation
check even if the strategy catches the exception. Filesystem sampling alone
cannot prove absence of transient opens. Prefer enforced access restrictions
plus an attempted-open audit. Test the real Linux mechanism before acceptance.

The agent never gets hidden raw files, reports, candidate logs, predictions,
checkpoints or Trackio records. Candidate runtime may observe only the
grader-supplied causal history needed to produce a decision; the hidden-tier
worker and every output stay inside the private service. This is a necessary
interpretation of prediction on hidden data, not permission to expose it to the
loop agent. The human approved this private worker boundary on 2026-10-27. No hidden artifacts are copied into the visible trial bundle.

## Strategy and accounting interface

`Strategy` declares `name`, all `markets`, `resolution`, `max_horizon_bars`,
`lookback_bars`, and `budget_tier` (`fast` or `long`). `fit(train, ctx)` sees only
the permitted training span and a context containing the grader-selected seed,
device, wall deadline, scratch directory and a read-only training ReplayEnv
factory. `predict(history, decision_times)` returns exactly `timestamp`,
`market`, `target_position`. All randomness derives from `ctx.seed`.
Checkpoints/artifacts belong under scratch. Score every declared market.

Positions must be finite, in `[-1,1]`, on the complete decision grid, with no
duplicate/missing keys; per-market absolute cap is 1, gross cap is the number
of declared markets, at fixed notional. Fractional targets are allowed but
variable sizing research remains #19. A decision after bar t closes executes
at bar t+1's open plus configured entry delay. Hold the executed position
until the next execution; delayed targets queue without retroactive changes.
Do not execute at the just-observed close or use a future range as a causal quote.

`ReplayEnv(data, markets, resolution, cost_model, window, seed)` exposes
`reset() -> obs` and `step(action) -> obs, reward, terminated, info`. Observations
contain completed bars through decision time, a bounded lookback, current
positions and funding accrued so far. Reset samples reproducible training-only
windows. Out-of-span access raises. No test, shadow or holdout episode is
available to `fit`. `info` carries gross return, turnover, fee, spread, slippage,
funding, delay and fill components plus execution timestamps.

Net reward in bps is executed position times return in bps, minus turnover cost
and signed settled funding. Charge transaction costs on absolute position
changes, with proper one-way/round-trip conversion from the vendored cost
quotes; do not double-charge a round trip per step. Funding uses the executed
position at the payment clock, with the original entry-exclusive/exit-inclusive
contract. Fixed notional defines gross-notional normalization, not compounded
strategy wealth. Specify liquidation at block ends so final costs cannot escape.

`reward.torch_step_pnl(positions, returns, costs, funding)` uses differentiable
tensor arithmetic and the same signed units as numpy/final scoring. Test
random float64 paths to `1e-9`, including fractional targets, flips, delay,
funding payments and costs, and check gradients. A vectorized tensor batch mode
runs N training windows on GPU without a Python loop over environments. PPO/GRPO
can use it later without a new accounting implementation.

## Rubric and reports

Every attempted trial, including schema errors, missing input/funding, crashes
and timeouts, yields a validated `grade_report.json`, score 0 on failure and
all available failure reasons. Integrity failures cannot be masked by economic
success. No unknown/NaN component may be silently omitted or imputed.

Tier 0 requires matching frozen hashes, enforced/audited path isolation,
causality at at least 20 random cutoffs **per block**, and a same-seed repeat of
one block's fit plus predict. Compare positions through each cutoff to `1e-9`.
Budget all blocks' fit/predict together: fast 600s, long 3600s, peak VRAM at most
22 GB, RSS at most 48 GB. Kill the entire child process group on timeout. Monitor
GPU memory independently of candidate self-reporting. A failed same-seed check
gets status `nondeterministic`; a killed budget overrun gets `timeout`.

Tier 1 requires all of: at least 25 trades per block and 150 overall; aggregate
dev net PnL strictly positive at 1x costs; at least three of five positive
blocks; aggregate net strictly positive at 2x costs; net PnL above the best
baseline on the identical blocks; and, for declared engineered features,
purged permutation evidence of at least one family beating noise controls.
A trade is an entry from flat or a sign flip, not every position adjustment.
Missing importance evidence fails that gate. Preserve the existing purged
importance/noise methodology rather than accepting a candidate-authored claim.
The [numeric input protocol](grader_feature_inputs.md) verifies that exposed
inputs reproduce the actual policy, audits their causal prefixes, and uses
fresh input-only workers for the unchanged vendored permutation/noise tests.

Tier 2 maps each component to `[0,1]`, clips linear maps, and averages the
weighted score over two grader-selected seeds:

- Mean daily-PnL annualized block Sharpe: 0 to 3, weight 0.30.
- Worst block Sharpe: -1 to 2, weight 0.20.
- Deflated Sharpe probability: direct, weight 0.20; trial count includes crashes.
- Cost robustness `net(2x)/net(1x)`: clipped, weight 0.15.
- Maximum drawdown / total gross notional: 0% maps to 1, 25% to 0, weight 0.10.
- Net PnL per unit turnover in bps: 0 to 10, weight 0.05.

Daily PnL uses the 24/7 UTC clock, with explicit observations of flat days and a
365.25-day annualization basis, never an implicit 252-day equity convention.
Ledger-wide trial accounting feeds the vendored DSR computation; retain
sufficient statistics/daily paths to recompute both incumbent and challenger
at the same current count. Reports identify data/strategy/grader/model hashes,
seeds, all gate reasons, block metrics, costs/stress, resources, contamination,
and artifact identities. Shadow versions remain private.

Tier 3 `judge.md` evaluates the actual diff and code for a pre-run hypothesis,
simplicity, absence of unexplained threshold fiddling, and absence of hard-coded
dates/market/regime special cases (four 0–1 values). It cannot rescue a failing
gate or independently authorize a keep. The README hypothesis is captured before
execution, not accepted as a retrospective narrative.

### Calibration, baselines and immutable freeze

Run flat, buy-and-hold, momentum (both 1h and 24h variants), zero-shot TimesFM
2.5 (median forecast sign), and the existing LSTM/meta-label recipe through the
same runner/clock/costs. Commit per-block metrics, resources and runtimes to
`context/grader_v1_baselines.md`. Calibration uses only these baselines before
freeze, never candidate strategy outcomes. Freeze and push `grader-v1` only
after the full suite and red-team gates pass. No anchors are chosen in P0.

The human resolved a specification conflict on 2026-10-27: a baseline cannot
strictly beat the best baseline, and flat/buy-and-hold fail the trade gate.
Thus all gated baseline scores may be 0 regardless of anchors, incompatible
with making the best final score 0.1–0.3. The approved interpretation is to calibrate and show pre-gate component scores
in the requested range, while preserving every gate and final rejected scores
of 0. Do not weaken gates to hit the range.

## Keep/discard driver and audit trail

The driver takes an exclusive GPU lock, verifies the pinned grader worktree,
and rejects commits touching anything outside strategy and permitted `src/`
features/models/training/policy code. It does not accept agent edits to the
driver, grader, data, dependencies or test-use ledger. Log the attempted commit
before running; crashes/timeouts/forbidden commits count as trials. Keep an
append-only `ledger.jsonl`, TSV results and dev-only Trackio records in project
`ethpredict-autoresearch` (existing research remains `ethpredict`). Capture
commit, hypothesis, grade report, block positions, checkpoints and Trackio ID
in each visible dev artifact bundle.

Recompute the incumbent at the new ledger trial count. Use 1,000 stationary
block-bootstrap resamples over daily PnL to estimate score standard error δ.
Keep only when `score_new > score_incumbent + δ`; otherwise restore the
incumbent in the driver's own disposable checkout. Preserve candidate commits
and receipts; never reset a shared/dirty checkout. Qualitative tie-breaks may
rank tied descriptions for humans, but cannot override that numeric keep rule.

After every trial, the trusted private service also grades shadow as ethgrader.
It stores the shadow result and rolling incumbent dev-minus-shadow check
privately. Every ten trials compare the gap; if it grew on three consecutive
checks, emit only `autoresearch/HUMAN_ALERT` and pause/notify the human. No gap
value or shadow score enters logs, TSV, Trackio, stdout/stderr, candidate scratch
or the visible ledger. Missing private grading fails closed rather than being
treated as a successful experiment. Privileged service authorization must be
narrow; it cannot be generic sudo of candidate-provided Python.

`program.md` describes setup on `autoresearch/<tag>`, the editable boundary,
`grep '^score:' run.log`, TSV/Trackio logging, crash accounting, keep/reset,
and continuous operation until HUMAN_ALERT. The human starts the open-ended
loop; implementation validates only a three-iteration dry run with one deliberate
crash and one discard, all three ledgered, with no readable shadow leakage.
Seed agenda: #30–#35, #37–#39 and the TimesFM track. #39 depends on out-of-scope
#19, so it remains a later agenda item, not an implemented sizing study.

## Red-team and phase acceptance

P3 must exercise the actual subprocess and scorer, not only isolated predicate
helpers, for each required adversary:

1. Next-close leakage fails causality.
2. Attempted private file open is denied and fails tier isolation.
3. Always-flat positions score 0 for trade count.
4. Three large winning trades fail trade count.
5. Wall-clock seeding is flagged nondeterministic.
6. Candidate grader changes do not affect scoring and are rejected by driver.
7. An over-budget sleeper is killed, status timeout, score 0.
8. A path profitable only at 1x fails the 2x gate.
9. Five declared markets with only one returned fail schema.
10. A missing scheduled funding payment fails closed.

P1 demonstrates hash/schema/duplicate/availability/window tests, whole-overlap
CSV parity and multi-market provenance. P2 demonstrates purge/embargo/no-overlap
tests plus a **human-executed** seal and real PermissionError. P3 also proves
numpy/torch/scoring parity, baseline receipts, pushed freeze tag and full suite.
P4 proves the three-attempt dry run and hidden-output isolation. P5 supplies a
frozen-backbone causal-embedding policy head and full bf16/AdamW two-stage
fine-tuning (quantile future log-return loss, then net-PnL with an L2/KL anchor).
Use long budget for full fine-tuning; record measured peak VRAM and steps/s.
Optional GRPO is only for nondifferentiable execution; it may be skipped.
These are untuned scaffolds: Tier 0 must pass, Tier 1 need not. Commit honest
grade/resource reports in `context/autoresearch_seed_strategies.md`.

## Open decisions and operational gates

1. P2 isolation is verified on 2026-10-28: the human completed sealing, the
   actual agent received 259 PermissionErrors, and the designated dev-only tool
   worker excludes duplicate CSV/Git/artifact sources. The host coordinator
   remains trusted. P3 must still prove private strategy-process isolation and
   P4 must implement the narrow grading broker; hidden grading is not ready.
2. Before P3 freeze, implement the approved pre-gate calibration distinction,
   and choose/freeze execution-cost configuration and bootstrap expected block
   length using declared methodology, without strategy tuning.
3. Test the Linux isolation mechanism on this machine before choosing it. A
   shadow worker cannot safely execute candidate code with broad ethgrader
   privileges. Dependency additions or upgrades require human approval.
4. The literal prohibition on a strategy seeing hidden data is interpreted as
   no direct private-lake access and no outputs to the agent: causal input slices
   are necessarily consumed by the sandboxed worker. This service boundary was
   explicitly approved by the human on 2026-10-27; prove its isolation before
   enabling hidden grading.
5. LoRA vs frozen/full fine-tuning is a future loop agenda comparison. Do not add
   PEFT, upgrade TimesFM or install RL libraries without human approval.
6. After P0–P5 are each merged, update PROJECT_STATUS.md, TASKS.md and
   experiment-readiness.md and run the full suite on main. Demonstrate
   `uv run python autoresearch/run_experiment.py --dry-run` from a fresh clone
   plus dev lake and the correctly provisioned private service. Do not start
   the open-ended loop or evaluate holdout to satisfy construction acceptance.

The fixed-evaluator/editable-candidate pattern is inspired by
[karpathy/autoresearch](https://github.com/karpathy/autoresearch). Its metric is
not evidence of trading profitability; this harness adds causal accounting,
physical tier isolation, economic rejection and complete trial accounting.
