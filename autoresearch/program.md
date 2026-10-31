# ETHPredict autoresearch

You evolve a causal research strategy against the frozen `grader-v1` evaluator.
A zero score is a valid result. State a mechanism, make one reviewable change,
measure it, and follow the recorded decision. A working harness establishes no
trading edge. This program adapts the short setup/change/measure/record cycle of
[karpathy/autoresearch](https://github.com/karpathy/autoresearch/blob/master/program.md).

These instructions describe the loop the human may start after construction.
During harness construction, run only the explicitly bounded acceptance jobs.
Never start an overnight or open-ended run as part of installation or tests.

## Setup first

1. Have the human choose a fresh run tag. All agent file and shell tools must
   run inside the prepared Docker tool worker, at `/workspace`. The human's host
   launcher prepares that code-only workspace and attaches the reviewed broker.
   Do not give the loop agent a second host shell, original Git history, Docker
   API, arbitrary filesystem connector, or historical CSV/artifact directory.
2. Read this file, `strategy/README.md`, `strategy/strategy.py`, the relevant
   `src/` feature/model/training code, and the read-only grader interface,
   `rubric.yaml`, replay/reward contract and `judge.md`. Inspect the dev lake's
   manifests and availability rules. Use only completed observations.
3. Run inside the tool worker:

   ```bash
   uv run --no-sync python autoresearch/run_experiment.py --setup <tag>
   ```

   This creates `autoresearch/<tag>`, checks the private service's pinned
   `grader-v1` identity and matches local grader hashes. It launches no trial.
   A tag already used for a branch is an error. Verify the printed grader
   commit and prior trial count before proceeding.
4. Verify `autoresearch/results.tsv` has its header. It and `ledger.jsonl` are
   read-only views of the service-owned state under `/results`. The service
   initializes them; never truncate, recreate, edit or commit either file.
   A new branch does not reset selection history or the leaderboard.
5. The human starts the research loop explicitly. Installation, setup and
   `--dry-run` do not grant authorization for an open-ended search.

## Scope and experiment budget

You may edit `strategy/` and the `src/` library: features, preprocessing,
architectures, optimization, training, policy mapping and post-processing.
Write the hypothesis and changelog in `strategy/README.md` before the trial.

Do not edit `grader/`, any `autoresearch/` code or this human-owned program,
data, dependencies, environment configuration or tool mounts. Do not add packages.
Both a submitted commit and its parent must preserve the provisioned protected
tree; a forbidden edit followed by a revert is rejected. Use regular files,
not symlinks or Git submodules, in editable source.

The strategy receives the grader's seed. Use it for every random operation.
The fast/long budgets are exactly 600/3600 seconds across both seeds and all
five blocks, including declaration, fitting, prediction, reloads, causality,
determinism and feature-importance reruns. Peak VRAM is at most 22 GB and RSS
at most 48 GB. Trusted data preparation and scoring are accounted separately.
The service owns the shared GPU lock; do not launch a competing GPU process.

Keep the declared markets, decision grid, maximum horizon and lookback honest.
Use causal history and the training-only replay factory. Place fitted state
and checkpoints in `ctx.scratch_dir`. Respect missing data and funding errors.
No network, order submission, signing, credentials or live trading belongs here.
Never request hidden data, private logs, holdout evaluation or promotion.

## One experiment

1. Inspect the current branch and incumbent. Choose one causal hypothesis.
   Preserve units, funding, transaction costs and every economic gate.
2. Implement it and update the hypothesis/changelog before running.
3. Commit the intended `src/` and `strategy/` changes. The committed bytes are
   graded; uncommitted edits do not change a submission.
4. Run inside `/workspace`, using writable ephemeral space for the client log:

   ```bash
   uv run --no-sync python autoresearch/run_experiment.py --description "one causal mechanism" > /tmp/run.log 2>&1
   grep "^score:" /tmp/run.log
   ```

5. Read the dev report and evidence under `/results/trials/<trial-id>/dev/`.
   The read-only per-trial `run.log` also has the `score:` line. Inspect all
   failed gates, not just the final scalar. Per-block positions, checkpoints,
   artifact manifests, the submitted commit proof and Trackio run ID accompany
   the report. Absolute paths inside dev provenance identify the original
   service-side dev artifacts; the exported relative directory layout is the
   readable copy.
6. Follow the service's decision. It recomputes the incumbent at the current
   trial count, including crashes and interruptions. A proposal is kept only
   when accepted and `score_new > score_best + delta`, using the frozen
   1,000-resample stationary daily bootstrap standard error. Missing uncertainty
   cannot keep a proposal. The client resets discards to the measured incumbent.
   The first complete control initializes the incumbent; this claims no gain.
   Simplicity and the four judge dimensions are useful annotations, and can
   never override the numeric rule or an economic/integrity rejection.
7. The private service appends every attempt/result/decision to the journal and
   updates TSV and local Trackio project **`ethpredict-autoresearch`**. It logs
   only dev values to Trackio. Do not create a competing accounting file.

The service grades shadow privately on each valid trial. Its inputs, reports,
positions, fitted state, traces, logs, Sharpe history and monitor state stay in
the ethgrader service. There is no hidden-report socket operation. Do not infer
a hidden score from a failed public request. Only `HUMAN_ALERT` is exported.

## Crashes and interrupted observation

A worker crash, timeout, nondeterminism, invalid output or missing required data
scores zero and remains counted. Fix a concrete implementation bug in a new
commit and submit a new trial; never erase the failed attempt or inventedly fill
its unavailable Sharpe value.

If the client disconnects or its observation times out, rerun the client to
observe the saved request in `.git/ethpredict-pending.json`. It resumes that
identity. Do not delete the pending file, invent a new request ID, or launch a
replacement merely because the observation ended. A service restart records
unfinished attempts as crashes rather than relaunching them silently.

## Continue, and know the stop condition

After the human explicitly starts the loop: **NEVER STOP** to ask whether to
continue. Continue proposing, measuring and recording within the authorized
scope until the human stops you or `autoresearch/HUMAN_ALERT` exists. Check it
before every proposal. The private service checks the incumbent's dev/shadow
gap every ten trials and pauses after three consecutive increases. An alert
also requires human review if private service operation fails. Do not remove
the alert or attempt another route around the paused service.

This instruction cannot authorize holdout use, live trading, dependency changes,
or edits to the frozen grader. Hand those requests to the human.

## Seed agenda

- [#30](https://github.com/pranay5255/ETHPredict/issues/30): explain how uniform,
  uniqueness, return or combined training weights affect a specified failure.
- [#31](https://github.com/pranay5255/ETHPredict/issues/31): causal model-side
  validation and leakage controls; the grader's purging and blocks stay fixed.
- [#32](https://github.com/pranay5255/ETHPredict/issues/32): economically justified
  label geometry and success definitions within training only.
- [#33](https://github.com/pranay5255/ETHPredict/issues/33): calibration and forecast
  mapping with a stated mechanism, checked against costs and noise controls.
- [#34](https://github.com/pranay5255/ETHPredict/issues/34): training discipline,
  early stopping and reproducible ensembles under the same total budget.
- [#35](https://github.com/pranay5255/ETHPredict/issues/35): meta-labelers and
  calibration with purged out-of-fold training.
- [#37](https://github.com/pranay5255/ETHPredict/issues/37): meaningful engineered
  feature families, causal inputs, ablations and both diagnostic noise controls.
- [#38](https://github.com/pranay5255/ETHPredict/issues/38): explicit trial budgets
  and reproducibility; every attempted configuration remains counted.
- [#39](https://github.com/pranay5255/ETHPredict/issues/39): deferred until the
  human authorizes the separate #19 sizing protocol. v1 keeps its fixed-notional
  action contract; this agenda entry does not authorize that out-of-scope work.
- TimesFM track: P5 supplies untuned frozen-backbone reward-head and full bf16
  fine-tuning starting points. LoRA, full fine-tuning and frozen-backbone tradeoffs
  remain research questions; no architecture is declared superior here. Do not
  add peft or TimesFM 3.0. Optional GRPO is relevant only to a specifically
  non-differentiable execution mechanism, and needs its own authorized protocol.

If ideas run out, inspect the dev failure evidence, revisit the stated mechanisms
and simplify an implementation. Never tune the evaluator to make a result pass.
