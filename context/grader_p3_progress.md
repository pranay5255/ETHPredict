# Grader P3 progress — 2026-10-28

This is the earlier worker/accounting checkpoint. Current integration progress,
the 519-test suite, all ten red teams and remaining acceptance work are recorded
in [the integration verification](grader_p3_integration_verification.md).

P2 merged at `86a9df0` in PR #62; issue #56 is closed. The human confirmed the
sealing summary, and the committed P2 receipt already contains that confirmation.
No sealing rerun or private-data access is required from the agent.

P3 is **in progress**, on `autoresearch/p3-grader-env`, issue #57. This checkpoint
is `004bf42` (with `6223bd1`, `81fb058`, `20e9b06`). There is no `grader-v1` tag
and P3 acceptance is not claimed.

## Verified checkpoint

- Standalone vendored cost/statistics/split/importance/selection/economic-gate
  primitives; source and vendored hashes recorded; no editable-library imports
  in the scoring package.
- Shared next-open accounting for NumPy scoring, scalar replay, Torch reward and
  GPU replay batches. Fixtures cover delay, partial/sample fills, 1x/2x costs,
  funding payment boundaries, terminal liquidation, gradients and `1e-9` parity.
- Actual Landlock/seccomp worker execution with private file audit; every
  prediction reloads the same read-only fitted state in a fresh child from a
  clean server. No future-data memory is forked from the scoring controller.
- Actual prefix files for at least 20 cutoffs, same-seed repeat fit/predict,
  missing/extra-market/schema rejection, caught denied opens, forbidden commit
  paths, timeout cleanup, tracer blinding detection, and server-death cleanup.
- Independent NVML telemetry and a trusted physical VRAM reservation; actual
  CUDA gradients and bounded training ReplayEnv access run inside the worker.
- Strict rubric parser and pre-gate components; the original trade count,
  positive-net, cross-block, 2x-cost, best-baseline and engineered-feature gates
  are preserved. Calibration remains `pending_baseline_runs`.

Full suite at this checkpoint:

```text
484 passed, 1 skipped, 23 warnings in 168.84s (0:02:48)
```

Executed from this worktree, with the existing primary environment and the real
P2 probes explicitly enabled:

```bash
env -u ETHPREDICT_SOURCE_REPO \
  UV_PROJECT_ENVIRONMENT=/home/pranay5255/Documents/ethpredict/ETHPredict/.venv \
  ETHPREDICT_AGENT_WORKSPACE=/tmp/ethpredict-agent-p2-isolated \
  ETHPREDICT_SEAL_PLAN=/tmp/ethpredict-p2-seal-plan.json \
  ETHPREDICT_REVIEWED_PLAN=/tmp/ethpredict-p2-seal-plan.json \
  OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  uv run --no-sync pytest -q
```

The skip is P1's historical CSV/lake overlap check, whose source interval now
crosses sealed months. Its 26,496-row exact pre-sealing parity receipt remains
committed. No private partition was read to repeat it. The 23 warnings match
existing SDK/statistical fixture warnings. No packages were added or upgraded;
the pinned existing SDK submodule was initialized for the full suite.

The locally cached TimesFM 2.5 artifact was inspected without downloading:
revision `1d952420fba87f3c6dee4f240de0f1a0fbc790e3`, `model.safetensors`
925,181,104 bytes, SHA-256
`2f776efe6245e42b24bc4153ffdf61810140210e4bd3b01fb21f7aa779ab6ce8`;
`config.json` SHA-256
`cd3315b760d5cc7e278d7afdf41b897031ced888fc6c115bd9b3ac0ea2c47408`.
This is artifact verification, not a baseline grade or contamination clearance.

## Work remaining at the earlier checkpoint

1. Complete grade/report orchestration and CLI, including model registration,
   source identity, trial accounting, bootstrap uncertainty, contamination labels
   and trusted engineered-feature/noise evaluation.
2. Run all baseline families and both momentum horizons through the actual
   grader on identical declared-market blocks. Commit per-block numbers and
   measured runtimes to `grader_v1_baselines.md`.
3. Demonstrate all ten red-team fixtures through complete grades, in addition to
   the tested component/subprocess paths above.
4. Calibrate only pre-gate component maps using the approved baseline
   interpretation. Preserve rejected final scores at zero and every gate.
5. Run the final full suite, freeze/push `grader-v1`, and finish P3's PR before
   P4 begins. P4's private service and P5's untuned seeds remain future phases.

No strategy search, hidden evaluation, consumed-test rerun, holdout evaluation or
strategy-edge claim occurred at this checkpoint.
