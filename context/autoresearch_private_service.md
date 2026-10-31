# P4 private service deployment

P3 is merged in PR #63 at `7fdc2fc`; `grader-v1` pins `58228cb`. P4 is on
`autoresearch/p4-loop` and is not accepted until the actual private three-trial
run completes. P5 has not started. No grader file changes are part of P4.

## Boundary and accounting

All loop-agent tools execute inside the existing code-only Docker worker.
It mounts a narrow Unix socket and a read-only dev export. It has no GPU,
external network, host shell, Docker API, original Git history or private lake.
The client sends bounded raw Git object proofs. The service checks their hashes,
the complete editable snapshot and both commits' protected root entries without
opening the agent checkout or running its Git configuration.

The service runs as ethgrader. A separate frozen code-only Git checkout retains
the exact published tag/commit and required source bytes. Directory-tree
metadata is retained for Git's archive implementation; unrelated data blobs and
historical parent commits are absent. The Python base and existing venv are
copied into a standalone runtime, with no new package installation or upgrade.
The staged interpreter is tested before its installed prefix is recorded.

Dev and shadow execute sequentially through the frozen production CLI, under a
shared external GPU lock. The shadow grader can consume only causal private
slices in its isolated worker, as approved by the human. Every private input,
log, fitted state, trace, position and report remains inside the service. There
is no holdout or hidden-report RPC operation. The private root and original P2
sealing stay unchanged. The service receives no Docker group membership and
needs no access through the human's home.

The authoritative dev event journal is append-only and owned by the private
service. Its public mirror and TSV are read-only. A separate private shadow
journal keeps its own Sharpe series at the same total attempt count. The keeper
recomputes the incumbent with the current count before comparing; only an
accepted score strictly above incumbent plus bootstrap standard error can keep.
Initialization establishes a measured control even when its economic score is
zero. Interrupted attempts remain counted and are not silently rerun.

Only manifest-verified dev artifacts are copied into public bundles; there is
no recursive copy of a trial's private parent. Local Trackio project
`ethpredict-autoresearch` receives an explicit dev metric whitelist with CPU/GPU
automatic logging disabled. The private monitor compares the incumbent every
ten trials, using an initial reference and three consecutive increases before
emitting `HUMAN_ALERT`. Service failures also pause for human review. The signal
contains no score, gap, private status or diagnostic values.

## Prepare and review as the ordinary construction user

Use a clean, committed P4 checkout and a fresh isolated workspace. The public
lake must match the reviewed P2 metadata. Example paths are deliberately explicit:

```bash
python -m scripts.agent_sandbox prepare --repo <P4-checkout> \
  --workspace <fresh-agent-workspace> --dev-lake <sealed-public-dev-lake> \
  --plan <reviewed-seal-plan.json> --venv <existing-.venv> \
  --image sha256:9c2ff16f962f9fb5b4625958ca0edb1e8e1e99291e178df4311c661af0e06b57
python -m autoresearch.provision stage --bundle <new-bundle-directory> \
  --workspace <fresh-agent-workspace> --venv <existing-.venv> \
  --model-path <pinned-timesfm-2.5-snapshot>
```

Staging reads only the public dev lake, public checkpoint and trusted code and
runtime. It copies no secrets, sealed prices, old CSVs or existing hidden
artifacts. The printed manifest SHA-256 identifies every staged file and allowed
internal symlink. Review `autoresearch/provision.py`, `service.py`, the generated
configuration and both unit files before installation. The runtime's source
commit is retained even if later commits only add documentation.

## Human-only installation

Sections 4.3 and 9 of the specification reserve sudo operations for the human.
The construction agent must not execute this step. After staging, use its exact
bundle path and manifest digest from the review receipt:

```bash
sudo /usr/bin/python3 -m autoresearch.provision install \
  --bundle <reviewed-bundle-directory> --manifest-sha256 <printed-sha256>
```

Run it from the reviewed P4 checkout in a normal terminal; a client supporting
shell escapes may prefix the command with `! `. It verifies the reviewed bytes,
copies a versioned runtime, verifies the installed copy, creates service-owned
state/export paths and a protected persistent GPU lock, and enables the socket.
It creates no user and moves no lake files. Existing runtime versions and trial
journals are not deleted. Service stdout/stderr go to the private service log,
not the readable journal. systemd runs the worker as ethgrader, hides home
directories, makes runtime/data read-only and kills the whole child group on
shutdown. Only state, dev exports and the GPU lock are writable.

## Connect, then run exactly three trials

After the human reports successful installation, the ordinary user connects
the prepared workspace and rechecks its real Docker boundary:

```bash
python -m scripts.agent_sandbox attach-broker --workspace <fresh-agent-workspace>
python -m scripts.agent_sandbox probe --workspace <fresh-agent-workspace>
python -m scripts.agent_sandbox run --workspace <fresh-agent-workspace> -- \
  /bin/bash -c 'uv run --no-sync python autoresearch/run_experiment.py --dry-run'
```

The dry run submits a flat control, a deliberate fit exception and an unchanged
flat proposal. It records exactly three additional authoritative attempts,
including the crash and discard. Request identities and progress are persisted;
if observation ends, resume the same command instead of replacing the job.
A completed dry-run command does not submit another group when repeated. A
fresh-clone acceptance check uses a new prepared workspace, while preserving
all prior service trial counts. The construction task does not start the
open-ended research loop.

The agent sees only `/results`, the dev-only TSV/ledger views and generic alert.
The human may inspect the shadow journal/reports directly inside
`/srv/ethpredict-grader/autoresearch/`; never paste their values into the agent
session. Private fixture canaries and real OS denials complement, rather than
replace, this deployment acceptance test.

## Verification checkpoint

Focused protocol, decision/journal, service transaction, client interruption,
provisioning and existing sandbox checks: **49 passed, 1 skipped in 7.67s**.
The skipped test is the optional real Docker integration without its explicit
workspace environment variable. All seven controller tests use clearly labelled
synthetic reports; they do not claim a real private evaluation. Local installed
Trackio delivery/readback is exercised in an isolated temporary directory.
After the last provisioning-copy assertion, its four focused tests passed again.
The full P4 suite and real private deployment/dry run remain required.
