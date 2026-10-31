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

The [runtime review receipt](autoresearch_p4_runtime_receipt.json) records the
completed 17.33 GB staging bundle, 99,654 files, internal symlinks, exact frozen
commit and reviewed manifest hash. Its standalone interpreter imported the
existing NumPy/pandas/Torch/TimesFM environment successfully. A selected-source
archive was checked against all 51 frozen source files byte for byte. Git emits
a warning for the omitted unrelated `.claude/settings.json` blob; the selected
archive is complete and identical. No unrelated historical data blob is copied.

The first full-suite run exposed an overlong temporary Unix-socket name; the
fixture now uses a Linux abstract socket and its real peer-credential test
passes with the durable artifact base. The interrupted run recorded one failure
and 232 passes; it is not a green result. A subsequent collection attempt found
an intentionally inaccessible generated red-team symlink under `artifacts/`.
The full command now includes `--ignore=artifacts`, excluding generated runtime
packages and fixtures from discovery. Collection proves **562 repository tests
plus all 62 pinned SDK tests**, 624 total, with no other test roots. The full run with real P2 permission/Docker checks and pinned CUDA model passed:

```text
623 passed, 1 skipped, 23 warnings in 729.24s (0:12:09)
```

A final startup-recovery audit then added Trackio delivery for interrupted
attempts and the documented generic human-review pause. The recovery, journal,
keeper and resumable-client checks passed `23 passed in 5.02s`. The final runtime
was rebuilt from `03005779079a557a2de95fbfa13892b5ff0fe13f`; all copied control code
matches that source, and its complete manifest was verified again. The prior
uninstalled construction bundle was removed; its historical receipt remains in
Git. The final full-suite repeat after this isolated recovery change passed:

```text
623 passed, 1 skipped, 23 warnings in 810.37s (0:13:30)
```

The fresh sparse clone materialized only code/project metadata, linked the
verified public lake and prepared a fresh code-only tool workspace. Its real
Docker probe again verified ten denied host paths, two absent historical Git
identities, zero capabilities, seccomp/no-new-privileges, read-only protected
code and loopback-only networking. No private experiment was launched. The [final fresh-clone preflight](autoresearch_p4_fresh_clone_preflight.json)
records the refreshed workspace: all 117 code files and 280 dev files are
present, and its source proof matches the reviewed service protected tree.
The actual fresh-clone dry run awaits installation.

## Current reviewed human command

The final bundle is identified by
`d66a32b3b7dc45d81c77c409426a20f69a640ca56387b1e2482fc8e7658b6136`.
The first human installation command found an existing runtime directory and
stopped at its protective guard. Read-only checks confirmed no installer was
still running, and the configuration, socket, GPU lock and public output path
were absent. A dedicated human-only completion helper now validates every
existing runtime file before mutation. It fills missing entries, completes only
byte-for-byte matching truncated prefixes and restores reviewed copy metadata.
Unexpected files, changed complete files, hard links, escaping links and active
deployments are rejected. Existing sealed data and trial histories are untouched.

Its ten focused tests passed, including real filesystem fixtures for partial
copies, conflicts, links and empty Git directories. The helper also validated all
99,654 entries of the actual staged payload without changing its reviewed hash.
It lives under `scripts/`, outside the immutable agent/control tree, so this
operational recovery requires no new payload or model copy. Run from a normal
terminal, as the human:

```bash
sudo /usr/bin/python3 /home/pranay5255/.codex/worktrees/autoresearch-p4/ETHPredict/scripts/complete_autoresearch_install.py --bundle /home/pranay5255/.codex/worktrees/autoresearch-p4/ETHPredict/artifacts/p4-provision-0300577 --manifest-sha256 d66a32b3b7dc45d81c77c409426a20f69a640ca56387b1e2482fc8e7658b6136
```

Only the human performs this operation: section 9 says to stop and ask for
anything that needs sudo. The expected summary has `private_runtime_installed`
and `socket_enabled` both true. Share only that summary or an installation
error; do not share any private financial report. Enabling the socket launches
no research experiment. The real three-trial jobs remain bounded acceptance
work after installation and completion of the current GPU test.
