# P2 tiers and physical isolation — verified 2026-10-28

The human completed production sealing and confirmed the script printed
`lake_sealed: true` and `private_files_moved: 247` after its built-in private
hash/ownership verification. The unprivileged agent then verified **259 actual
PermissionErrors**. `/srv/ethpredict-grader` is owned by ethgrader UID 997/GID 994,
mode 0700. The reviewed private-source files are absent from the public lake.
The public view contains 275 dev partitions and five dev-only manifests; 150
private partitions and 97 recovery files were moved into protected storage.

The evidence receipt is [grader_p2_isolation_receipt.json](grader_p2_isolation_receipt.json).
No strategy, holdout or previously consumed test evaluation was performed. P0
and P1 are merged; P2 acceptance is verified and PR #62 is ready to merge.
P3–P5 have not started. No grader-v1 freeze or working research loop is claimed.

## Tier and block contract

`grader/tiers.yaml` preserves dev `[2025-06-04,2026-05-01)`, shadow
`[2026-05-01,2026-08-01)` and holdout `[2026-08-01,2026-10-16)` UTC. October 16's
partial final day remains outside the evaluation window but is physically sealed.
Unknown/duplicate configuration fields fail closed. Five contiguous evaluation
groups follow an initial training group; the training window grows. Inclusive
`[anchor-lookback,anchor+horizon]` dependency spans purge lookback+horizon anchors.
The vendored sequence-plus-horizon embargo removes future-side populations;
pure growing-window training has none. Tests prove both cases and no overlap.
`grader/vendor_sources.json` records the byte-identical split module's source hash.

The [Claude permission rules](https://code.claude.com/docs/en/permissions#read-and-edit)
use `Edit(/grader/**)` for edits and writes, `Read(//srv/ethpredict-grader/**)` for
the absolute private root, and `Read(/autoresearch/shadow*)`. These are defense in
depth. The actual Unix permission and isolated-worker probes supply the boundary.

## Actual acceptance evidence

```text
<existing .venv>/bin/python scripts/verify_sealed_tiers.py \
  --plan /tmp/ethpredict-p2-seal-plan.json --output /tmp/ethpredict-p2-denial.json
private_path_denial: passed
permission_errors: 259
agent_uid: 1000
public_lake_private_copies: 0

ETHPREDICT_SEAL_PLAN=/tmp/ethpredict-p2-seal-plan.json <existing .venv>/bin/python \
  -m pytest tests/test_grader_sealing.py::test_actual_agent_user_cannot_open_sealed_paths -q
1 passed in 0.17s
```

The private contents were not opened by the agent. Their hash/ownership check is
attested by the human's successful sealing output. The agent independently
checks public manifests, source absence and OS denial. A missing file alone is
not treated as permission denial. Real assertions run with the real agent UID.

Full suite with the pinned SDK submodule initialized, real Unix permission gate,
reviewed-plan validation and actual Docker integration enabled:

```text
env -u ETHPREDICT_SOURCE_REPO \
UV_PROJECT_ENVIRONMENT=<existing .venv> \
ETHPREDICT_AGENT_WORKSPACE=/tmp/ethpredict-agent-p2-isolated \
ETHPREDICT_SEAL_PLAN=/tmp/ethpredict-p2-seal-plan.json \
ETHPREDICT_REVIEWED_PLAN=/tmp/ethpredict-p2-seal-plan.json \
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 uv run --no-sync pytest -q
428 passed, 1 skipped, 23 warnings in 124.54s
```

The one skip is P1's original **pre-sealing** CSV/lake integration check; its
26,496-row exact-match receipt remains committed from P1. It must not read the
now-sealed window again. The real P2 PermissionError and Docker tests both pass.
Other checks: `bash -n scripts/seal_grader_tiers.sh` and `git diff --check` pass.
Existing warnings are SDK websocket deprecations and constant-series statistics.

Fixtures also cover copy-only planning, pre-move hashes, recovery quarantine,
interruption/resume/idempotence, source preservation, unsafe paths/hard links,
strict stored-plan validation, fixed boundaries, complete private-partition
coverage, duplicate keys, manifest ownership, forged denial plans and CLI gates.
No packages were added/upgraded. No remote Docker image was fetched.

## Historical copies and the isolated agent tool worker

The old source repository contains tracked shadow-period CSVs, raw recovery
receipts, historical Git objects and derived experiment outputs. These remain
historical evidence. They are **not** made private merely by moving the lake.
`grader_readable_copy_inventory.json` preserves the pre-sealing path inventory.
The host coordinator and this construction session remain trusted; the loop
agent must receive only the isolated tool worker described here.

`scripts/agent_sandbox.py` prepares a new code-only workspace and a fresh Git
repository. It copies only declared source/grader/strategy code, project metadata
and, when they exist, the trusted loop program/driver. It copies no CSVs,
artifacts, context receipts, credentials, old Git objects or remotes. The dev lake
is validated against the reviewed public manifest view before every launch and
mounted read-only. There is no data copy or additional production data move.

The local image contains only empty mount-point directories/symlinks. Its layer
hash is verified, and unreviewed images/hooks are rejected. Existing OS libraries,
Python and the project environment are read-only binds. Docker was already
installed; the rootless namespace alternative failed this host's loopback setup,
so the installed Docker backend is used without downloading a base image.
The image ID used for acceptance is
`sha256:9c2ff16f962f9fb5b4625958ca0edb1e8e1e99291e178df4311c661af0e06b57`.

The worker runs as UID/GID 1000, drops all capabilities, sets no-new-privileges,
uses the default seccomp filter and a private PID/network namespace, and exposes
only loopback. No Docker socket, host home, private root or original repository
is mounted. Only `src/`, `strategy/` and the fresh `.git` database are writable;
the remaining workspace and runtime are read-only. Scratch space is ephemeral.
No GPU device is exposed to this P2 tool worker; P3 supplies strategy execution.

The actual probe verified ten forbidden host paths (including old CSVs, Git,
recovery receipts, historical output and credential paths) unavailable; two
historical Git identities absent; no remotes/alternate object store; read-only
protected code; writable candidate paths; loopback-only networking, zero
capabilities, no-new-privileges and seccomp. It imported pandas 3.0.4 and torch
2.11.0+cu128 and read an eligible dev partition. An integration fixture adds a
real, readable host canary outside the mounts and proves the worker cannot open it.

For a new source checkout after human sealing:

```bash
python -m scripts.agent_sandbox build-image
# Use the printed immutable image ID in the next command.
python -m scripts.agent_sandbox prepare --repo <trusted-checkout> \
  --workspace <new-isolated-directory> --dev-lake <sealed-public-lake> \
  --plan <reviewed-plan.json> --venv <existing-.venv> --image sha256:<printed-id>
python -m scripts.agent_sandbox probe --workspace <new-isolated-directory>
python -m scripts.agent_sandbox run --workspace <new-isolated-directory> -- /bin/bash
```

The current tested workspace is `/tmp/ethpredict-agent-p2-isolated`. All loop-agent
file/shell tools must execute inside this worker; do not grant that agent a second
host shell, Docker API, original checkout or arbitrary filesystem connector.
The trusted grading coordinator stays outside and will expose a narrow grading
request/result interface in P4. Running a model or starting the open-ended loop
is outside P2 acceptance and is reserved for the human after P3–P5.

## Human-only provisioning on a fresh machine

The attachment sections 4.3/9 reserve user creation and moving production data
for the human. The agent never runs sudo, creates the account, changes sudoers or
executes the production move. Stop collectors/source readers and review the
script and read-only plan before applying. In a normal terminal omit a leading `!`;
in a client that supports shell escapes, use `! <command>`.

Create ethgrader once if absent:

```bash
sudo useradd --system --user-group --home-dir /srv/ethpredict-grader --shell /usr/sbin/nologin ethgrader
```

Generate/review `--plan` before any move, then have the human apply it:

```bash
sudo env ETHPREDICT_SEAL_PYTHON=<existing-.venv>/bin/python bash scripts/seal_grader_tiers.sh \
  --apply --source <original-public-lake> --approved-plan <reviewed-plan.json>
```

The plan/journal are persisted privately before per-file copy/fsync/verify/rename
and source removal. Directory modes are 0700; files are 0600; all are owned by
ethgrader. Original manifest bytes remain private. Repeat the same command to
resume after interruption; do not regenerate a plan from a partly moved source.
A source/hash/manifest mismatch stops for review. `--verify-private` is available
for independent human root verification. The grader's later private service runs
as ethgrader; this account is never added to the Docker group here.

## Acceptance checklist

- [x] Literal tiers, five blocks, exact purge/embargo and no-overlap tests.
- [x] Tested read-only plan and byte-preserving, resumable seal script.
- [x] Effective Claude deny rules.
- [x] Human created ethgrader and completed the production seal/private verification.
- [x] Actual agent UID receives PermissionError at all 259 planned private paths.
- [x] Designated loop-agent tool worker excludes legacy files/history and passes real isolation probes.
- [x] Full suite passes with real permission and Docker integration enabled.
- [ ] Merge P2 PR #62; only then start P3.
