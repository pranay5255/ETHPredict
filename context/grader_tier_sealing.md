# P2 sealing runbook and acceptance record

Status: implementation ready for human review; **P2 is not accepted or merged**.
P0/#60 and P1/#61 are merged. P3–P5 have not started. No sudo, account creation,
production data moves, hidden grading or holdout evaluation has been performed.

## What is implemented

`grader/tiers.yaml` preserves dev `[2025-06-04,2026-05-01)`, shadow
`[2026-05-01,2026-08-01)` and holdout `[2026-08-01,2026-10-16)` UTC. The
October 16 partial day stays outside the literal evaluation window; it is still
sealed rather than left public. Unknown/duplicate tier fields fail closed.
Five contiguous evaluation groups follow an initial training group. Training
grows; inclusive dependency spans `[anchor-lookback,anchor+horizon]` remove
exactly lookback+horizon preceding anchors. The vendored sequence-plus-horizon
embargo removes future-side populations; pure growing-window fits have none.
Audits retain proposed/removed/retained row IDs and bounds. The source hash in
`grader/vendor_sources.json` verifies `grader/splits.py` is a byte-identical copy
of the existing module, with no editable-library imports.

`.claude/settings.json` denies grader edits/writes and private/shadow reads.
Per the [official Claude Code permission documentation](https://code.claude.com/docs/en/permissions#read-and-edit),
`Edit(/grader/**)` covers both editing and writing, and the absolute private
path uses `Read(//srv/ethpredict-grader/**)`. A `Write(path)` rule is not an
effective path restriction. These rules remain defense in depth, not OS isolation.

`scripts/seal_grader_tiers.sh` defaults to a read-only plan. Its Python helper:

- Verifies every original manifest partition and rejects symlinks, hard links,
  untracked source files and unexpected source changes.
- Enrolls **150** May–October monthly files plus **97** funding recovery
  backups/receipts; no private-period copy remains in that recovery tree.
- Archives original manifest bytes privately and creates separate dev-only and
  private manifests without changing partition bytes or filling data.
- Requires a human root invocation and an existing distinct `ethgrader` user;
  first execution also requires the reviewed plan to match the current inventory.
  Every resumed plan is revalidated for exact schema, safe source/destination
  paths, fixed tier boundaries, all five markets, and complete partition enrollment
  before any private directory or source file is changed.
- Stores its plan/journal privately before copying. It copies, fsyncs, verifies,
  renames and only then removes each source, allowing safe same-plan resume.
  It never recursively deletes an unknown directory.
- Enforces private directory 0700/file 0600 ownership by ethgrader. The root
  verification checks hashes, modes, manifests and absence of extra public files.

No strategy is run. Private manifests support the same hash-verifying loader,
with explicit separate roots for public training and private evaluation later.

## Verified local evidence

The read-only production plan was generated without changing the lake:

```text
ETHPREDICT_SEAL_PYTHON=/home/pranay5255/Documents/ethpredict/ETHPredict/.venv/bin/python \
bash scripts/seal_grader_tiers.sh --plan \
--source /home/pranay5255/Documents/ethpredict/ETHPredict/data/lake/lighter \
--output /tmp/ethpredict-p2-seal-plan.json
mode: read_only_plan
partitions_to_move: 150
recovery_files_to_quarantine: 97
```

Focused tests (including the real P1 CSV/lake parity fixture):

```text
ETHPREDICT_SOURCE_REPO=<original checkout> ETHPREDICT_REVIEWED_PLAN=/tmp/ethpredict-p2-seal-plan.json \
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 <existing .venv>/bin/python -m pytest \
tests/test_grader_sealing.py tests/test_grader_tiers.py tests/test_grader_data.py -q
72 passed, 1 skipped (final focused run; real OS gate remains pending)
bash -n scripts/seal_grader_tiers.sh: passed
git diff --check: passed
```

Fixtures cover growing windows, exact purging, genuine future-side embargo,
nonoverlap, all decision resolutions, tier edges, invalid clocks/declarations,
plan-only immutability, byte-preserving moves, interrupted/partial copy resume,
idempotence, pre-move hash rejection, backup quarantine, unsafe paths, ownership
modes and failure of a denial probe against readable files. The **skipped** test
is the actual agent-user PermissionError gate. It is not mocked or claimed passed.

Current full suite with the initialized pinned SDK submodule, P1 integration and reviewed production-plan schema validation enabled:

```text
UV_PROJECT_ENVIRONMENT=<existing .venv> ETHPREDICT_SOURCE_REPO=<original checkout> \
ETHPREDICT_REVIEWED_PLAN=/tmp/ethpredict-p2-seal-plan.json OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 uv run --no-sync pytest -q
418 passed, 1 skipped, 23 warnings in 122.58s (0:02:02)
```

The full suite and focused regression run verify unknown fields, invalid hashes, changed tier
boundaries, absolute/traversing source paths, mismatched destinations, dev-data
moves, missing/duplicate partitions or markets, changed manifest ownership and
forged empty denial plans are rejected before mutation. It also verifies the
reviewed production plan still validates; a fresh read-only plan is identical
to the one already supplied for human review. Both CLI guard tests pass.
No dependencies were added/upgraded. The full-suite skip is the human-only
permission acceptance test, so the green local suite does not establish sealing.

Current machine evidence: agent UID 1000; ethgrader account absent;
`/srv/ethpredict-grader` absent. `ETHPREDICT_SEAL_PLAN` must be set only after
the human has executed sealing. Existing historical inputs/artifacts remain
unchanged; no collection or reward evaluation was performed.

## Human commands — do not run from the agent

The attachment's sections 4.3 and 9 explicitly require the human to create the
user and move the data. Review this PR/script and the plan before running these
commands with `!`. Stop collectors and other source-data readers first.

The account is currently absent. Create it once:

```bash
! sudo useradd --system --user-group --home-dir /srv/ethpredict-grader --shell /usr/sbin/nologin ethgrader
```

Apply the reviewed plan (247 source files move; original bytes are retained privately):

```bash
! sudo env ETHPREDICT_SEAL_PYTHON=/home/pranay5255/Documents/ethpredict/ETHPredict/.venv/bin/python bash /tmp/ethpredict-autoresearch-p2/scripts/seal_grader_tiers.sh --apply --source /home/pranay5255/Documents/ethpredict/ETHPredict/data/lake/lighter --approved-plan /tmp/ethpredict-p2-seal-plan.json
```

This command itself prints private hash/ownership verification. If interrupted,
repeat the **same** command; it resumes the persisted plan. Do not regenerate a
plan from a partially moved source or delete the private journal. On any source
hash/manifest mismatch, stop and review. If the source legitimately changed
before the first move, regenerate and review the read-only plan first.

Optional independent private verification (also human, as root):

```bash
! sudo env ETHPREDICT_SEAL_PYTHON=/home/pranay5255/Documents/ethpredict/ETHPredict/.venv/bin/python bash /tmp/ethpredict-autoresearch-p2/scripts/seal_grader_tiers.sh --verify-private --source /home/pranay5255/Documents/ethpredict/ETHPredict/data/lake/lighter
```

The agent can then run the unprivileged acceptance probe and test:

```bash
/home/pranay5255/Documents/ethpredict/ETHPredict/.venv/bin/python /tmp/ethpredict-autoresearch-p2/scripts/verify_sealed_tiers.py --plan /tmp/ethpredict-p2-seal-plan.json --output /tmp/ethpredict-p2-denial.json

ETHPREDICT_SEAL_PLAN=/tmp/ethpredict-p2-seal-plan.json /home/pranay5255/Documents/ethpredict/ETHPredict/.venv/bin/python -m pytest /tmp/ethpredict-autoresearch-p2/tests/test_grader_sealing.py::test_actual_agent_user_cannot_open_sealed_paths -q
```

Every planned private path must raise **PermissionError**. A nonexistent path,
readable path or missing manifest is failure, not successful isolation. Pair that
probe with the human's private hash/ownership verification to prove the intended
files actually exist. Do not publish private plan/journal contents or data.

## Additional human decision: readable historical copies

`scripts/audit_grader_readable_copies.py` inventories paths without reading
market values or strategy outcomes. The committed
`grader_readable_copy_inventory.json` shows **15** tracked source files needing
restriction, raw funding-recovery observations in a committed receipt, eight
worktrees with readable Git history, and 1,226 derived files requiring review.
Those include CSVs with May/June data inside shadow. The old consumed test is
already development evidence; moving a monthly file cannot make it clean again.

The lake script deliberately does not silently remove tracked files, rewrite
Git history, change other worktrees or delete historical experiments. These are
not fixed by chmod on `/srv`. An unrestricted agent can recover old CSVs from
Git objects or the public remote, even after a working-file move.

Before claiming the full agent isolation required by the attachment, provision
a restricted agent workspace/session containing only permitted code and dev
data, with no raw private-period CSVs/receipts, derived private outputs, other
worktrees or original historical Git objects. Preserve the original evidence in
human-controlled storage. The trusted driver must handle candidate commits and
publication without granting the loop agent arbitrary access to the historical
repository; restrict its network/raw-source access as well. The strategy worker
needs its own enforced filesystem/network boundary in P3. The human has approved
causal slices within that private worker, not unrestricted private storage access.

An existing host session with unrestricted filesystem/network access does not
meet that boundary. Choose/provision the restricted runtime before P2 acceptance;
the monthly move and PermissionError test prove only their specific scope.
No sudoers rule or broad privileged Python capability is installed here.

## Acceptance checklist

- [x] Literal tiers and five-block builder; purge, embargo and no-overlap fixtures.
- [x] Tested seal script and real read-only plan, including recovery copies.
- [x] Effective Claude deny rules.
- [ ] Human created ethgrader and ran the production seal.
- [ ] Real agent-user PermissionError test passed, paired with private hash verification.
- [ ] Historical copies restricted in the actual agent runtime.
- [ ] P2 PR merged after all required acceptance evidence; only then start P3.
