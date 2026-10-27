# P3 worker boundary — implementation in progress

P2 is merged in #62 (`86a9df0`). The human confirmed `lake_sealed: true` and
`private_files_moved: 247`; the P2 receipt records 259 real OS denials. Nothing
in P3 changes private ownership, invokes sudo, or evaluates holdout.

The worker boundary is implemented and has actual subprocess tests. The full
grader, baseline bank, calibration, all ten integrated red-team grades and
`grader-v1` freeze are still pending. This document is not P3 acceptance.

## Controller and worker memory

The scoring controller never imports candidate modules or interprets their
pickles. A separate Python server preloads existing libraries, including Ray's
bundled cloudpickle. It receives request **paths**, never market data or fitted
objects. Its memory is made non-dumpable before any candidate call. Only a fresh
forked child reads the request, permitted Parquet input, or fitted pickle.

Each predict call reloads the same fit output read-only and discards its own
state. Untruncated prediction caches cannot flow into a truncated rerun.
`checks.causal_slice` filters on `available_at_ms`, not source-open timestamps.
Its tests inspect the actual input files for 20 distinct prefixes. Another test
reruns both fit and predict with the same seed; a wall-clock-seeded fixture
fails the comparison.

The child closes inherited descriptors before candidate loading, redirects
stdin to `/dev/null` and stdout/stderr to its own scratch, then installs Linux
restrictions. It gets only the code snapshot, frozen grader, explicit runtime,
its input, prior fit output read-only, and its writable output. The snapshot
contains `src/` and `strategy/` source files with fresh paths; it excludes Git,
historical CSV, artifacts, credentials, and the candidate's grader directory.
The parent validates position bytes from a bounded regular file with no symlink
or hard links, then checks exact schema, UTC clock, every market, and finite caps.

## Linux enforcement and audit

This host supports Landlock ABI 4; ABI 3 or later is required for truncate
confinement. Landlock denies reads/writes outside the allowed paths. Seccomp
denies network creation, ptrace/process-memory/FD theft, SysV IPC, namespace and
mount changes, metadata permission/ACL changes, and process-group escape. It
also denies syscall numbers beyond this reviewed profile's ceiling, so a newer
kernel API unknown to libseccomp cannot bypass the metadata restrictions.

The installed CUDA driver writes helper-thread names under proc during startup.
Trusted child bootstrap initializes CUDA and a tiny gradient before sealing,
without loading market data or candidate code. Candidate code receives no broad
`/proc`, `/tmp`, home, shared-memory directory, or Docker socket access. The
CPU runner receives no NVIDIA device access.

Strace follows the server and descendants. It must observe successful Landlock
installation and the worker's exit. File opens after confinement are resolved
against explicit roots, including annotated dirfds and resumed syscalls.
Denied opens fail the audit even when the strategy catches PermissionError.
Pointer-only/blinded, truncated, or incomplete traces fail closed. The child is
dumpable solely so its existing tracer can decode it; trusted parent/server
remain non-dumpable. Seccomp prevents the child changing dumpability or its
parent-death signal after sealing. Parent death kills the worker, and normal
completion kills any remaining descendant in its process group.

Kernel reference: [Landlock documentation](https://docs.kernel.org/userspace-api/landlock.html).
Landlock alone does not cover metadata changes or already open descriptors;
descriptor closure and seccomp are necessary parts of this boundary.

## Resource accounting

All candidate calls, including declaration and integrity reruns, consume the
same runner wall-clock budget; grader file preparation/scoring is separate.
Timeout kills the worker and clean server. Every started call, including a
timeout, has a controller-side record. Child `wait4` peak RSS supplements
10 ms aggregate descendant RSS sampling; inability to read resource telemetry
fails closed. CUDA initialization is charged to candidate-call time.

For CUDA, an independent data-free process reserves total device memory minus
22,000,000,000 bytes. This physically limits candidate VRAM between telemetry
samples, including native allocations outside PyTorch's allocator. Display and
driver overhead reduce usable memory further. The guard must remain alive.
The controller samples per-process allocated memory through installed NVML,
deduplicating compute/graphics entries and counting worker descendants. It
does not trust candidate allocator peaks. RSS limit is 48,000,000,000 bytes.

Actual GPU tests verify CUDA gradients, training ReplayEnv access, reservation
receipt, and nonzero independent VRAM measurements. Accounting tests compare
NumPy, Torch, scalar replay, GPU batch replay and final scoring to `1e-9`.

## Private deployment remains separate

The approved private boundary allows isolated workers to consume grader-supplied
causal slices. All private input, fitted state, stdout/stderr, traces, positions
and reports must remain under the private service, inaccessible to the loop
agent. These development tests use synthetic/public inputs only. No hidden
evaluation is enabled yet. The ethgrader account cannot traverse the human's
0750 home; private deployment will require human provisioning of a separate
runtime and frozen checkout after the service artifacts are concrete.
