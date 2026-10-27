"""Trusted controller for isolated, freshly fitted/reloaded strategy workers."""
from __future__ import annotations

from dataclasses import dataclass
import json
import os
from pathlib import Path
import selectors
import signal
import subprocess
import sys
import time

import pandas as pd
import numpy as np
import psutil

from .audit import AuditResult, audit_trace
from .contracts import Declaration, check_regular_file, read_positions, regular_bytes, write_market_data, write_replay
from .data import DataContractError, MarketData
from .sandbox import protect_parent_memory
from .resources import GPU_CAP_BYTES, RSS_CAP_BYTES, GpuGuard, GpuMonitor


class WorkerFailure(RuntimeError):
    def __init__(self, message, *, status='rejected', reasons=()):
        super().__init__(message)
        self.status, self.reasons = status, tuple(reasons)


@dataclass(frozen=True)
class CallResult:
    operation: str
    output: Path
    duration_seconds: float
    peak_rss_bytes: int
    peak_vram_bytes: int
    audit: AuditResult
    status: str = 'completed'


def snapshot_candidate(source: Path, target: Path):
    """Copy code only: no .git, CSV, artifacts, credentials or mutable grader."""
    source = source.resolve(strict=True)
    target.mkdir(parents=True)
    strategy_root = source/'strategy'
    repository = source
    if (source/'strategy.py').is_file():
        strategy_root = source
        repository = next((p for p in source.parents if (p/'src').is_dir()), source.parent)
    total = 0
    for name, root in [('src', repository/'src'), ('strategy', strategy_root)]:
        if not root.exists():
            continue
        if root.is_symlink():
            raise WorkerFailure('Candidate code directory is a symlink')
        for path in sorted(root.rglob('*')):
            if path.is_symlink():
                raise WorkerFailure('Candidate code symlinks are forbidden')
            if path.is_dir() or path.suffix not in {'.py', '.json', '.yaml', '.yml', '.md', '.toml', '.txt'}:
                continue
            if '__pycache__' in path.parts:
                continue
            content = regular_bytes(path, maximum=32 * 1024**2)
            total += len(content)
            if total > 32 * 1024**2:
                raise WorkerFailure('Candidate code exceeds 32 MiB')
            destination = target / name / path.relative_to(root)
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_bytes(content)
    if not (target / 'strategy' / 'strategy.py').is_file():
        raise WorkerFailure('Missing strategy/strategy.py')


def runtime_paths(*, gpu=False):
    # No whole-home, whole-/proc, host /tmp, shared-memory or credential mounts.
    read = ['/usr', str(Path(sys.executable).resolve().parent.parent), str(Path(sys.prefix)),
            '/sys', '/proc/driver/nvidia', '/proc/cpuinfo', '/proc/meminfo',
            '/proc/sys/vm/mmap_min_addr',
            '/proc/self/maps', '/proc/self/status', '/proc/self/stat', '/proc/self/statm',
            '/proc/self/fd',  # own descriptors only, after inherited descriptor closure
            '/etc/ld.so.cache', '/etc/localtime', '/etc/nsswitch.conf', '/etc/passwd', '/etc/group',
            '/dev/random', '/dev/urandom']
    write = ['/dev/null']
    if gpu:
        write += ['/dev/nvidia0', '/dev/nvidiactl', '/dev/nvidia-uvm', '/dev/nvidia-uvm-tools']
    return [p for p in read if Path(p).exists()], [p for p in write if Path(p).exists()]


class StrategyRunner:
    """One candidate, one aggregate wall-clock budget, fresh child for every call.

    The parent never imports candidate code or reads a candidate pickle. Fit
    output is reused read-only; predict state is always discarded between calls.
    All fit/predict calls, including causality and determinism reruns, count.
    """
    def __init__(self, candidate, directory, *, seed, device='cpu', registered_model_paths=None, lake_view=None):
        protect_parent_memory()
        self.directory = Path(directory).resolve()
        self.directory.mkdir(parents=True)
        self.candidate = self.directory / 'candidate'
        snapshot_candidate(Path(candidate), self.candidate)
        self.seed, self.device = seed, device
        self.model_paths = {key: str(Path(p).resolve(strict=True)) for key, p in (registered_model_paths or {}).items()}
        self.calls = []
        self.declaration = None
        self.lake_view = lake_view
        self.elapsed = 0.
        self.limit = 600.
        self.deadline = time.monotonic() + self.limit
        self.trace = self.directory / 'private-worker.trace'
        self.trace_offset = 0
        self.worker_pid = None
        self.server_pid = None
        self._selector = selectors.DefaultSelector()
        self._response_buffer = b''
        self.guard = None
        self.gpu_reservation = None
        self.gpu_monitor = None
        if device not in {'cpu', 'cuda'}:
            raise ValueError('The frozen runner supports cpu or the single cuda device')
        if device == 'cuda':
            self.guard = GpuGuard(self.directory)
            self.gpu_reservation = dict(self.guard.receipt)
            try:
                self.gpu_monitor = GpuMonitor()
            except BaseException:
                self.guard.close()
                raise
        environment = {'PATH': '/usr/bin:/bin', 'HOME': str(self.directory),
                       'PYTHONDONTWRITEBYTECODE': '1', 'OMP_NUM_THREADS': '1',
                       'OPENBLAS_NUM_THREADS': '1', 'MKL_NUM_THREADS': '1',
                       'CUBLAS_WORKSPACE_CONFIG': ':4096:8', 'HF_HUB_DISABLE_IMPLICIT_TOKEN': '1'}
        root = Path(__file__).resolve().parent.parent
        bootstrap = f'import sys;sys.path.insert(0,{str(root)!r});from grader.worker import main;main()'
        self.process = subprocess.Popen(['strace', '-f', '-q', '-yy', '-s', '4096',
            '-e', 'trace=%file,%process,%network,%ipc,prctl,landlock_restrict_self',
            '-o', str(self.trace), sys.executable, '-I', '-c', bootstrap],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=open(self.directory / 'server-stderr.log', 'wb'),
            bufsize=0, env=environment, start_new_session=True)
        self._selector.register(self.process.stdout, selectors.EVENT_READ)
        try:
            ready = self._response(time.monotonic() + 60.)
            if set(ready) != {'ready', 'pid'} or ready['ready'] is not True:
                raise WorkerFailure('Worker server did not become ready')
            self.server_pid = ready['pid']
        except BaseException:
            self.close()
            raise

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

    def close(self):
        if getattr(self, 'worker_pid', None):
            try:
                os.killpg(self.worker_pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            self.worker_pid = None
        process = getattr(self, 'process', None)
        if process is not None and process.poll() is None:
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            process.wait(timeout=5)
        if process is not None:
            for pipe in [process.stdin, process.stdout]:
                if pipe:
                    pipe.close()
        self._selector.close()
        if self.gpu_monitor is not None:
            self.gpu_monitor.close()
            self.gpu_monitor = None
        if self.guard is not None:
            self.guard.close()
            self.guard = None

    def _response(self, deadline, *, resources=None):
        while time.monotonic() < deadline:
            if b'\n' in self._response_buffer:
                line, self._response_buffer = self._response_buffer.split(b'\n', 1)
                return json.loads(line)
            if resources is not None:
                if self.guard is not None:
                    self.guard.check()
                descendants = []
                try:
                    descendants = psutil.Process(self.server_pid).children(recursive=True)
                except psutil.NoSuchProcess:
                    pass
                except psutil.AccessDenied as exc:
                    raise WorkerFailure('Cannot audit worker resources', reasons=['budget']) from exc
                rss = 0
                for child in descendants:
                    try:
                        rss += child.memory_info().rss
                    except psutil.NoSuchProcess:
                        pass
                    except psutil.AccessDenied as exc:
                        raise WorkerFailure('Cannot audit worker RSS', reasons=['budget']) from exc
                resources['peak_rss_bytes'] = max(resources['peak_rss_bytes'], rss)
                if rss > RSS_CAP_BYTES:
                    raise WorkerFailure('Worker RSS exceeded 48 GB', reasons=['budget'])
                if self.gpu_monitor is not None:
                    resources['pids'].update(p.pid for p in descendants)
                    allocations = self.gpu_monitor.processes()
                    vram = sum(amount for pid, amount in allocations.items() if pid in resources['pids'])
                    resources['peak_vram_bytes'] = max(resources['peak_vram_bytes'], vram)
                    if vram > GPU_CAP_BYTES:
                        raise WorkerFailure('Worker VRAM exceeded 22 GB', reasons=['budget'])
                if self.trace.stat().st_size > 256 * 1024**2:
                    raise WorkerFailure('Worker audit exceeded 256 MiB', reasons=['tier_isolation', 'budget'])
            if self._selector.select(timeout=min(.01, max(0, deadline-time.monotonic()))):
                chunk = os.read(self.process.stdout.fileno(), 65_536)
                if not chunk:
                    raise WorkerFailure('Worker server stopped unexpectedly', reasons=['tier_isolation'])
                self._response_buffer += chunk
            if self.process.poll() is not None:
                raise WorkerFailure('Worker server stopped unexpectedly', reasons=['tier_isolation'])
        raise WorkerFailure('Aggregate fit/predict budget exhausted', status='timeout', reasons=['budget'])

    def _call(self, operation, *, data=None, state=None, replay=None, decision_times=None,
              dev_lake=None, train_anchor_end_ms=None):
        if self.elapsed >= self.limit:
            raise WorkerFailure('Aggregate fit/predict budget exhausted', status='timeout', reasons=['budget'])
        call = self.directory / f'call-{len(self.calls):04d}-{operation}'
        inputs, output = call / 'input', call / 'output'
        inputs.mkdir(parents=True)
        output.mkdir()
        if data is not None:
            write_market_data(data, inputs / 'market')
        if replay is not None:
            write_replay(replay, inputs / 'replay.npz')
        if self.declaration is not None and self.declaration.uses_dev_lake and self.lake_view is not None:
            cutoff = max(int(item.frame.available_at_ms.max()) for item in data.datasets.values())
            dev_lake = self.lake_view(cutoff)
        read, write = runtime_paths(gpu=self.device == 'cuda')
        models = {} if operation == 'declare' else {key: self.model_paths[key] for key in self.declaration.pretrained_models}
        read += [str(self.candidate), str(Path(__file__).resolve().parent), str(inputs)] + list(models.values())
        if state is not None:
            read.append(str(Path(state).parent))
        if dev_lake is not None:
            read.append(str(Path(dev_lake).resolve(strict=True)))
        write.append(str(output))
        # The request itself is read by trusted bootstrap code before confinement.
        request = call / 'request.json'
        read.append(str(request))
        remaining = self.limit - self.elapsed
        self.deadline = time.monotonic() + remaining
        environment = {'PATH': '/usr/bin:/bin', 'HOME': str(output), 'TMPDIR': str(output),
                       'XDG_CACHE_HOME': str(output), 'HF_HOME': str(output / 'hf'),
                       'HF_HUB_OFFLINE': '1', 'HF_HUB_DISABLE_IMPLICIT_TOKEN': '1',
                       'PYTHONDONTWRITEBYTECODE': '1', 'OMP_NUM_THREADS': '1',
                       'OPENBLAS_NUM_THREADS': '1', 'MKL_NUM_THREADS': '1',
                       'CUBLAS_WORKSPACE_CONFIG': ':4096:8'}
        specification = {'operation': operation, 'candidate': str(self.candidate),
            'entry': str(self.candidate / 'strategy' / 'strategy.py'), 'output': str(output),
            'data': str(inputs / 'market') if data is not None else None,
            'replay': str(inputs / 'replay.npz') if replay is not None else None,
            'state': str(state) if state is not None else None, 'seed': self.seed, 'device': self.device,
            'models': models,
            'deadline': self.deadline, 'decision_times_ms': [] if decision_times is None else list(
                pd.DatetimeIndex(decision_times).as_unit('ms').asi8.astype(int).tolist()),
            'dev_lake': str(dev_lake) if dev_lake is not None else None,
            'train_anchor_end_ms': train_anchor_end_ms, 'environment': environment,
            'read_only': read, 'writable': write, 'max_output_bytes': 4 * 1000**3}
        request.write_text(json.dumps(specification, allow_nan=False))
        offset = self.trace.stat().st_size
        start = time.monotonic()
        resources = {'peak_rss_bytes': 0, 'peak_vram_bytes': 0, 'pids': set()}
        recorded = False
        try:
            self.process.stdin.write((json.dumps({'request_path': str(request)}) + '\n').encode())
            self.process.stdin.flush()
            started = self._response(self.deadline, resources=resources)
            if set(started) != {'started'} or type(started['started']) is not int:
                raise WorkerFailure('Invalid worker start response')
            self.worker_pid = started['started']
            finished = self._response(self.deadline, resources=resources)
            if set(finished) != {'finished', 'returncode', 'peak_rss_bytes'} or finished['finished'] != self.worker_pid:
                raise WorkerFailure('Invalid worker completion response')
            with self.trace.open('r') as source:
                source.seek(offset)
                trace = source.read()
            audit = audit_trace(trace, server_pid=self.server_pid, worker_pid=self.worker_pid,
                                cwd=output, read_only=read, writable=write)
            peak_rss = max(resources['peak_rss_bytes'], finished['peak_rss_bytes'])
            resource_ok = peak_rss <= RSS_CAP_BYTES and resources['peak_vram_bytes'] <= GPU_CAP_BYTES
            result = CallResult(operation, output, time.monotonic()-start,
                                peak_rss,
                                resources['peak_vram_bytes'], audit,
                                'completed' if audit.passed and finished['returncode'] == 0 and resource_ok else 'rejected')
            self.calls.append(result)
            recorded = True
            if not audit.passed:
                reasons = ['tier_isolation', *audit.violations]
                reasons += [] if resource_ok else ['budget']
                reasons += [] if not finished['returncode'] else ['crash']
                raise WorkerFailure('Strategy file audit failed', reasons=reasons)
            if not resource_ok:
                raise WorkerFailure('Recorded peak RSS/VRAM exceeded the frozen budget', reasons=['budget'])
            if finished['returncode']:
                raise WorkerFailure(f'Strategy {operation} exited with code {finished["returncode"]}', reasons=['crash'])
            return result
        except BaseException as exc:
            if not recorded:
                self.calls.append(CallResult(operation, output, time.monotonic()-start,
                    resources['peak_rss_bytes'], resources['peak_vram_bytes'],
                    AuditResult(0, ('incomplete worker execution',), False), getattr(exc, 'status', 'rejected')))
            self.close()
            raise
        finally:
            self.elapsed += time.monotonic()-start
            self.worker_pid = None

    def declare(self):
        result = self._call('declare')
        declaration = Declaration.parse(json.loads(regular_bytes(result.output / 'declaration.json', maximum=32_768)))
        self.limit = 600. if declaration.budget_tier == 'fast' else 3600.
        from .models import verified_model_paths
        verified_model_paths(declaration.pretrained_models, self.model_paths)
        self.declaration = declaration
        return declaration

    def fit(self, data: MarketData, *, replay=None, dev_lake=None, train_anchor_end_ms=None):
        if self.declaration is None or set(data.datasets) != set(self.declaration.markets):
            raise DataContractError('Fit requires the declared market population')
        if replay is not None:
            if set(replay.markets) != set(data.datasets) or any(
                    not np.isin(replay.source_times_ms, item.frame.timestamp_ms).all()
                    for item in data.datasets.values()):
                raise DataContractError('Replay factory escapes the supplied training markets or timestamps')
        result = self._call('fit', data=data, replay=replay, dev_lake=dev_lake,
                            train_anchor_end_ms=train_anchor_end_ms)
        # Check shape/size without reading or interpreting executable contents.
        path = result.output / 'state.pkl'
        check_regular_file(path, maximum=4 * 1000**3)
        return path

    def predict(self, state, history: MarketData, decision_times, declaration):
        times = pd.DatetimeIndex(decision_times)
        if self.declaration != declaration or set(history.datasets) != set(declaration.markets) or not len(times):
            raise DataContractError('Prediction does not match the declared market population')
        cutoff = int(times[-1].value // 1_000_000)
        if any('available_at_ms' not in item.frame or item.frame.available_at_ms.gt(cutoff).any()
               for item in history.datasets.values()):
            raise DataContractError('Prediction history contains unavailable observations')
        result = self._call('predict', state=state, data=history, decision_times=decision_times)
        return read_positions(result.output / 'positions.parquet', declaration, decision_times)
