"""Independent GPU telemetry using the machine's installed NVML C API.

The reviewed nvmlProcessInfo_t layout is from /usr/include/nvml.h. No Python
NVML package, model code, or candidate-reported peak statistic is required.
"""
from __future__ import annotations

import ctypes
import json
import os
from pathlib import Path
import selectors
import signal
import subprocess
import sys


GPU_CAP_BYTES = 22 * 1000**3
RSS_CAP_BYTES = 48 * 1000**3


class ProcessInfo(ctypes.Structure):
    _fields_ = [('pid', ctypes.c_uint), ('used_memory', ctypes.c_ulonglong),
                ('gpu_instance', ctypes.c_uint), ('compute_instance', ctypes.c_uint)]


class GpuMonitor:
    def __init__(self):
        self.library = ctypes.CDLL('libnvidia-ml.so.1')
        self._check(self.library.nvmlInit_v2())
        self.device = ctypes.c_void_p()
        self._check(self.library.nvmlDeviceGetHandleByIndex_v2(0, ctypes.byref(self.device)))

    @staticmethod
    def _check(result):
        if result:
            raise RuntimeError(f'NVML resource accounting failed: {result}')

    def processes(self):
        # Either API reports the entire process allocation, so deduplicate by PID.
        result = {}
        for name in ['nvmlDeviceGetComputeRunningProcesses_v3', 'nvmlDeviceGetGraphicsRunningProcesses_v3']:
            size = 64
            while True:
                count = ctypes.c_uint(size)
                entries = (ProcessInfo * size)()
                status = getattr(self.library, name)(self.device, ctypes.byref(count), entries)
                if status == 7:  # NVML_ERROR_INSUFFICIENT_SIZE
                    size = count.value + 16
                    if size > 16384:
                        raise RuntimeError('Invalid NVML process count')
                    continue
                self._check(status)
                for item in entries[:count.value]:
                    if item.used_memory == 2**64 - 1:
                        raise RuntimeError('Per-process GPU memory is unavailable')
                    result[item.pid] = max(result.get(item.pid, 0), item.used_memory)
                break
        return result

    def close(self):
        self._check(self.library.nvmlShutdown())


class GpuGuard:
    """Reserve total device bytes minus 22 GB in a separate trusted process.

    This enforces a physical upper bound between telemetry samples, including
    native allocations that bypass PyTorch's allocator. Driver/display overhead
    makes the candidate's usable budget slightly smaller, never larger. The
    guard never receives any market data and must remain alive for the trial.
    """
    def __init__(self, directory: Path):
        root = Path(__file__).resolve().parent.parent
        bootstrap = f'import sys;sys.path.insert(0,{str(root)!r});from grader.gpu_guard import main;main()'
        self.process = subprocess.Popen([sys.executable, '-I', '-c', bootstrap],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=open(directory / 'gpu-guard.log', 'wb'),
            env={'PATH': '/usr/bin:/bin', 'HOME': str(directory), 'OMP_NUM_THREADS': '1',
                 'PYTHONDONTWRITEBYTECODE': '1'}, start_new_session=True)
        selector = selectors.DefaultSelector()
        selector.register(self.process.stdout, selectors.EVENT_READ)
        try:
            if not selector.select(timeout=60.):
                raise RuntimeError('GPU guard startup timed out')
            self.receipt = json.loads(self.process.stdout.readline())
            if set(self.receipt) != {'pid', 'total_bytes', 'reserved_bytes', 'cap_bytes'} or self.receipt['cap_bytes'] != GPU_CAP_BYTES:
                raise RuntimeError('Invalid GPU guard receipt')
            if self.receipt['reserved_bytes'] < max(0, self.receipt['total_bytes'] - GPU_CAP_BYTES):
                raise RuntimeError('GPU guard did not reserve the required memory')
        except BaseException:
            self.close()
            raise
        finally:
            selector.close()

    def check(self):
        if self.process.poll() is not None:
            raise RuntimeError('GPU guard stopped during the trial')

    def close(self):
        if self.process.poll() is None:
            os.killpg(self.process.pid, signal.SIGKILL)
            self.process.wait(timeout=5)
        for pipe in [self.process.stdin, self.process.stdout]:
            if pipe:
                pipe.close()
