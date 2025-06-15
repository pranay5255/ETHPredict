"""GPU resource management for experiment orchestration."""

import threading
from contextlib import contextmanager
from typing import Optional

class GPUSemaphore:
    """Manages GPU access across multiple processes."""
    
    def __init__(self, max_gpus: int = 1):
        self._semaphore = threading.Semaphore(max_gpus)
        self._lock = threading.Lock()
        self._active_gpus = 0
        
    @contextmanager
    def acquire(self):
        """Acquire GPU resource with context manager."""
        self._semaphore.acquire()
        with self._lock:
            self._active_gpus += 1
        try:
            yield
        finally:
            with self._lock:
                self._active_gpus -= 1
            self._semaphore.release()
            
    @property
    def active_gpus(self) -> int:
        """Get number of currently active GPUs."""
        with self._lock:
            return self._active_gpus

# Global GPU manager instance
gpu_manager: Optional[GPUSemaphore] = None

def init_gpu_manager(max_gpus: int = 1):
    """Initialize global GPU manager."""
    global gpu_manager
    gpu_manager = GPUSemaphore(max_gpus) 