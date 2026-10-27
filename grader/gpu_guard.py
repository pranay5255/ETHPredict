"""Trusted data-free process holding the physical VRAM reservation."""
import json
import os
import sys

import torch

from .resources import GPU_CAP_BYTES
from .sandbox import protect_parent_memory


def main():
    protect_parent_memory()
    torch.cuda.init()
    total = torch.cuda.get_device_properties(0).total_memory
    reserve = max(0, total - GPU_CAP_BYTES)
    allocation = torch.empty(reserve, dtype=torch.uint8, device='cuda')
    torch.cuda.synchronize()
    print(json.dumps({'pid': os.getpid(), 'total_bytes': total,
                      'reserved_bytes': allocation.numel(), 'cap_bytes': GPU_CAP_BYTES}), flush=True)
    sys.stdin.buffer.read(1)  # controller owns this pipe; no candidate can access it


if __name__ == '__main__':
    main()
