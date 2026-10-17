#!/usr/bin/env python3
"""Execute notebooks/research_pipeline.ipynb and print each cell as it starts.

`jupyter nbconvert --execute` holds cell output until the cell finishes. The
feature cells read the full Lighter tape and take about a minute each, so the
terminal looks stuck after the kernel warning. Ctrl+C during that silence
closes the kernel socket, and ipykernel then prints a ZMQ traceback.

From the repository root:

    uv run --with nbclient --with ipykernel python notebooks/execute_research_pipeline.py
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import nbformat
from nbclient import NotebookClient

ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = ROOT / "notebooks" / "research_pipeline.ipynb"


def _ensure_kernel() -> None:
    subprocess.check_call(
        [sys.executable, "-m", "ipykernel", "install", "--sys-prefix", "--name", "python3"],
        stdout=subprocess.DEVNULL,
    )


def _preview(source: str) -> str:
    for line in source.splitlines():
        stripped = line.strip()
        if stripped and not stripped.startswith("#"):
            return stripped[:100]
    return "(empty)"


def _shutdown(client: NotebookClient) -> None:
    kernel_manager = client.km
    if kernel_manager is None:
        return
    try:
        kernel_manager.shutdown_kernel(now=True)
    except Exception:
        return


def main() -> int:
    _ensure_kernel()
    notebook = nbformat.read(NOTEBOOK, as_version=4)
    code_cells = [
        index
        for index, cell in enumerate(notebook.cells)
        if cell.cell_type == "code" and str(cell.source).strip()
    ]
    code_index = {index: position for position, index in enumerate(code_cells, start=1)}

    def on_cell_start(cell, cell_index: int, **_kwargs) -> None:
        position = code_index.get(cell_index)
        if position is None:
            return
        print(f"[{position}/{len(code_cells)}] {_preview(cell.source)}", flush=True)

    client = NotebookClient(
        notebook,
        timeout=3600,
        kernel_name="python3",
        resources={"metadata": {"path": str(ROOT)}},
    )
    # IPC keeps the kernel on a local socket. CRITICAL hides ipykernel's
    # TCP warning and the socket traceback it prints if the run is cancelled.
    client.extra_arguments = ["--log-level=CRITICAL"]
    client.on_cell_start = on_cell_start
    original_create = client.create_kernel_manager

    def create_kernel_manager():
        kernel_manager = original_create()
        kernel_manager.transport = "ipc"
        return kernel_manager

    client.create_kernel_manager = create_kernel_manager
    print(f"Executing {NOTEBOOK.relative_to(ROOT)}", flush=True)
    try:
        client.execute()
    except KeyboardInterrupt:
        _shutdown(client)
        print(
            "\nStopped. The last line above is the cell that was running. "
            "Feature cells take about a minute and do not print again until they finish.",
            flush=True,
        )
        return 130
    nbformat.write(notebook, NOTEBOOK)
    print(f"Wrote {NOTEBOOK.relative_to(ROOT)}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
