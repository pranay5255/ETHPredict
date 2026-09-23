"""Verify that the local ETHPredict Trackio project can write and read a run."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import yaml

from src.utils.trackio_logging import log_trackio_run, trackio_settings


def main() -> None:
    config = yaml.safe_load(Path("configs/config.yml").read_text(encoding="utf-8"))
    settings = trackio_settings(config)
    if settings.get("space_id") or settings.get("server_url"):
        raise RuntimeError("The Trackio doctor checks the local project only")
    config["tracking"]["trackio"].update({"auto_log_gpu": False, "auto_log_cpu": False})
    project = str(settings.get("project", "ethpredict"))
    name = "setup_smoke/" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    receipt = Path("artifacts/trackio_doctor") / (name.replace("/", "_") + ".json")
    log_trackio_run(
        config,
        name=name,
        group="setup_smoke",
        run_config={"stage": "trackio_setup", "read_only": True},
        metrics={"setup": {"readback": 1}},
        receipt_path=receipt,
    )
    print(json.dumps({"project": project, "run": name, "receipt": str(receipt)}, indent=2))


if __name__ == "__main__":
    main()
