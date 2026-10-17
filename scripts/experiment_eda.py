"""Read-only experiment EDA and guarded cleanup for the local Trackio archive.

Usage: .venv/bin/python scripts/experiment_eda.py clean --apply
       .venv/bin/python scripts/experiment_eda.py build
       .venv/bin/python scripts/experiment_eda.py serve
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import shutil
import sys
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
ARTIFACTS = ROOT / "artifacts"
FRONTEND = ROOT / "eda_frontend"
PROJECT = "ethpredict_eda"
SOURCE = "ethpredict"


def _deps():
    import trackio
    from trackio.sqlite_storage import SQLiteStorage
    return trackio, SQLiteStorage


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def resolve_path(value: Any) -> Path | None:
    if not isinstance(value, str) or not value:
        return None
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def get(data: dict, dotted: str, default=None):
    value = data
    for part in dotted.split("."):
        if not isinstance(value, dict):
            return default
        value = value.get(part)
    return default if value is None else value


def finite(value):
    try:
        number = float(value)
        return number if math.isfinite(number) else None
    except (TypeError, ValueError):
        return None


def inventory():
    trackio, _ = _deps()
    runs = list(trackio.Api().runs(SOURCE))
    smoke = [r for r in runs if r.config.get("smoke") is True]
    setup = [r for r in runs if r.config.get("stage") == "trackio_setup"]
    research = [r for r in runs if r not in smoke and r not in setup]
    if any(r.config.get("smoke") is True for r in research):
        raise RuntimeError("Smoke classification overlap")
    return {"smoke": smoke, "setup": setup, "research": research}


def cleanup(apply: bool):
    groups = inventory()
    before = {key: sorted(r.id for r in value) for key, value in groups.items()}
    local_dirs = []
    for manifest_path in (ARTIFACTS / "runs").glob("*/manifest.json"):
        manifest = read_json(manifest_path)
        if manifest.get("smoke") is True:
            local_dirs.append(manifest_path.parent)
    receipts = []
    for run in groups["setup"]:
        receipt = ARTIFACTS / "trackio_doctor" / (run.name.replace("/", "_") + ".json")
        if receipt.is_file():
            payload = read_json(receipt)
            if payload.get("run") != run.name:
                raise RuntimeError(f"Setup receipt mismatch: {receipt}")
            receipts.append(receipt)
    report = {"before_ids": before, "local_smoke_dirs": [str(p.relative_to(ROOT)) for p in local_dirs],
              "setup_receipts": [str(p.relative_to(ROOT)) for p in receipts]}
    if not apply:
        print(json.dumps(report, indent=2))
        return
    if not groups["smoke"] and not groups["setup"] and not local_dirs and not receipts:
        print(f"No smoke/setup records remain; preserved {len(groups['research'])} research runs")
        return
    # The inventory is persisted before deletion, including every protected research ID.
    audit = ARTIFACTS / "eda_cleanup_inventory.json"
    audit.write_text(json.dumps(report, indent=2) + "\n")
    for run in groups["smoke"] + groups["setup"]:
        if run.id not in before["smoke"] + before["setup"]:
            raise RuntimeError("Run ID changed during cleanup")
        if not run.delete():
            raise RuntimeError(f"Trackio refused deletion of {run.id}")
    for directory in local_dirs:
        if directory.parent != ARTIFACTS / "runs" or read_json(directory / "manifest.json").get("smoke") is not True:
            raise RuntimeError(f"Unsafe artifact deletion: {directory}")
        shutil.rmtree(directory)
    for receipt in receipts:
        receipt.unlink()
    after = inventory()
    if sorted(r.id for r in after["research"]) != before["research"] or after["smoke"] or after["setup"]:
        raise RuntimeError("Post-cleanup Trackio inventory mismatch")
    if any(p.exists() for p in local_dirs + receipts):
        raise RuntimeError("Smoke artifacts remain")
    print(f"Removed {len(groups['smoke'])} smoke runs, {len(groups['setup'])} setup checks, "
          f"{len(local_dirs)} smoke directories, {len(receipts)} matching receipts; "
          f"preserved {len(after['research'])} research runs")


def _flat(data: dict, prefix=""):
    out = {}
    for key, value in data.items():
        name = f"{prefix}.{key}" if prefix else key
        if isinstance(value, dict):
            out.update(_flat(value, name))
        elif finite(value) is not None:
            out[name] = float(value)
    return out


def _local_manifest(config):
    path = resolve_path(config.get("manifest_path") or get(config, "report_paths.manifest")
                        or get(config, "artifact_paths.manifest"))
    if path and path.is_file():
        return read_json(path), path
    return None, path


def _cohort(run_id: str, local: dict | None):
    if run_id.startswith("afml_freeze") or "afml_freeze" in run_id:
        return "AFML freeze"
    if run_id.startswith("v2_") or (local and get(local, "config.version") == 2):
        return "V2 early"
    return "V2 archive"


def _family(run_id: str):
    for family in ("bars", "features", "sampling", "accounting", "repro"):
        if family in run_id:
            return family
    return "early MVP" if "mvp" in run_id or "heavy" in run_id else "other"


def _artifacts(config: dict, local: dict | None, manifest_path: Path | None):
    paths = dict(config.get("artifact_paths") or {})
    if local:
        paths.update(local.get("artifact_paths") or {})
    paths["manifest"] = str(manifest_path) if manifest_path else config.get("manifest_path")
    availability = {}
    missing = []
    for key, raw in paths.items():
        if key == "config":
            continue
        path = resolve_path(raw)
        available = bool(path and path.is_file())
        availability[key] = available
        if not available:
            missing.append(key)
    return availability, missing, paths


def _forecast(local: dict | None, logs: dict):
    rows = []
    if local:
        for fold in local.get("folds", []) or []:
            for horizon, metrics in (fold.get("metrics") or {}).items():
                rows.append({"split": "validation", "horizon": horizon, "fold": fold.get("fold"),
                             "mae": finite(get(metrics, "return.mae")), "rmse": finite(get(metrics, "return.rmse")),
                             "log_loss": finite(get(metrics, "direction.log_loss")),
                             "direction_accuracy": finite(get(metrics, "direction.accuracy")), "source": "local fold"})
        for split, horizons in (local.get("metrics", {}).get("forecast") or {}).items():
            for horizon, metrics in horizons.items():
                rows.append({"split": split, "horizon": horizon, "fold": None,
                             "mae": finite(get(metrics, "return.mae")), "rmse": finite(get(metrics, "return.rmse")),
                             "log_loss": finite(get(metrics, "direction.log_loss")),
                             "direction_accuracy": finite(get(metrics, "direction.accuracy")), "source": "local aggregate"})
    if not rows:
        grouped = defaultdict(dict)
        for key, value in logs.items():
            if key.startswith("forecast."):
                parts = key.split(".")
                if len(parts) >= 5:
                    grouped[(parts[1], parts[2])][".".join(parts[3:])] = value
        for (split, horizon), m in grouped.items():
            rows.append({"split": split, "horizon": horizon, "fold": None,
                         "mae": finite(m.get("return.mae")), "rmse": finite(m.get("return.rmse")),
                         "log_loss": finite(m.get("direction.log_loss")),
                         "direction_accuracy": finite(m.get("direction.accuracy")), "source": "Trackio aggregate"})
    return rows


def _backtest(local: dict | None, logs: dict):
    output = {}
    for split in ("validation", "test"):
        source = get(local or {}, f"metrics.{split}")
        prefix = "backtest" if (local or {}).get("stage") == "forecast_benchmark" else "alpha"
        metric = source if isinstance(source, dict) else {
            key[len(f"{prefix}.{split}."):]: value for key, value in logs.items()
            if key.startswith(f"{prefix}.{split}.")
        }
        output[split] = {key: finite(metric.get(key)) for key in
                         ("trades", "trades_per_day", "gross_pnl", "fees", "net_pnl", "turnover",
                          "max_drawdown", "candidate_coverage", "coverage")}
    return output


def _prediction_details(paths, available):
    if not available.get("oof_predictions"):
        return []
    import pandas as pd
    frame = pd.read_parquet(resolve_path(paths["oof_predictions"]))
    details = []
    for horizon in ("next_5m", "next_hour"):
        p, y, r, pred = (f"{horizon}_{suffix}" for suffix in
                         ("direction_prob", "true_direction", "true_return", "pred_return"))
        if not {p, y, r, pred}.issubset(frame.columns):
            continue
        subset = frame[[p, y, r, pred] + (["fold"] if "fold" in frame else [])].dropna()
        if subset.empty:
            continue
        subset = subset.copy()
        subset["bucket"] = pd.cut(subset[p], bins=[i / 10 for i in range(11)], include_lowest=True)
        for (fold, bucket), group in subset.groupby(["fold", "bucket"], observed=True) if "fold" in subset else [
            ( (None, bucket), group) for bucket, group in subset.groupby("bucket", observed=True)]:
            details.append({"horizon": horizon, "fold": int(fold) if fold is not None else None,
                            "bucket": str(bucket), "count": len(group), "mean_probability": float(group[p].mean()),
                            "realized_frequency": float(group[y].mean()), "mean_pred_return": float(group[pred].mean()),
                            "mean_realized_return": float(group[r].mean())})
    return details


def _trade_details(paths, available):
    import pandas as pd
    details = {}
    for split, key in (("validation", "validation_trades"), ("test", "test_trades")):
        if not available.get(key):
            continue
        frame = pd.read_parquet(resolve_path(paths[key]))
        if "net_pnl" not in frame:
            continue
        pnl = frame["net_pnl"].dropna()
        # Cap payload size while preserving complete cumulative endpoints.
        step = max(1, math.ceil(len(pnl) / 250))
        equity = pnl.cumsum().iloc[::step]
        details[split] = {"net_pnl": [float(x) for x in pnl.iloc[::step]],
                          "equity": [float(x) for x in equity], "trade_count": len(pnl)}
    return details


def normalize(run, logs):
    config = run.config or {}
    local, manifest_path = _local_manifest(config)
    available, missing, paths = _artifacts(config, local, manifest_path)
    stage = config.get("stage", "unknown")
    run_id = str(config.get("run_id") or run.name.split("/")[1] if "/" in run.name else run.name)
    local_config = local.get("config", {}) if local else {}
    local_metrics = _flat(local.get("metrics", {})) if local else {}
    prefix = "backtest" if stage == "forecast_benchmark" else "alpha"
    disagreements = []
    for split in ("validation", "test"):
        for metric in ("trades", "gross_pnl", "fees", "net_pnl", "turnover", "max_drawdown"):
            a = local_metrics.get(f"{split}.{metric}")
            b = finite(logs.get(f"{prefix}.{split}.{metric}"))
            if a is not None and b is not None and not math.isclose(a, b, rel_tol=1e-6, abs_tol=1e-6):
                disagreements.append(f"{split}.{metric}: local {a:g} vs Trackio {b:g}")
    status = str((local or {}).get("status") or config.get("status") or config.get("trackio_status") or "unknown")
    if status == "success":
        status = "completed"
    backtest = _backtest(local, logs)
    if status == "completed" and backtest["validation"]["trades"] == 0:
        status = "no trade"
    role = config.get("selection_status") or "unselected"
    if config.get("is_raw_best") and config.get("is_best_trade_qualified"):
        role = "raw best + trade qualified"
    elif config.get("is_raw_best"):
        role = "raw best"
    elif config.get("is_best_trade_qualified"):
        role = "trade qualified best"
    identity = (local or {}).get("run_identity", {})
    cfg = local_config
    costs = cfg.get("costs", {})
    horizon = get(cfg, "alpha_backtest.horizon") or config.get("horizon") or "multiple"
    raw_files = get(local or {}, "raw_data.files") or get(identity, "raw_data.files") or get(config, "run_identity.raw_data.files") or []
    data_id = ",".join(str(x.get("sha256")) for x in raw_files if x.get("sha256")) or None
    data_id = data_id or get(identity, "raw_data.fingerprint") or get(local or {}, "raw_data.fingerprint") or get(cfg, "data.lighter.start_date")
    split_id = (local or {}).get("split_manifest_hash") or config.get("split_manifest_hash")
    split_settings = get(cfg, "validation")
    split_settings_id = json.dumps(split_settings, sort_keys=True) if split_settings else None
    cost_id = ",".join(f"{k}={costs[k]}" for k in sorted(costs)) if costs else None
    test_reuse = bool(get(cfg, "research.final_test_guard.do_not_reuse_test") is False and
                      (local or {}).get("final_test_evaluation", {}).get("reused"))
    if get(cfg, "research.final_test_guard.do_not_reuse_test") is False and _family(run_id) != "early MVP":
        test_reuse = True  # guard was disabled; label as possible reuse, not proven duplication
    row = {"id": run.id, "name": run.name, "run_id": run_id, "stage": stage,
           "family": _family(run_id), "cohort": _cohort(run_id, local), "status": status,
           "selection_role": role, "horizon": horizon, "data_id": data_id,
           "split_id": split_id, "split_settings_id": split_settings_id, "cost_id": cost_id,
           "config_hash": get(identity, "resolved_config_hash") or get(identity, "config_hash"),
           "bar_type": get(cfg, "bars.type"), "feature_set": get(cfg, "features.include"),
           "sampling_mode": get(cfg, "sampling.mode"), "trial_count": get(config, "run_identity.trial_count", 1),
           "backtest": backtest, "forecast": _forecast(local, logs),
           "manifest_path": str(manifest_path) if manifest_path else None,
           "artifact_available": available, "missing_artifacts": missing,
           "evidence": "local artifacts" if local else "Trackio aggregate only",
           "discrepancies": disagreements, "test_reuse_flag": test_reuse,
           "source_url": f"http://127.0.0.1:7860/run?project={SOURCE}&selected_run_id={run.id}"}
    if local:
        row["reliability"] = _prediction_details(paths, available)
        row["trades"] = _trade_details(paths, available)
    else:
        row["reliability"] = []
        row["trades"] = {}
    return row


def _legacy_rows():
    rows = []
    for path in (ARTIFACTS / "runs").glob("*/manifest.json"):
        m = read_json(path)
        if m.get("smoke") is True or "stage1" not in m:
            continue
        for section, items in (("model", get(m, "stage1.trials", [])), ("strategy", get(m, "stage2.strategies", []))):
            for item in items:
                metrics = item.get("metrics", {})
                rows.append({"id": f"legacy:{m.get('run_id')}:{section}:{item.get('trial_id') or item.get('strategy_id')}",
                             "name": f"{m.get('run_id')}/{section}/{item.get('trial_id') or item.get('strategy_id')}",
                             "run_id": m.get("run_id"), "stage": f"legacy {section}", "family": "legacy staged",
                             "cohort": "Legacy staged", "status": "completed", "selection_role": "historical",
                             "horizon": m.get("target"), "data_id": None, "split_id": None,
                             "split_settings_id": None, "cost_id": None,
                             "config_hash": None, "bar_type": None, "feature_set": None, "sampling_mode": None,
                             "trial_count": 1, "backtest": {"validation": metrics if section == "strategy" else {}, "test": {}},
                             "forecast": [], "manifest_path": str(path), "artifact_available": {"manifest": True},
                             "missing_artifacts": [], "evidence": "local historical", "discrepancies": [],
                             "test_reuse_flag": False, "source_url": None, "reliability": [], "trades": {}})
    return rows


def _conclusion(row):
    reasons = []
    if row["status"] in ("failed", "skipped"):
        reasons.append(f"trial {row['status']}")
    if row["status"] == "no trade":
        reasons.append("zero validation trades")
    if row["evidence"] == "Trackio aggregate only":
        reasons.append("prediction and trade files unavailable")
    if row["missing_artifacts"] and row["evidence"] != "Trackio aggregate only":
        reasons.append("missing: " + ", ".join(row["missing_artifacts"][:4]))
    if row["discrepancies"]:
        reasons.append("local and Trackio metrics disagree")
    if row["test_reuse_flag"]:
        reasons.append("final-test reuse guard disabled")
    n = row["backtest"]["validation"].get("trades")
    pnl = row["backtest"]["validation"].get("net_pnl")
    if row["status"] == "no trade":
        statement = "Candidate generation produced no validation trades."
    elif n is not None and n > 0 and pnl is not None:
        statement = f"{int(n)} validation trades; net PnL {pnl:,.2f}."
    elif row["forecast"]:
        statement = "Forecast metrics recorded; trading evidence is incomplete."
    else:
        statement = "Historical result; comparable evidence is incomplete."
    return statement, reasons


def _publish(rows, report):
    trackio, _ = _deps()
    try:
        existing = list(trackio.Api().runs(PROJECT))
    except ValueError:
        existing = []
    for run in existing:
        if not run.delete():
            raise RuntimeError(f"Could not replace distilled run {run.id}")
    for row in rows:
        name = row["id"].replace(":", "_")
        config = {key: row[key] for key in ("id", "name", "run_id", "stage", "family", "cohort", "status", "selection_role", "horizon", "evidence")}
        config["source_project"] = SOURCE
        config["source_run_id"] = row["id"] if not row["id"].startswith("legacy:") else None
        config["manifest_path"] = row["manifest_path"]
        config["missing_artifacts"] = row["missing_artifacts"]
        config["conclusion"] = row["conclusion"]
        trackio.init(project=PROJECT, name=name, config=config, auto_log_cpu=False, auto_log_gpu=False)
        metrics = {"evidence.missing_artifacts": len(row["missing_artifacts"]),
                   "evidence.discrepancies": len(row["discrepancies"])}
        for split in ("validation", "test"):
            for key, value in row["backtest"][split].items():
                if finite(value) is not None:
                    metrics[f"backtest.{split}.{key}"] = float(value)
        for f in row["forecast"]:
            if f["fold"] is None:
                for key in ("mae", "rmse", "log_loss", "direction_accuracy"):
                    if f[key] is not None:
                        metrics[f"forecast.{f['split']}.{f['horizon']}.{key}"] = f[key]
        trackio.log(metrics)
        trackio.finish()
    (FRONTEND / "data.json").write_text(json.dumps({"rows": rows, "report": report}, allow_nan=False) + "\n")


def build():
    _, storage = _deps()
    groups = inventory()
    if groups["smoke"] or groups["setup"]:
        raise RuntimeError("Clean source Trackio first: run clean --apply")
    summary_counts = {r.config.get("run_id"): r.config.get("trial_count")
                      for r in groups["research"] if r.config.get("stage") == "meta_label_mvp_summary"}
    rows = []
    for run in groups["research"]:
        logs = storage.get_logs(SOURCE, run.name, run_id=run.id)
        row = normalize(run, logs[-1] if logs else {})
        row["trial_count"] = int(summary_counts.get(row["run_id"]) or 1)
        row["conclusion"], row["insufficient_reasons"] = _conclusion(row)
        rows.append(row)
    rows.extend(_legacy_rows())
    for row in rows:
        if "conclusion" not in row:
            row["conclusion"], row["insufficient_reasons"] = _conclusion(row)
    counts = Counter(row["status"] for row in rows if row["cohort"] != "Legacy staged")
    report = {"generated_utc": datetime.now(timezone.utc).isoformat(), "source_runs": len(groups["research"]),
              "historical_rows": sum(r["cohort"] == "Legacy staged" for r in rows),
              "status_counts": dict(counts),
              "local_manifest_runs": sum(r["evidence"] == "local artifacts" for r in rows),
              "trackio_only_runs": sum(r["evidence"] == "Trackio aggregate only" for r in rows),
              "interpretation": ["No model is promoted from these archived outcomes.",
                                 "Zero-trade trials are excluded from net-PnL rankings.",
                                 "Paired candidate comparisons require matching cohort, source data, split settings, horizon, and costs; sample indices can still differ.",
                                 "Validation versus test is a stability diagnostic; repeated final-test use is flagged."]}
    FRONTEND.mkdir(exist_ok=True)
    _publish(rows, report)
    (ARTIFACTS / "eda_report.json").write_text(json.dumps(report, indent=2) + "\n")
    lines = ["# ETHPredict archived experiment EDA", "",
             f"Built: {report['generated_utc']}", "",
             f"Source Trackio research runs: {report['source_runs']}",
             f"Local manifest runs: {report['local_manifest_runs']}",
             f"Trackio-only runs: {report['trackio_only_runs']}",
             f"Legacy staged rows: {report['historical_rows']}", "", "## Status", ""]
    lines.extend(f"- {key}: {value}" for key, value in sorted(report["status_counts"].items()))
    lines += ["", "## Interpretation", ""]
    lines.extend(f"- {item}" for item in report["interpretation"])
    lines += ["", "The dashboard ledger lists every experiment, source run, and missing evidence. "
              "Forecast and trading conclusions stay scoped to their data, split, horizon, costs, and cohort.", ""]
    (ARTIFACTS / "eda_report.md").write_text("\n".join(lines))
    print(json.dumps(report, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    clean = sub.add_parser("clean", help="Inventory or delete explicitly marked smoke/setup records")
    clean.add_argument("--apply", action="store_true")
    sub.add_parser("build", help="Rebuild distilled Trackio project and static EDA data")
    sub.add_parser("serve", help="Serve the custom Trackio frontend on localhost:7861")
    args = parser.parse_args()
    if args.command == "clean":
        cleanup(args.apply)
    elif args.command == "build":
        build()
    else:
        trackio, _ = _deps()
        trackio.show(project=PROJECT, frontend_dir=FRONTEND, server_port=7861,
                     host="127.0.0.1", open_browser=False, block_thread=True)


if __name__ == "__main__":
    sys.path.insert(0, str(ROOT))
    main()
