import multiprocessing as mp
from pathlib import Path
import itertools
import json
import yaml

from ..adaptor import adapter

RESULTS_DIR = Path("results")
RESULTS_DIR.mkdir(exist_ok=True)


def _run_trial(cfg, trial_id):
    data = adapter.load_data(cfg)
    X, y = adapter.make_features(cfg, data)
    res = adapter.train_models(cfg, X, y)
    path = RESULTS_DIR / f"{trial_id}.json"
    with open(path, "w") as f:
        json.dump({"trial": trial_id, "metrics": res}, f)
    return path


def run_single(cfg, tag=""):
    trial_id = tag or cfg.get("experiment", {}).get("id", "exp")
    _run_trial(cfg, trial_id)


def _expand_grid(cfg):
    grid_params = {k: v for k, v in cfg.items() if isinstance(v, list)}
    if not grid_params:
        return [cfg]
    keys, values = zip(*grid_params.items())
    trials = []
    for combo in itertools.product(*values):
        new_cfg = yaml.safe_load(yaml.dump(cfg))
        for key, val in zip(keys, combo):
            new_cfg[key] = val
        trials.append(new_cfg)
    return trials


def run_grid(cfg, max_par=1):
    trials = _expand_grid(cfg)
    with mp.Pool(processes=max_par) as pool:
        for i, _ in enumerate(pool.imap_unordered(lambda c: _run_trial(c, f"trial_{i}"), trials), 1):
            pass


def generate_report(input_dir: Path, out_file: Path):
    lines = []
    for p in Path(input_dir).glob("*.json"):
        with open(p) as f:
            data = json.load(f)
        lines.append(f"<p>{p.name}: {data['metrics']}</p>")
    html = "\n".join(lines)
    out_file.write_text(html)
