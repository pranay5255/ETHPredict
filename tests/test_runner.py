from pathlib import Path

import pandas as pd

from src.runner import Config, run_backtest_and_sim


def test_run_backtest_and_sim_is_noop_in_lighter_scope():
    cfg = Config(dataset="lighter", trainer={"framework": "pytorch"}, market_maker={}, foundry={}, flags={})

    assert run_backtest_and_sim(cfg, pd.DataFrame({"return": [0]}), Path("model.pt")) is None
