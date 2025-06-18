from pathlib import Path
import pandas as pd
import pytest

from src.runner import run_backtest_and_sim, Config


def test_dex_sim_flag():
    cfg = Config(dataset="x", trainer={"framework": "pytorch"}, market_maker={}, foundry={}, flags={"enable_dex_sim": True})
    with pytest.raises(NotImplementedError):
        run_backtest_and_sim(cfg, pd.DataFrame({"return": [0]}), Path("model.pt"))

