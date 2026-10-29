import sys
import importlib.util

import numpy as np
import pandas as pd
import pytest

from grader.baselines import materialize
from grader.checks import causality_check
from grader.runner import StrategyRunner
from test_grader_runner import market_data


def test_worker_payloads_cannot_be_imported_as_grader_modules():
    assert importlib.util.find_spec('grader.baselines.strategies.lstm_recipe') is None
    assert importlib.util.find_spec('grader.baselines.strategies.timesfm_zero_shot') is None


@pytest.mark.parametrize('name,lookback',[('flat',1),('buy_and_hold',1),('momentum_1h',2),('momentum_24h',25)])
def test_simple_baselines_use_real_worker_and_have_causal_complete_positions(tmp_path,name,lookback):
    data=market_data(56)
    source=materialize(name,tmp_path/'source',markets=['ETH'],resolution='1h')
    times=pd.DatetimeIndex(pd.to_datetime(data.frame.available_at_ms.iloc[lookback:],unit='ms',utc=True))
    with StrategyRunner(source,tmp_path/'run',seed=17) as runner:
        declaration=runner.declare()
        assert declaration.lookback_bars==lookback and not declaration.uses_dev_lake
        state=runner.fit(data)
        positions=runner.predict(state,data,times,declaration)
        expected=0. if name=='flat' else 1.
        np.testing.assert_array_equal(positions.target_position,np.full(len(times),expected))
        assert causality_check(runner,state,data,times,declaration,positions,block=1,seed=29).passed
    assert not any(name.startswith('ethpredict_candidate') for name in sys.modules)
