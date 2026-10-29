import json
import pandas as pd
import pytest
import torch

from grader.baselines import materialize
from grader.checks import causality_check
from grader.runner import StrategyRunner
from test_grader_evaluation_data import fixture_data


@pytest.mark.parametrize('device',['cpu','cuda'])
def test_existing_recipe_fits_in_real_worker_and_predicts_causally_without_future_labels(tmp_path,device):
    if device=='cuda' and not torch.cuda.is_available():
        pytest.skip('CUDA hardware required')
    data=fixture_data(800)
    source=materialize('lstm_recipe',tmp_path/'source',markets=['ETH'],resolution='1h')
    view=lambda cutoff:data.view(tmp_path/'causal-view',cutoff)
    with StrategyRunner(source,tmp_path/'run',seed=17,device=device,lake_view=view) as runner:
        declaration=runner.declare()
        assert declaration.lookback_bars==173 and declaration.max_horizon_bars==1
        train=data.market_slice(slice(0,600))
        replay=data.replay(train,tmp_path/'train-replay',training=True)
        state=runner.fit(train,replay=replay,train_anchor_end_ms=int(train.frame.timestamp_ms.iloc[-2]))
        history=data.market_slice(slice(600-declaration.lookback_bars,628))
        times=pd.DatetimeIndex(pd.to_datetime(data.candles.frame.available_at_ms.iloc[600:628],unit='ms',utc=True))
        positions=runner.predict(state,history,times,declaration)
        assert len(positions)==28 and positions.target_position.abs().le(1).all()
        packet=runner.last_features
        assert packet is not None and set(packet.families)==set(declaration.engineered_features)
        decoded=runner.predict_features(state,packet,declaration)
        pd.testing.assert_frame_equal(positions.assign(timestamp=positions.timestamp.dt.as_unit('ns')),decoded)
        request=json.loads((runner.calls[-1].output.parent/'request.json').read_text())
        assert request['data'] is None and request['dev_lake'] is None
        check=causality_check(runner,state,history,times,declaration,positions,block=1,seed=17)
        assert check.passed
