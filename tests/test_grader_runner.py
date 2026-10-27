from pathlib import Path
import textwrap

import numpy as np
import pandas as pd
import pytest

from grader.contracts import Declaration, read_positions
from grader.data import DataContractError, LoadedDataset, MarketData
from grader.runner import StrategyRunner, WorkerFailure


CAUSAL = '''
import pandas as pd
import numpy as np
class Strategy:
    name='causal-fixture'
    markets=['ETH']
    resolution='1h'
    max_horizon_bars=1
    lookback_bars=2
    budget_tier='fast'
    def fit(self, train, ctx):
        self.mean=float(train.frame.close.mean())
        self.calls=0
    def predict(self, history, decision_times):
        self.calls+=1
        data=history.datasets['ETH'].frame
        indexed=pd.Series(np.sign(data.close-self.mean).to_numpy(),index=pd.to_datetime(data.timestamp_ms+3600000,unit='ms',utc=True))
        return pd.DataFrame({'timestamp':decision_times,'market':'ETH','target_position':indexed.reindex(decision_times).to_numpy(dtype=float)+0.*self.calls})
'''


def candidate(tmp_path, code=CAUSAL):
    root=tmp_path/'source';(root/'strategy').mkdir(parents=True)
    (root/'strategy'/'strategy.py').write_text(textwrap.dedent(code))
    return root


def market_data(count=8):
    clock=pd.date_range('2025-09-01',periods=count,freq='h',tz='UTC')
    frame=pd.DataFrame({'timestamp_ms':clock.as_unit('ms').asi8,
                        'available_at_ms':clock.as_unit('ms').asi8+3600000,
                        'close':100.+np.arange(count)})
    return MarketData({'ETH':LoadedDataset(frame,{'synthetic':True})})


def test_real_runner_fits_predicts_and_reloads_identical_fit_state(tmp_path):
    source=candidate(tmp_path)
    # A modified candidate grader copy must never be copied or imported.
    (source/'grader').mkdir();(source/'grader'/'__init__.py').write_text('raise RuntimeError("mutable grader")')
    data=market_data()
    times=pd.to_datetime(data.frame.available_at_ms.iloc[2:],unit='ms',utc=True)
    with StrategyRunner(source,tmp_path/'run',seed=13) as runner:
        declaration=runner.declare()
        state=runner.fit(data)
        first=runner.predict(state,data,times,declaration)
        second=runner.predict(state,data,times,declaration)
        pd.testing.assert_frame_equal(first,second)
        assert len(runner.calls)==4 and all(item.audit.passed for item in runner.calls)
        assert runner.elapsed>0 and max(item.peak_rss_bytes for item in runner.calls)>0
    assert not (tmp_path/'run'/'candidate'/'grader').exists()


def test_caught_private_open_still_rejects_by_actual_audit(tmp_path):
    secret=tmp_path/'synthetic-secret';secret.write_text('must not read')
    code=CAUSAL.replace('self.calls+=1',f'''self.calls+=1
        try:open({str(secret)!r}).read()
        except PermissionError:pass''')
    source=candidate(tmp_path,code)
    data=market_data()
    times=pd.to_datetime(data.frame.available_at_ms,unit='ms',utc=True)
    with StrategyRunner(source,tmp_path/'run',seed=13) as runner:
        declaration=runner.declare();state=runner.fit(data)
        with pytest.raises(WorkerFailure,match='audit') as failure:
            runner.predict(state,data,times,declaration)
        assert 'tier_isolation' in failure.value.reasons
        assert any(str(secret) in reason for reason in failure.value.reasons)


def test_budget_kills_worker_and_records_timeout(tmp_path):
    code=CAUSAL.replace('self.calls+=1','import time;time.sleep(20);self.calls+=1')
    data=market_data()
    with StrategyRunner(candidate(tmp_path,code),tmp_path/'run',seed=1) as runner:
        declaration=runner.declare();state=runner.fit(data)
        runner.limit=runner.elapsed+.2  # internal bounded fixture; no CLI override
        with pytest.raises(WorkerFailure) as error:
            runner.predict(state,data,pd.to_datetime(data.frame.available_at_ms,unit='ms',utc=True),declaration)
        assert error.value.status=='timeout' and error.value.reasons==('budget',)
        assert runner.process.poll() is not None


def test_position_reader_rejects_unknown_fields_incomplete_grid_and_symlinks(tmp_path):
    declaration=Declaration.parse(dict(name='fixture',markets=['ETH','BTC'],resolution='1h',
        max_horizon_bars=1,lookback_bars=1,budget_tier='fast',engineered_features=[],pretrained_models=[]))
    times=pd.date_range('2025-09-01',periods=3,freq='h',tz='UTC')
    frame=pd.DataFrame([(time,market,0.) for time in times for market in ['BTC','ETH']],
                       columns=['timestamp','market','target_position'])
    path=tmp_path/'positions.parquet';frame.to_parquet(path,index=False)
    assert len(read_positions(path,declaration,times))==6
    for bad in [frame.assign(unknown=1),frame.iloc[:-1],frame.assign(target_position=np.nan),
                frame.assign(target_position=1.01),frame.assign(timestamp=frame.timestamp.dt.tz_localize(None))]:
        bad.to_parquet(path,index=False)
        with pytest.raises(DataContractError):read_positions(path,declaration,times)
    link=tmp_path/'symlink';link.symlink_to(path)
    with pytest.raises(OSError):read_positions(link,declaration,times)


def test_cuda_runner_has_independent_vram_receipt_and_training_factory(tmp_path):
    import torch
    from test_grader_replay import replay_fixture
    if not torch.cuda.is_available():
        pytest.skip('CUDA hardware required for the actual resource boundary')
    code=CAUSAL.replace('self.mean=float(train.frame.close.mean())','''self.mean=float(train.frame.close.mean())
        import torch
        self.weight=torch.nn.Parameter(torch.ones(8,device=ctx.device,dtype=torch.float64))
        self.weight.square().sum().backward()
        assert torch.equal(self.weight.grad,torch.full_like(self.weight,2))
        env=ctx.replay_env(lookback=1)
        _,reward,_,_=env.step(np.zeros(len(env.data.markets)))
        assert np.isfinite(reward)''')
    code=code.replace("markets=['ETH']", "markets=['ETH','BTC']")
    replay,frames,_=replay_fixture(tmp_path)
    data=MarketData({name:LoadedDataset(frame.assign(available_at_ms=frame.timestamp_ms+3600000),
                                      {'synthetic':True}) for name,frame in frames.items()})
    with StrategyRunner(candidate(tmp_path,code),tmp_path/'run',seed=17,device='cuda') as runner:
        receipt=runner.guard.receipt
        assert receipt['cap_bytes']==22*1000**3
        assert receipt['reserved_bytes']>=receipt['total_bytes']-receipt['cap_bytes']
        runner.declare();runner.fit(data,replay=replay)
        assert max(call.peak_vram_bytes for call in runner.calls)>0
        assert max(call.peak_vram_bytes for call in runner.calls)<=receipt['cap_bytes']


def test_replay_factory_and_prediction_inputs_cannot_escape_training_or_available_history(tmp_path):
    from test_grader_replay import replay_fixture
    replay,_,_=replay_fixture(tmp_path)
    data=market_data()
    with StrategyRunner(candidate(tmp_path),tmp_path/'run',seed=17) as runner:
        declaration=runner.declare()
        with pytest.raises(DataContractError,match='Replay factory'):
            runner.fit(data,replay=replay)
        state=runner.fit(data)
        times=pd.DatetimeIndex(pd.to_datetime(data.frame.available_at_ms.iloc[:-1],unit='ms',utc=True))
        with pytest.raises(DataContractError,match='unavailable'):
            runner.predict(state,data,times,declaration)


def test_killing_the_clean_server_cannot_leave_an_unaudited_worker_running(tmp_path):
    import os
    import psutil
    code=CAUSAL.replace('self.calls+=1','import os,signal;os.kill(os.getppid(),signal.SIGKILL);self.calls+=1')
    data=market_data()
    with StrategyRunner(candidate(tmp_path,code),tmp_path/'run',seed=1) as runner:
        declaration=runner.declare();state=runner.fit(data)
        with pytest.raises(WorkerFailure):
            runner.predict(state,data,pd.DatetimeIndex(pd.to_datetime(data.frame.available_at_ms,unit='ms',utc=True)),declaration)
        assert runner.process.poll() is not None
        assert runner.calls[-1].status=='rejected'
