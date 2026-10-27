from dataclasses import asdict
from pathlib import Path
import subprocess

import numpy as np
import pandas as pd
import pytest

from grader.checks import causal_slice, causality_check, check_candidate_commit, determinism_check, verify_frozen_grader
from grader.data import DataContractError, LoadedDataset, MarketData
from grader.runner import StrategyRunner
from test_grader_runner import CAUSAL, candidate, market_data


def test_causal_strategy_passes_twenty_actual_prefix_runs_and_repeat_fit(tmp_path):
    data=market_data(28)
    times=pd.DatetimeIndex(pd.to_datetime(data.frame.available_at_ms.iloc[2:],unit='ms',utc=True))
    with StrategyRunner(candidate(tmp_path),tmp_path/'run',seed=29) as runner:
        declaration=runner.declare();state=runner.fit(data)
        positions=runner.predict(state,data,times,declaration)
        check=causality_check(runner,state,data,times,declaration,positions,block=1,seed=17)
        assert check.passed and len(set(check.details['cutoffs_ms']))==20
        assert determinism_check(runner,data,data,times,declaration,positions).passed
        assert len(runner.calls)==25
    # Every prefix file really omits unavailable rows; no full-data side channel.
    inputs=list((tmp_path/'run').glob('call-*-predict/input/market/ETH.parquet'))
    lengths=[len(pd.read_parquet(path)) for path in inputs]
    assert max(lengths)==28 and len(set(lengths))==21


def test_future_close_strategy_fails_actual_causality_check(tmp_path):
    code=CAUSAL.replace('np.sign(data.close-self.mean)',
                        'np.sign(data.close.shift(-1,fill_value=self.mean)-self.mean)')
    data=market_data(28)
    times=pd.DatetimeIndex(pd.to_datetime(data.frame.available_at_ms.iloc[2:],unit='ms',utc=True))
    with StrategyRunner(candidate(tmp_path,code),tmp_path/'run',seed=29) as runner:
        declaration=runner.declare();state=runner.fit(data)
        positions=runner.predict(state,data,times,declaration)
        check=causality_check(runner,state,data,times,declaration,positions,block=1,seed=17)
        assert not check.passed and check.details['mismatch_at_ms']


def test_wall_clock_seed_fails_repeat_fit_and_predict(tmp_path):
    code=CAUSAL.replace('self.mean=float(train.frame.close.mean())',
        'import time;self.offset=np.random.default_rng(time.time_ns()).uniform(-.9,.9);self.mean=float(train.frame.close.mean())')
    code=code.replace('indexed.reindex(decision_times).to_numpy(dtype=float)+0.*self.calls',
                      'np.full(len(decision_times),self.offset)')
    data=market_data()
    times=pd.DatetimeIndex(pd.to_datetime(data.frame.available_at_ms,unit='ms',utc=True))
    with StrategyRunner(candidate(tmp_path,code),tmp_path/'run',seed=29) as runner:
        declaration=runner.declare();state=runner.fit(data)
        positions=runner.predict(state,data,times,declaration)
        check=determinism_check(runner,data,data,times,declaration,positions)
        assert not check.passed and check.name=='determinism'


def test_causal_slice_uses_availability_not_source_timestamp():
    data=market_data(8)
    sliced=causal_slice(data,end_ms=int(data.frame.timestamp_ms.iloc[4]))
    assert len(sliced.frame)==4
    assert sliced.frame.available_at_ms.le(data.frame.timestamp_ms.iloc[4]).all()


def git(root,*args):
    return subprocess.check_output(['git','-C',str(root),*args],text=True).strip()


def test_frozen_hashes_and_commit_paths_ignore_mutable_grader(tmp_path):
    root=tmp_path/'repo';root.mkdir();git(root,'init','-q')
    git(root,'config','user.name','fixture');git(root,'config','user.email','fixture@example.invalid')
    (root/'grader').mkdir();(root/'grader'/'score.py').write_text('score=0\n')
    (root/'strategy').mkdir();(root/'strategy'/'strategy.py').write_text('position=0\n')
    git(root,'add','.');git(root,'commit','-qm','initial');git(root,'tag','grader-v1')
    base=git(root,'rev-parse','HEAD')
    assert verify_frozen_grader(root).passed
    (root/'strategy'/'strategy.py').write_text('position=.1\n')
    git(root,'add','.');git(root,'commit','-qm','allowed')
    assert check_candidate_commit(root,base,'HEAD').passed
    (root/'grader'/'score.py').write_text('score=1\n')
    git(root,'add','.');git(root,'commit','-qm','forbidden')
    assert not verify_frozen_grader(root).passed
    assert check_candidate_commit(root,base,'HEAD').details['forbidden']==['grader/score.py']
