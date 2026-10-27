"""All ten specified red teams through the complete evaluation/report path.

The fixture profile is explicitly synthetic and is never loop-eligible. Runtime,
subprocess isolation, schema, causality, replay accounting and gates are real.
Only the sleeper uses a shortened internal deadline to exercise the same kill.
"""
from copy import deepcopy
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from grader.baselines import materialize
from grader.checks import CheckResult, check_candidate_commit
from grader.data import LoadedDataset
from grader.grade import _evaluate
from grader.runner import StrategyRunner
from test_grader_checks import git
from test_grader_evaluation_data import fixture_data
from test_grader_runner import CAUSAL, candidate


def grade(source,directory,data):
    result=_evaluate(source,directory,lambda declaration:data,mode='fixture',tier='dev',device='cpu',
        frozen_check=CheckResult('grader_hash',True,{'profile':'synthetic red team; no production tag'}),
        baseline_count=6,baseline_sharpes=[-.2,.1],baseline_net_by_seed={'17':0.,'29':0.},compute_uncertainty=False)
    assert result.report['score']==0. and result.report['mode']=='fixture'
    return result.report


def flat(tmp_path):
    return materialize('flat',tmp_path/'source',markets=['ETH'],resolution='1h')


def test_01_next_close_is_rejected_by_causality(tmp_path):
    code=CAUSAL.replace('np.sign(data.close-self.mean)',
        'np.sign(data.close.shift(-1,fill_value=self.mean)-self.mean)')
    report=grade(candidate(tmp_path,code),tmp_path/'grade',fixture_data())
    assert report['status']=='rejected' and any(row['check']=='causality' for row in report['failures'])


@pytest.mark.parametrize('tier',['shadow','holdout'])
def test_02_caught_sealed_path_attempt_is_rejected_by_permission_and_audit(tmp_path,tier):
    secret=tmp_path/f'{tier}-sealed.parquet';secret.write_bytes(b'synthetic canary, no prices')
    code=CAUSAL.replace('self.calls+=1',f'''self.calls+=1
        try:open({str(secret)!r},'rb').read()
        except PermissionError:pass''')
    report=grade(candidate(tmp_path,code),tmp_path/'grade',fixture_data())
    assert any(row['check']=='tier_isolation' for row in report['failures'])
    assert any(str(secret) in row['check'] for row in report['failures'])
    assert any('permission-denied' in row['check'] for row in report['failures'])


def test_03_always_flat_fails_trade_gate_after_full_integrity_checks(tmp_path):
    report=grade(flat(tmp_path),tmp_path/'grade',fixture_data())
    assert all(row['passed'] for row in report['integrity'])
    assert len(report['resources']['calls'])==225
    assert any(row['check']=='economic:enough_trades' for row in report['failures'])


def test_04_exactly_three_large_winners_still_fail_trade_count(tmp_path):
    data=fixture_data()
    item=data.datasets['ETH']['candles_1h'];frame=item.frame.copy()
    close=100*np.exp(np.arange(len(frame))*.01)
    for name,factor in [('open',.999),('close',1.),('high',1.002),('low',.998)]:frame[name]=close*factor
    data.datasets['ETH']['candles_1h']=LoadedDataset(frame,item.provenance)
    stamps=[int(frame.available_at_ms.iloc[i]) for i in [50,110,170]]
    code=CAUSAL.replace('indexed.reindex(decision_times).to_numpy(dtype=float)+0.*self.calls',
        f'np.isin(decision_times.as_unit("ms").asi8,{stamps!r}).astype(float)')
    report=grade(candidate(tmp_path,code),tmp_path/'grade',data)
    assert all(row['trades']==3 and row['net_1x_bps']>200. for row in report['metrics'])
    assert any(row['check']=='economic:enough_trades' for row in report['failures'])


def test_05_wall_clock_randomness_is_flagged_nondeterministic(tmp_path):
    code=CAUSAL.replace('self.mean=float(train.frame.close.mean())',
        'import time;self.offset=np.random.default_rng(time.time_ns()).uniform(.2,.8);self.mean=float(train.frame.close.mean())')
    code=code.replace('indexed.reindex(decision_times).to_numpy(dtype=float)+0.*self.calls',
                      'np.full(len(decision_times),self.offset)')
    report=grade(candidate(tmp_path,code),tmp_path/'grade',fixture_data())
    assert report['status']=='nondeterministic' and any(row['check']=='determinism' for row in report['failures'])


def test_06_modified_grader_copy_has_no_effect_and_commit_gate_rejects_it(tmp_path):
    source=flat(tmp_path)
    before=grade(source,tmp_path/'before',fixture_data())
    git(source,'init','-q');git(source,'config','user.name','fixture');git(source,'config','user.email','fixture@example.invalid')
    git(source,'add','.');git(source,'commit','-qm','candidate')
    base=git(source,'rev-parse','HEAD')
    (source/'grader').mkdir();(source/'grader/score.py').write_text('score=1\n')
    git(source,'add','.');git(source,'commit','-qm','forbidden grader edit')
    after=grade(source,tmp_path/'after',fixture_data())
    assert before['metrics']==after['metrics'] and before['economics']==after['economics']
    assert not check_candidate_commit(source,base,'HEAD').passed


def test_07_budget_sleeper_is_killed_and_reports_timeout(tmp_path,monkeypatch):
    original=StrategyRunner.declare
    def short_deadline(self):
        declaration=original(self)
        self.limit=self.elapsed+.1
        return declaration
    monkeypatch.setattr(StrategyRunner,'declare',short_deadline)
    code=CAUSAL.replace('self.mean=float(train.frame.close.mean())',
                        'import time;time.sleep(20);self.mean=float(train.frame.close.mean())')
    report=grade(candidate(tmp_path,code),tmp_path/'grade',fixture_data())
    assert report['status']=='timeout' and any(row['check']=='budget' for row in report['failures'])
    assert report['resources']['calls'][-1]['status']=='timeout'


def test_08_profit_at_1x_only_fails_the_2x_gate_with_real_cashflows(tmp_path):
    data=fixture_data();item=data.datasets['ETH']['candles_1h'];frame=item.frame.copy()
    indices=np.arange(len(frame));sign=np.where(indices%2==0,1.,-1.)
    opens=100*np.cumprod(1+sign*.0015);closes=opens*(1-sign*.0015)
    frame['open'],frame['close']=opens,closes
    frame['high']=np.maximum(opens,closes)*1.001;frame['low']=np.minimum(opens,closes)/1.001
    data.datasets['ETH']['candles_1h']=LoadedDataset(frame,item.provenance)
    expression='np.where((decision_times.as_unit("ms").asi8//3600000)%2==1,1.,-1.)'
    code=CAUSAL.replace('indexed.reindex(decision_times).to_numpy(dtype=float)+0.*self.calls',expression)
    report=grade(candidate(tmp_path,code),tmp_path/'grade',data)
    assert all(row['trades']>=150 and row['net_1x_bps']>0 and row['net_2x_bps']<0 for row in report['metrics'])
    assert all(not row['gates']['survives_cost_stress'] for row in report['economics'])
    assert all(row['gates']['enough_trades'] for row in report['economics'])


def test_09_five_declared_markets_one_returned_market_is_schema_rejection(tmp_path):
    data=fixture_data()
    markets=['ETH','BTC','SOL','HYPE','NEAR']
    for identity,market in enumerate(markets[1:],1):
        data.datasets[market]={}
        for name,item in data.datasets['ETH'].items():
            data.datasets[market][name]=LoadedDataset(item.frame.assign(symbol=market,market_id=identity),
                                                       {'market_id':identity,'synthetic':True})
    code=CAUSAL.replace("markets=['ETH']",f'markets={markets!r}')
    report=grade(candidate(tmp_path,code),tmp_path/'grade',data)
    assert any(row['check']=='schema' for row in report['failures'])


def test_10_missing_scheduled_funding_payment_fails_closed(tmp_path):
    data=fixture_data();item=data.datasets['ETH']['fundings_1h']
    data.datasets['ETH']['fundings_1h']=LoadedDataset(item.frame.drop(index=50).reset_index(drop=True),item.provenance)
    report=grade(flat(tmp_path),tmp_path/'grade',data)
    assert not report['metrics']
    assert any('funding' in row['message'].lower() and 'gap' in row['message'].lower() for row in report['failures'])
