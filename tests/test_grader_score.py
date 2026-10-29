from copy import deepcopy

import numpy as np
import pandas as pd
import pytest
import yaml

from grader.costs import CostModel
from grader.data import DataContractError
from grader.env import prepare_replay
from grader.score import block_evidence, component_score, economic_report, load_rubric


def execution_fixture(tmp_path, block, *, move_bps=15., count=80):
    clock=pd.date_range('2025-09-01',periods=count,freq='h',tz='UTC')+pd.Timedelta(days=4*(block-1))
    sign=np.where(np.arange(count)%2==0,1.,-1.)
    opens=100*np.cumprod(1+sign*move_bps/10000.)
    # Final close continues the declared alternating fixture; the strategy uses
    # only its fixed parity schedule. This is a synthetic cost gate, not an edge.
    close=opens*(1+np.roll(sign,-1)*move_bps/10000.)
    frame=pd.DataFrame({'timestamp_ms':clock.as_unit('ms').asi8,'open':opens,'close':close,
                        'high':np.maximum(opens,close)*1.001,'low':np.minimum(opens,close)/1.001,'volume':100.})
    funding=tmp_path/f'funding-{block}.parquet'
    pd.DataFrame({'timestamp':clock,'rate':0.}).to_parquet(funding,index=False)
    model=CostModel({'mode':'dynamic','fee_bps':4.,'spread_bps':2.,'slippage_bps':1.,
        'dynamic_funding':{'enabled':True,'historical':True,'paths':[str(funding)],'rate_column':'rate','rate_unit':'bps'}},
        frame.assign(close_time=frame.timestamp_ms+3600000-1),granularity='1h')
    data=prepare_replay({'ETH':frame},{'ETH':model},resolution='1h')
    targets=np.sign(data.returns_bps)
    return data,targets


def test_flat_pre_gate_components_are_reported_but_every_gate_is_preserved(tmp_path):
    blocks=[]
    for block in range(1,6):
        data,_=execution_fixture(tmp_path,block)
        blocks.append(block_evidence(data,np.zeros((data.steps,1)),block=block))
    score=component_score(blocks,trial_sharpes=[-.1,.1],trial_count=4)
    assert score['pre_gate_score']==pytest.approx(.1)
    assert score['raw_components']['drawdown']==0.
    report=economic_report(blocks,best_baseline_net_bps=0.)
    assert not report['passed']
    assert {'enough_trades','block_trade_coverage','positive_net_expectancy',
            'positive_across_blocks','survives_cost_stress','beats_best_baseline'}<=set(report['failures'])
    assert all(block.daily.net_1x_bps.eq(0).all() for block in blocks)


def test_actual_accounting_profit_at_1x_and_loss_at_2x_fails_only_relevant_gate(tmp_path):
    blocks=[]
    for block in range(1,6):
        data,targets=execution_fixture(tmp_path,block)
        blocks.append(block_evidence(data,targets,block=block))
    assert all(b.net_1x_bps>0 and b.net_2x_bps<0 and b.trades>=25 for b in blocks)
    report=economic_report(blocks,best_baseline_net_bps=-1e9)
    assert report['failures']==['survives_cost_stress']


def test_three_large_winners_cannot_pass_trade_coverage(tmp_path):
    blocks=[]
    for block in range(1,6):
        data,targets=execution_fixture(tmp_path,block,move_bps=1000.)
        targets[:]=0.
        for index in [2,22,42]:targets[index]=np.sign(data.returns_bps[index])
        evidence=block_evidence(data,targets,block=block)
        assert evidence.trades==3 and evidence.net_1x_bps>0
        blocks.append(evidence)
    report=economic_report(blocks,best_baseline_net_bps=0.)
    assert 'enough_trades' in report['failures'] and 'block_trade_coverage' in report['failures']


def test_comparability_and_strict_best_baseline_gate(tmp_path):
    blocks=[]
    for block in range(1,6):
        data,targets=execution_fixture(tmp_path,block,move_bps=30.)
        blocks.append(block_evidence(data,targets,block=block))
    net=sum(block.net_1x_bps for block in blocks)
    assert not economic_report(blocks,best_baseline_net_bps=net)['passed']
    assert economic_report(blocks,best_baseline_net_bps=net-1.)['passed']
    assert not economic_report(blocks,best_baseline_net_bps=None)['passed']
    assert not economic_report(blocks[:4],best_baseline_net_bps=-1e9)['passed']
    assert 'credible_features' in economic_report(blocks,best_baseline_net_bps=-1e9,
                                                engineered_features=['returns'])['failures']


def test_dsr_counts_crashes_and_increases_hurdle_at_same_daily_path(tmp_path):
    blocks=[]
    for block in range(1,6):
        data,targets=execution_fixture(tmp_path,block,move_bps=30.)
        blocks.append(block_evidence(data,targets,block=block))
    a=component_score(blocks,trial_sharpes=[-.2,.1],trial_count=2)
    b=component_score(blocks,trial_sharpes=[-.2,.1],trial_count=102)
    assert b['trial_count']==102
    assert b['deflated_sharpe_evidence']['benchmark_sharpe']>a['deflated_sharpe_evidence']['benchmark_sharpe']
    assert b['components']['deflated_sharpe']<=a['components']['deflated_sharpe']


def test_rubric_rejects_unknown_fields_duplicates_and_gate_weakening(tmp_path):
    rubric=load_rubric()
    for bad in [dict(rubric,unknown=1), {**rubric,'economic':{**rubric['economic'],'min_trades':3}},
                {**rubric,'components':{**rubric['components'],'invented':{'weight':.2,'zero':0.,'one':1.}}}]:
        path=tmp_path/'rubric.yaml';path.write_text(yaml.safe_dump(bad))
        with pytest.raises(DataContractError):load_rubric(path)
    path.write_text('version: 1\nversion: 2\n')
    with pytest.raises(DataContractError,match='Duplicate'):load_rubric(path)
