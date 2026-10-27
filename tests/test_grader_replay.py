from copy import deepcopy

import numpy as np
import pandas as pd
import pytest
import torch

from grader.costs import CostModel
from grader.env import ReplayEnv, TorchReplayBatch, prepare_replay, score_positions
from grader.reward import numpy_step_pnl, torch_path_pnl, torch_step_pnl


def replay_fixture(tmp_path, *, delay=0, fill=1., sampled=False, count=48):
    rng=np.random.default_rng(19)
    clock=pd.date_range('2025-09-01',periods=count,freq='h',tz='UTC')
    frames,models={},{}
    for market,offset in [('ETH',0),('BTC',1)]:
        opens=100*np.exp(np.cumsum(rng.normal(0,.003,count)))
        closes=opens*np.exp(rng.normal(0,.001,count))
        frame=pd.DataFrame({'timestamp_ms':clock.as_unit('ms').asi8,'open':opens,'close':closes,
                            'high':np.maximum(opens,closes)*(1+.002+np.arange(count)*.00001),
                            'low':np.minimum(opens,closes)/(1+.002+np.arange(count)*.00001),
                            'volume':100.})
        funding=tmp_path/f'{market}.csv'
        pd.DataFrame({'timestamp':clock,'rate':rng.normal(.1,.15,count)}).to_csv(funding,index=False)
        config={'mode':'dynamic','fee_bps':4.,'spread_bps':2.,'slippage_bps':1.,
                'dynamic_spread':{'enabled':True,'source':'high_low_proxy','range_fraction':.1},
                'dynamic_funding':{'enabled':True,'historical':True,'paths':[str(funding)],
                                   'rate_column':'rate','rate_unit':'bps'},
                'execution':{'mode':'maker' if fill<1 else 'taker','maker_fee_bps':1.,
                             'maker_fill_probability':fill,'fill_policy':'sampled' if sampled else 'expected_haircut',
                             'seed':23+offset,'entry_delay_bars':delay}}
        path=frame.assign(close_time=frame.timestamp_ms+3_600_000-1)
        frames[market]=frame;models[market]=CostModel(config,path,granularity='1h')
    return prepare_replay(frames,models,resolution='1h'),frames,models


@pytest.mark.parametrize('delay',[0,1,3,80])
@pytest.mark.parametrize('fill,sampled',[(1.,False),(.6,False),(.6,True)])
def test_numpy_torch_and_final_scoring_parity_with_delay_fills_funding_and_exit(tmp_path,delay,fill,sampled):
    data,_,_=replay_fixture(tmp_path,delay=delay,fill=fill,sampled=sampled)
    targets=np.random.default_rng(7).uniform(-1,1,(data.steps,2))
    final=score_positions(data,targets)
    tensor=lambda array:torch.tensor(np.asarray(array)[None],dtype=torch.float64)
    for multiple in [1.,2.]:
        expected=score_positions(data,targets,stress_multiple=multiple)
        net,positions=torch_path_pnl(tensor(targets),tensor(data.returns_bps),
            tensor(data.unit_costs_bps),tensor(data.funding_bps),tensor(data.fill_factors),
            delay=delay,exit_costs=tensor(data.exit_costs_bps),stress_multiple=multiple)
        np.testing.assert_allclose(net[0].numpy(),expected['net_bps'],rtol=0,atol=1e-9)
        np.testing.assert_allclose(positions[0].numpy(),expected['positions'],rtol=0,atol=1e-9)
    env=ReplayEnv(data,lookback=1,seed=7)
    reward=[]
    for target in targets:
        observation,value,terminated,info=env.step(target)
        reward.append(value)
    np.testing.assert_allclose(reward,final['net_bps'].sum(axis=1),rtol=0,atol=1e-9)
    assert terminated and np.array_equal(observation['positions'],np.zeros(2))
    assert observation['timestamp_ms']==int(data.source_times_ms[-1]+3_600_000)
    with pytest.raises(ValueError,match='terminated'):env.step(np.zeros(2))


def test_completed_observations_next_open_and_actual_funding_clock(tmp_path):
    data,frames,models=replay_fixture(tmp_path,count=6)
    targets=np.ones((data.steps,2))
    score=score_positions(data,targets)
    for market_index,market in enumerate(data.markets):
        model=models[market]
        for step in range(data.steps):
            begin=pd.Timestamp(int(data.source_times_ms[step+1]),unit='ms',tz='UTC')
            end=(pd.Timestamp(int(data.source_times_ms[step+2]),unit='ms',tz='UTC')
                 if step+2<len(data.source_times_ms) else pd.Timestamp(int(data.source_times_ms[-1]+3_600_000-1),unit='ms',tz='UTC'))
            selected=model.funding[(model.funding.timestamp>begin)&(model.funding.timestamp<=end)]
            assert score['funding_bps'][step,market_index]==pytest.approx(selected.rate_bps.sum())
        # Last interval excludes the following tier's opening payment.
        assert score['funding_bps'][-1,market_index]==0.
    env=ReplayEnv(data,lookback=2)
    observation=env.reset()
    assert len(observation['bars'])==2
    assert observation['timestamp_ms']==int(data.source_times_ms[1]+3_600_000)
    assert np.array_equal(observation['bars'],data.bars[:2])
    assert not data.bars.flags.writeable
    observation['bars'][:]=0
    assert not np.array_equal(observation['bars'],data.bars[:2])


def test_source_cost_quote_conversion_components_and_stress_match(tmp_path):
    data,_,models=replay_fixture(tmp_path,count=8)
    target=np.ones((data.steps,2))
    expected=score_positions(data,target)
    for column,market in enumerate(data.markets):
        for key in ['fee_bps','spread_bps','slippage_bps']:
            entry=models[market].quote({'sample_index':0,'entry_idx':0,'horizon_bars':1,'side':1})[key]/2
            exit=models[market].quote({'sample_index':data.steps,'entry_idx':data.steps,'horizon_bars':1,'side':-1})[key]/2
            assert expected[key][:,column].sum()==pytest.approx(entry+exit)
    np.testing.assert_allclose(expected['transaction_cost_bps'],
        expected['fee_bps']+expected['spread_bps']+expected['slippage_bps'],atol=1e-12)
    stressed=score_positions(data,target,stress_multiple=2.)
    np.testing.assert_allclose(stressed['transaction_cost_bps'],2*expected['transaction_cost_bps'])
    np.testing.assert_allclose(stressed['funding_bps'],np.maximum(expected['funding_bps'],0)*2+np.minimum(expected['funding_bps'],0))


def test_public_constructor_accepts_market_frames_and_cost_models(tmp_path):
    data,frames,models=replay_fixture(tmp_path,count=8)
    env=ReplayEnv(frames,list(frames),'1h',models,(0,data.steps),23)
    prepared=ReplayEnv(data,seed=23)
    for _ in range(data.steps):
        assert env.step([.5,-.5])[1]==pytest.approx(prepared.step([.5,-.5])[1],abs=1e-9)


def test_partial_episode_liquidates_at_its_own_cost_quote(tmp_path):
    data,_,models=replay_fixture(tmp_path,count=30)
    end=8
    env=ReplayEnv(data,window=(0,end))
    costs=np.zeros(2)
    for _ in range(end):
        _,_,_,info=env.step([1.,1.]);costs+=info['transaction_cost_bps']
    for i,market in enumerate(data.markets):
        quotes=[models[market].quote({'sample_index':index,'entry_idx':index,'horizon_bars':1,'side':side})
                for index,side in [(0,1),(end,-1)]]
        expected=sum(sum(quote[key] for key in ['fee_bps','spread_bps','slippage_bps'])/2 for quote in quotes)
        assert costs[i]==pytest.approx(expected,abs=1e-9)


@pytest.mark.parametrize('device',['cpu','cuda'])
@pytest.mark.parametrize('delay',[0,2])
def test_batched_windows_match_scalar_environments_without_future_observations(tmp_path,device,delay):
    if device=='cuda' and not torch.cuda.is_available(): pytest.skip('CUDA unavailable')
    data,_,_=replay_fixture(tmp_path,delay=delay,fill=.7,count=80)
    batch=TorchReplayBatch(data,8,12,lookback=3,seed=27,device=device)
    starts=batch.starts.cpu().numpy().tolist()
    scalar=[ReplayEnv(data,window=(start,start+12),lookback=3,seed=27) for start in starts]
    rng=np.random.default_rng(10)
    for step in range(12):
        target=rng.uniform(-1,1,(8,2))
        observation,reward,done,info=batch.step(torch.tensor(target,dtype=torch.float64,device=device))
        references=[env.step(action) for env,action in zip(scalar,target)]
        np.testing.assert_allclose(reward.cpu().numpy(),[item[1] for item in references],rtol=0,atol=1e-9)
        for i,item in enumerate(references):
            np.testing.assert_allclose(observation['bars'][i].cpu().numpy(),item[0]['bars'],rtol=0,atol=1e-9)
            np.testing.assert_allclose(observation['positions'][i].cpu().numpy(),item[0]['positions'],rtol=0,atol=1e-9)
        assert done==(step==11)


def test_torch_reward_gradients_and_nonexecuted_actions(tmp_path):
    data,_,_=replay_fixture(tmp_path,count=20)
    targets=torch.tensor(np.random.default_rng(9).uniform(.1,.8,(3,data.steps,2)),dtype=torch.float64,requires_grad=True)
    tensor=lambda array:torch.tensor(np.repeat(np.asarray(array)[None],3,axis=0),dtype=torch.float64)
    net,_=torch_path_pnl(targets,tensor(data.returns_bps),tensor(data.unit_costs_bps),
        tensor(data.funding_bps),tensor(data.fill_factors),delay=1,exit_costs=tensor(data.exit_costs_bps))
    net.sum().backward()
    assert torch.isfinite(targets.grad).all() and targets.grad.abs().sum()>0
    assert torch.equal(targets.grad[:,-1],torch.zeros_like(targets.grad[:,-1]))
    never=targets.detach().clone().requires_grad_(True)
    value,_=torch_path_pnl(never,tensor(data.returns_bps),tensor(data.unit_costs_bps),
        tensor(data.funding_bps),tensor(data.fill_factors),delay=100,exit_costs=tensor(data.exit_costs_bps))
    value.sum().backward()
    assert torch.equal(never.grad,torch.zeros_like(never))


@pytest.mark.parametrize('action',[[np.nan,0],[2,0],[0],[[0,0]]])
def test_bad_actions_fail_closed(tmp_path,action):
    data,_,_=replay_fixture(tmp_path,count=8)
    with pytest.raises(ValueError):ReplayEnv(data).step(action)


def test_out_of_training_windows_and_unknown_funding_fail_closed(tmp_path):
    data,frames,models=replay_fixture(tmp_path,count=8)
    with pytest.raises(ValueError,match='outside'):ReplayEnv(data,window=(0,data.steps+1))
    with pytest.raises(ValueError,match='outside'):ReplayEnv(data).reset(episode_steps=data.steps+1)
    with pytest.raises(ValueError,match='outside'):TorchReplayBatch(data,4,data.steps+1)
    models['ETH'].funding=None
    with pytest.raises(ValueError,match='dynamic funding'):prepare_replay(frames,models,resolution='1h')


@pytest.mark.parametrize('missing',[0,3,7])
def test_missing_scheduled_payment_never_reaches_reward(tmp_path,missing):
    _,frames,models=replay_fixture(tmp_path,count=8)
    model=models['ETH'];source=tmp_path/'ETH.csv'
    frame=pd.read_csv(source);frame.drop(index=missing).to_csv(source,index=False)
    with pytest.raises(ValueError):CostModel(model.config,model.path,granularity='1h')


def test_reward_rejects_nonfinite_shapes_negative_costs_and_dtype_mismatch():
    x=np.ones((2,3))
    for invalid in [np.full((2,3),np.nan),np.ones((3,2))]:
        with pytest.raises(ValueError):numpy_step_pnl(x,invalid,x,x)
    with pytest.raises(ValueError):numpy_step_pnl(x,x,-x,x)
    a=torch.ones(2,dtype=torch.float64)
    with pytest.raises(ValueError):torch_step_pnl(a,a.float(),a,a)
