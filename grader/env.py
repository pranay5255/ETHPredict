"""Frozen replay accounting, shared by numpy scoring and GPU batch training.

The first source bar supplies a completed observation. A decision at its close
executes at the following source bar's open, plus the declared delay. Funding is
entry-exclusive/exit-inclusive. A final liquidation uses the last source close
at its inclusive close_time (one millisecond before the tier boundary), avoiding
any read of the following tier's opening funding payment.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import numpy as np
import pandas as pd
import torch

from .costs import CostModel
from .cost_stress import stress_cashflows
from .data import RESOLUTIONS
from .reward import numpy_step_pnl, torch_step_pnl


@dataclass(frozen=True)
class ReplayData:
    markets: tuple[str, ...]
    resolution: str
    bars: np.ndarray              # [source bars, markets, open/high/low/close/volume]
    source_times_ms: np.ndarray
    returns_bps: np.ndarray       # [steps, markets], starts at NEXT open
    funding_bps: np.ndarray       # raw signed payer rate sum per interval
    unit_costs_bps: np.ndarray    # [steps, markets, buy/sell]
    exit_costs_bps: np.ndarray    # [markets, buy/sell]
    fill_factors: np.ndarray     # indexed by original decision, not delayed execution
    unit_components_bps: np.ndarray  # [steps, markets, buy/sell, fee/spread/slippage]
    exit_components_bps: np.ndarray
    delay: int
    provenance: dict

    @property
    def steps(self):
        return len(self.returns_bps)


def prepare_replay(frames: Mapping[str, pd.DataFrame], models: Mapping[str, CostModel],
                   *, resolution: str) -> ReplayData:
    """Freeze already permitted source data and vendored CostModel quotes.

    The caller supplies ONLY an allowed training or evaluation span. Dynamic
    funding must be loaded by CostModel, whose boundary/gap checks are reused.
    No alternate implementation of fees/spread/slippage/fills is introduced.
    """
    if resolution not in RESOLUTIONS or not frames or set(frames) != set(models):
        raise ValueError('Declare all markets, models and a known resolution')
    markets = tuple(frames)
    times = None
    bars, returns, funding, costs, exits, fills, components, exit_components, manifests = [], [], [], [], [], [], [], [], {}
    delays = {model.delay for model in models.values()}
    if len(delays) != 1:
        raise ValueError('All markets must share the execution clock/delay')
    interval = RESOLUTIONS[resolution]
    for market in markets:
        frame = frames[market].reset_index(drop=True)
        if not pd.api.types.is_integer_dtype(frame.timestamp_ms) or pd.api.types.is_bool_dtype(frame.timestamp_ms):
            raise ValueError('Replay timestamps require integer milliseconds')
        clock = frame.timestamp_ms.to_numpy(dtype=np.int64)
        if len(clock) < 2 or (np.diff(clock) != interval).any() or (clock % interval).any():
            raise ValueError('Replay needs complete source bars and at least one step')
        if times is not None and not np.array_equal(times, clock):
            raise ValueError('Markets have different source clocks')
        times = clock
        values = frame[['open','high','low','close','volume']].to_numpy(dtype=np.float64)
        if not np.isfinite(values).all() or (values[:, :4] <= 0).any() or (values[:, 4] < 0).any():
            raise ValueError('Invalid replay prices or volume')
        if (values[:, 1] < np.max(values[:, [0,2,3]], axis=1)).any() or (
                values[:, 2] > np.min(values[:, [0,1,3]], axis=1)).any():
            raise ValueError('Invalid replay OHLC geometry')
        model = models[market]
        if model.funding is None:
            raise ValueError('Replay requires identified dynamic funding; static zero is not a substitute')
        if model.path is None or len(model.path) != len(frame):
            raise ValueError('CostModel price path must match the permitted replay span')
        model_clock = np.asarray([model.timestamp(i).value // 1_000_000 for i in range(len(frame))])
        if not np.array_equal(model_clock, clock + interval - 1):
            raise ValueError('CostModel must use the same inclusive source close clock')
        begin, end = clock[1:], np.r_[clock[2:], clock[-1] + interval - 1]
        finish_prices = np.r_[values[2:, 0], values[-1, 3]]
        returns.append((finish_prices / values[1:, 0] - 1) * 10_000.)
        left = np.searchsorted(model.funding_times, begin * 1_000_000, side='right')
        right = np.searchsorted(model.funding_times, end * 1_000_000, side='right')
        funding.append(model.funding_cumulative[right] - model.funding_cumulative[left])
        quoted, fill, component_rows = [], [], []
        for index in range(len(frame)-1):
            sides, component_sides = [], []
            for side in [1, -1]:
                quote = model.quote({'sample_index':index,'entry_idx':index,'horizon_bars':1,'side':side})
                row=[quote[key]/2. for key in ['fee_bps','spread_bps','slippage_bps']]
                component_sides.append(row);sides.append(sum(row))
            quoted.append(sides)
            component_rows.append(component_sides)
            fill.append(model.fill({'sample_index':index,'horizon':1})['fill_factor'])
        liquidation, liquidation_components = [], []
        for side in [1, -1]:
            quote = model.quote({'sample_index':len(frame)-1,'entry_idx':len(frame)-1,'horizon_bars':1,'side':side})
            row=[quote[key]/2. for key in ['fee_bps','spread_bps','slippage_bps']]
            liquidation_components.append(row);liquidation.append(sum(row))
        bars.append(values);costs.append(quoted);exits.append(liquidation);fills.append(fill)
        components.append(component_rows);exit_components.append(liquidation_components)
        manifests[market] = model.manifest()
    arrays = [np.stack(bars, axis=1), times.copy(), np.stack(returns, axis=1),
              np.stack(funding, axis=1), np.stack(costs, axis=1),
              np.asarray(exits), np.stack(fills, axis=1), np.stack(components,axis=1), np.asarray(exit_components)]
    for array in arrays:
        array.flags.writeable = False
    return ReplayData(markets, resolution, *arrays, delays.pop(), manifests)


def _action(action, shape):
    value = np.asarray(action, dtype=np.float64)
    if value.shape != shape or not np.isfinite(value).all() or (np.abs(value) > 1).any():
        raise ValueError('Actions must be finite target positions in [-1,1], one per market')
    return value


def step_cashflows(data: ReplayData, index: int, positions, previous, *, liquidate=False,
                   stress_multiple=1.) -> dict:
    """Canonical final-scoring path and ReplayEnv transition accounting."""
    positions, previous = _action(positions, (len(data.markets),)), _action(previous, (len(data.markets),))
    delta = positions - previous
    direction = (delta < 0).astype(int)
    components = data.unit_components_bps[index,np.arange(len(data.markets)),direction] * np.abs(delta)[:,None]
    if liquidate:
        exit_direction = (positions > 0).astype(int)  # selling a long / buying a short
        exit_components = data.unit_components_bps[index+1] if index+1<data.steps else data.exit_components_bps
        components = components + exit_components[np.arange(len(data.markets)),exit_direction] * np.abs(positions)[:,None]
    cost = components.sum(axis=-1)
    funding = positions * data.funding_bps[index]
    stressed = stress_cashflows(cost, funding, stress_multiple)
    transaction = cost * stress_multiple
    funding = stressed - transaction
    reward = numpy_step_pnl(positions, data.returns_bps[index], transaction, funding)
    turnover = np.abs(delta) + (np.abs(positions) if liquidate else 0.)
    trades = ((previous == 0) & (positions != 0)) | (previous * positions < 0)
    return {'net_bps': reward, 'gross_bps': positions * data.returns_bps[index],
            'transaction_cost_bps': transaction, 'funding_bps': funding,
            'fee_bps':components[:,0]*stress_multiple,'spread_bps':components[:,1]*stress_multiple,
            'slippage_bps':components[:,2]*stress_multiple,
            'fill_factor':data.fill_factors[index-data.delay].copy() if index>=data.delay else np.zeros(len(data.markets)),
            'execution_timestamp_ms':np.full(len(data.markets),data.source_times_ms[index+1],dtype=np.int64),
            'turnover': turnover, 'trades': trades.astype(int), 'positions': positions.copy()}


def score_positions(data: ReplayData, targets, *, stress_multiple=1.) -> dict:
    targets = _action(targets, (data.steps, len(data.markets)))
    if not np.isfinite(stress_multiple) or stress_multiple < 1:
        raise ValueError('Invalid cost stress multiple')
    position = np.zeros(len(data.markets))
    records = []
    for index in range(data.steps):
        previous = position.copy()
        if index >= data.delay:
            decision = index - data.delay
            position = position + data.fill_factors[decision] * (targets[decision] - position)
        records.append(step_cashflows(data,index,position,previous,
                       liquidate=index==data.steps-1,stress_multiple=stress_multiple))
    return {key: np.stack([record[key] for record in records]) for key in records[0]}


class ReplayEnv:
    def __init__(self, data, markets=None, resolution=None, cost_model=None,
                 window=None, seed=0, *, lookback=1):
        if not isinstance(data,ReplayData):
            frames={key:value.frame for key,value in data.datasets.items()} if hasattr(data,'datasets') else data
            if not isinstance(frames,Mapping) or not frames:
                raise ValueError('Replay data must be MarketData, market frames or prepared ReplayData')
            declared=tuple(frames) if markets is None else tuple(markets)
            models=cost_model if isinstance(cost_model,Mapping) else ({declared[0]:cost_model} if len(declared)==1 else {})
            if set(declared)!=set(frames) or any(not isinstance(value,CostModel) for value in models.values()):
                raise ValueError('Provide a vendored CostModel for every declared market')
            data=prepare_replay({market:frames[market] for market in declared},models,resolution=resolution)
        elif cost_model is not None:
            models=cost_model if isinstance(cost_model,Mapping) else ({data.markets[0]:cost_model} if len(data.markets)==1 else {})
            if set(models)!=set(data.markets) or any(model.identity!=data.provenance[market]['id'] for market,model in models.items()):
                raise ValueError('Prepared replay came from different cost models')
        if markets is not None and tuple(markets) != data.markets or resolution is not None and resolution != data.resolution:
            raise ValueError('Replay market/resolution contract differs')
        if type(lookback) is not int or lookback < 1 or data.steps < lookback:
            raise ValueError('Insufficient replay lookback')
        self.data, self.seed, self.lookback = data, int(seed), lookback
        self.rng = np.random.default_rng(self.seed)
        if window is None:
            window = (lookback-1, data.steps)
        if not isinstance(window, tuple) or len(window) != 2 or not (
                lookback-1 <= window[0] < window[1] <= data.steps):
            raise ValueError('Episode window is outside the supplied training span')
        self.window = window
        self.reset()

    def _obs(self):
        end = min(self.index, len(self.data.bars)-1)
        return {'bars':self.data.bars[end-self.lookback+1:end+1].copy(),
                'timestamp_ms':int(self.data.source_times_ms[end]+RESOLUTIONS[self.data.resolution]),
                'positions':self.positions.copy(),'funding_accrued_bps':self.funding.copy(),
                'pending_targets':np.stack(self.queue) if self.queue else np.empty((0,len(self.positions)))}

    def reset(self, *, episode_steps=None):
        left,right=self.window
        if episode_steps is not None:
            if type(episode_steps) is not int or not 1 <= episode_steps <= right-left:
                raise ValueError('Requested episode is outside the training span')
            left=int(self.rng.integers(left,right-episode_steps+1));right=left+episode_steps
        self.index,self.end=left,right
        self.positions=np.zeros(len(self.data.markets));self.funding=np.zeros_like(self.positions)
        self.queue=[];self.terminated=False
        return self._obs()

    def step(self, action):
        if self.terminated:
            raise ValueError('Episode terminated; reset before stepping')
        action=_action(action,(len(self.data.markets),))
        self.queue.append(action.copy())
        previous=self.positions.copy()
        if len(self.queue)>self.data.delay:
            target=self.queue.pop(0)
            decision=self.index-self.data.delay
            self.positions += self.data.fill_factors[decision]*(target-self.positions)
        final=self.index+1==self.end
        record=step_cashflows(self.data,self.index,self.positions,previous,liquidate=final)
        self.funding+=record['funding_bps']
        self.index+=1;self.terminated=final
        observation=self._obs()
        if final:
            observation['positions']=np.zeros_like(self.positions)
            observation['pending_targets']=np.empty((0,len(self.positions)))
        return observation,float(record['net_bps'].sum()),final,record


class TorchReplayBatch:
    """N simultaneous training-only episodes; no Python loop over environments."""
    def __init__(self, data: ReplayData, batch_size, episode_steps, *, lookback=1,
                 seed=0, device='cpu', dtype=torch.float64):
        if any(type(value) is not int or value < 1 for value in [batch_size,episode_steps,lookback]) or episode_steps+lookback-1>data.steps:
            raise ValueError('Batch windows are outside the supplied training span')
        self.data,self.batch_size,self.length,self.lookback=data,batch_size,episode_steps,lookback
        self.device,self.dtype=torch.device(device),dtype
        self.rng=np.random.default_rng(seed)
        self.bars=torch.tensor(data.bars,device=device,dtype=dtype)
        self.device=self.bars.device  # Resolve aliases such as cuda -> cuda:0.
        self.returns=torch.tensor(data.returns_bps,device=device,dtype=dtype)
        self.rates=torch.tensor(data.funding_bps,device=device,dtype=dtype)
        self.costs=torch.tensor(data.unit_costs_bps,device=device,dtype=dtype)
        self.exit_costs=torch.tensor(data.exit_costs_bps,device=device,dtype=dtype)
        self.all_exit_costs=torch.cat([self.costs,self.exit_costs[None]],dim=0)
        self.fills=torch.tensor(data.fill_factors,device=device,dtype=dtype)
        self.reset()

    def _obs(self):
        index=self.starts+self.step_index
        history=index[:,None]+torch.arange(1-self.lookback,1,device=self.device)[None,:]
        return {'bars':self.bars[history], 'positions':self.positions,
                'funding_accrued_bps':self.funding,
                'pending_targets':torch.stack(self.queue,dim=1) if self.queue else self.positions[:,None,:][:,:0]}

    def reset(self):
        self.starts=torch.tensor(self.rng.integers(self.lookback-1,self.data.steps-self.length+1,size=self.batch_size),device=self.device)
        self.step_index=0;self.queue=[]
        self.positions=torch.zeros((self.batch_size,len(self.data.markets)),device=self.device,dtype=self.dtype)
        self.funding=torch.zeros_like(self.positions)
        return self._obs()

    def step(self, action):
        if self.step_index>=self.length or action.shape!=self.positions.shape or (
                action.device!=self.device or action.dtype!=self.dtype or not bool(torch.isfinite(action).all()) or bool((action.abs()>1).any())):
            raise ValueError('Invalid batch action or terminated episode')
        index=self.starts+self.step_index
        previous=self.positions
        self.queue.append(action)
        if len(self.queue)>self.data.delay:
            target=self.queue.pop(0)
            self.positions=previous+self.fills[index-self.data.delay]*(target-previous)
        delta=self.positions-previous
        market=torch.arange(len(self.data.markets),device=self.device)[None,:]
        direction=(delta<0).long()
        transaction=self.costs[index[:,None],market,direction]*delta.abs()
        final=self.step_index+1==self.length
        if final:
            transaction=transaction+self.all_exit_costs[(index+1)[:,None],market,(self.positions>0).long()]*self.positions.abs()
        funding=self.positions*self.rates[index]
        reward=torch_step_pnl(self.positions,self.returns[index],transaction,funding)
        self.funding=self.funding+funding
        self.step_index+=1
        observation=self._obs()
        if final:
            observation['positions']=torch.zeros_like(self.positions)
            observation['pending_targets']=self.positions[:,None,:][:,:0]
        return observation,reward.sum(dim=1),final,{'net_bps':reward,'transaction_cost_bps':transaction,'funding_bps':funding}
