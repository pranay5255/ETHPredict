from dataclasses import replace
import json
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from grader.contracts import Declaration
from grader.data import DataContractError
from grader.feature_packets import FeaturePacket, NOISE_NAMES, feature_noise, read_packet, write_packet
from grader.policy_importance import policy_fold_importance, summarize_policy_importance
from grader.runner import StrategyRunner, WorkerFailure
from grader.checks import causality_check, causal_slice
from test_grader_runner import candidate, market_data


def declaration():
    return Declaration.parse(dict(name='input-fixture', markets=['ETH'], resolution='1h',
        max_horizon_bars=1, lookback_bars=2, budget_tier='fast', engineered_features=['signal'],
        pretrained_models=[], uses_dev_lake=False))


def packet():
    times = pd.date_range('2025-09-01', periods=24, freq='h', tz='UTC')
    noise = feature_noise(times.as_unit('ms').asi8, seed=17, market='ETH')
    frame = pd.DataFrame({'timestamp':times, 'market':'ETH', 'signal':np.arange(24, dtype=float),
        **{name:noise[:,i].astype(float) for i,name in enumerate(NOISE_NAMES)}})
    return FeaturePacket(frame, {'signal':['signal']}, {name:[name] for name in NOISE_NAMES},
        {name:0 for name in NOISE_NAMES})


def test_frozen_noise_is_finite_stateless_and_seed_and_market_specific():
    clock = pd.date_range('2025-09-01', periods=1000, freq='h', tz='UTC').as_unit('ms').asi8
    full = feature_noise(clock, seed=17, market='ETH')
    assert np.isfinite(full).all() and full.dtype == np.float32
    np.testing.assert_array_equal(full[14:61], feature_noise(clock[14:61], seed=17, market='ETH'))
    assert not np.array_equal(full, feature_noise(clock, seed=29, market='ETH'))
    assert not np.array_equal(full, feature_noise(clock, seed=17, market='BTC'))
    assert abs(full.mean()) < .1 and .9 < full.std() < 1.1


def test_packet_roundtrip_normalizes_time_units_and_keeps_numeric_inputs(tmp_path):
    original=packet()
    write_packet(original,tmp_path)
    times=pd.DatetimeIndex(original.frame.timestamp).as_unit('ms')
    restored=read_packet(tmp_path,declaration(),times,seed=17)
    expected=original.frame.assign(timestamp=original.frame.timestamp.dt.as_unit('ns'))
    pd.testing.assert_frame_equal(expected,restored.frame)


@pytest.mark.parametrize('corruption', ['unknown', 'family', 'duplicate_column', 'lag', 'nan', 'fake_noise', 'missing_row'])
def test_packet_rejects_unknown_unpartitioned_nonfinite_and_forged_inputs(tmp_path, corruption):
    value = packet()
    if corruption == 'nan': value.frame.loc[0, 'signal'] = np.nan
    if corruption == 'fake_noise': value.frame.loc[0, NOISE_NAMES[0]] = 0.
    if corruption == 'missing_row': value = replace(value, frame=value.frame.iloc[:-1])
    write_packet(value, tmp_path)
    metadata = json.loads((tmp_path/'features.json').read_text())
    if corruption == 'unknown': metadata['claimed_importance'] = 1.
    if corruption == 'family': metadata['families'] = {'undeclared':['signal']}
    if corruption == 'duplicate_column': metadata['families']['signal'].append(NOISE_NAMES[0])
    if corruption == 'lag': metadata['noise_lags'][NOISE_NAMES[0]] = 3
    (tmp_path/'features.json').write_text(json.dumps(metadata))
    with pytest.raises(DataContractError):
        read_packet(tmp_path, declaration(), packet().frame.timestamp, seed=17)


POLICY = '''
import numpy as np
import pandas as pd
from grader.feature_packets import FeaturePacket, NOISE_NAMES, feature_noise
class Strategy:
    name='fitted-input-fixture'
    markets=['ETH']
    resolution='1h'
    max_horizon_bars=1
    lookback_bars=2
    budget_tier='fast'
    engineered_features=['signal']
    uses_dev_lake=False
    def fit(self, train, ctx):
        self.seed=ctx.seed
        x=train.frame.close.to_numpy()-100.
        # An actual fitted numeric policy, with both noise controls ingested.
        noise=ctx.noise(train.frame.available_at_ms.to_numpy(), market='ETH')
        X=np.column_stack([x,noise])
        self.coef=np.linalg.lstsq(X,.8*x,rcond=None)[0]
    def feature_packet(self, history, times):
        source=history.frame
        index=pd.to_datetime(source.available_at_ms,unit='ms',utc=True)
        signal=pd.Series(source.close.to_numpy()-100.,index=index).reindex(times).to_numpy()
        noise=feature_noise(pd.DatetimeIndex(times).as_unit('ms').asi8,seed=self.seed,market='ETH')
        frame=pd.DataFrame({'timestamp':times,'market':'ETH','signal':signal,
            **{name:noise[:,i].astype(float) for i,name in enumerate(NOISE_NAMES)}})
        return FeaturePacket(frame,{'signal':['signal']},{name:[name] for name in NOISE_NAMES},{name:0 for name in NOISE_NAMES})
    def predict_features(self, packet):
        frame=packet.frame
        values=frame[['signal',*NOISE_NAMES]].to_numpy() @ self.coef
        return frame[['timestamp','market']].assign(target_position=np.clip(values,-1.,1.))
    def predict(self, history, times):
        return self.predict_features(self.feature_packet(history,times))
'''


def test_actual_fitted_policy_has_purged_five_fold_signal_importance_and_causal_inputs(tmp_path):
    from grader.tiers import build_walk_forward_blocks
    data = market_data(180)
    data.datasets['ETH'].frame['close'] = 100.+np.sin(np.arange(180)*2.7)
    with StrategyRunner(candidate(tmp_path, POLICY), tmp_path/'run', seed=17) as runner:
        config = runner.declare()
        blocks = build_walk_forward_blocks(pd.DatetimeIndex(pd.to_datetime(data.frame.timestamp_ms.iloc[:-1],unit='ms',utc=True)),
                                          resolution='1h',lookback_bars=2,max_horizon_bars=1)
        folds = []
        for block in blocks:
            train = causal_slice(data,end_ms=int(data.frame.available_at_ms.iloc[block.train[-1]]))
            history = causal_slice(data,end_ms=int(data.frame.available_at_ms.iloc[block.evaluation[-1]]))
            times = pd.DatetimeIndex(pd.to_datetime(data.frame.available_at_ms.iloc[block.evaluation], unit='ms', utc=True))
            state = runner.fit(train)
            positions = runner.predict(state, history, times, config)
            inputs = runner.last_features
            replay = SimpleNamespace(markets=('ETH',), steps=len(times), delay=0,
                returns_bps=np.sign(inputs.frame.signal.to_numpy())[:,None])
            folds.append(policy_fold_importance(runner,state,inputs,config,positions,replay,block))
            if block.block == 0:
                assert causality_check(runner,state,history,times,config,positions,block=1,seed=17,
                                       reference_features=inputs).passed
        report = summarize_policy_importance(folds)
        assert report['status'] == 'completed' and not report['noise_control_failures']
        signal = next(row for row in report['summary'] if row['feature'] == 'signal')
        assert signal['category'] == 'positive' and signal['negative_log_loss_drop'] > .1
        assert all(not set(fold['fit_indices']) & set(fold['evaluation_indices']) for fold in folds)
        assert all(call.audit.passed for call in runner.calls)


def test_future_feature_inputs_fail_causality_even_when_positions_are_flat(tmp_path):
    code = POLICY.replace("signal=pd.Series(source.close.to_numpy()-100.,index=index).reindex(times).to_numpy()",
        "signal=np.full(len(times), float(source.close.iloc[-1]))")
    code = code.replace("values=frame[['signal',*NOISE_NAMES]].to_numpy() @ self.coef", "values=np.zeros(len(frame))")
    data = market_data(48)
    times = pd.DatetimeIndex(pd.to_datetime(data.frame.available_at_ms.iloc[2:],unit='ms',utc=True))
    with StrategyRunner(candidate(tmp_path,code),tmp_path/'run',seed=17) as runner:
        config=runner.declare(); state=runner.fit(data)
        positions=runner.predict(state,data,times,config)
        result=causality_check(runner,state,data,times,config,positions,block=1,seed=17)
        assert not result.passed and not result.details['feature_inputs_agree']


def test_permutation_worker_cannot_read_the_normal_prediction_history(tmp_path):
    raw_path=tmp_path/'run/call-0002-predict/input/market/ETH.parquet'
    code=POLICY.replace('self.seed=ctx.seed', f"self.seed=ctx.seed; self.raw_path={str(raw_path)!r}")
    code=code.replace('frame=packet.frame', "try:open(self.raw_path,'rb').read()\n        except PermissionError:pass\n        frame=packet.frame")
    with StrategyRunner(candidate(tmp_path,code),tmp_path/'run',seed=17) as runner:
        data=market_data(48)
        config=runner.declare(); state=runner.fit(data)
        times=pd.DatetimeIndex(pd.to_datetime(data.frame.available_at_ms.iloc[2:],unit='ms',utc=True))
        runner.predict(state,data,times,config)
        assert raw_path.exists() and runner.calls[-1].audit.passed
        with pytest.raises(WorkerFailure,match='audit'):
            runner.predict_features(state,runner.last_features,config)


def test_missing_input_protocol_is_unavailable_not_a_credible_feature_pass():
    block=SimpleNamespace(block=0,audit={'purged':True})
    fold=policy_fold_importance(None,None,None,None,None,None,block)
    report=summarize_policy_importance([fold])
    assert report['status']=='unavailable' and not report['summary']


def test_noise_winner_cannot_satisfy_the_credible_feature_gate(tmp_path):
    from grader.score import block_evidence, economic_report
    from test_grader_score import execution_fixture
    value=packet()
    class NoiseRunner:
        seed=17
        def predict_features(self,state,inputs,config):
            return inputs.frame[['timestamp','market']].assign(
                target_position=np.tanh(inputs.frame[NOISE_NAMES[0]].to_numpy()))
    runner=NoiseRunner()
    reference=runner.predict_features(None,value,None)
    replay=SimpleNamespace(markets=('ETH',),steps=len(value.frame),delay=0,
        returns_bps=np.sign(value.frame[NOISE_NAMES[0]].to_numpy())[:,None])
    folds=[policy_fold_importance(runner,None,value,declaration(),reference,replay,
        SimpleNamespace(block=i,train=np.arange(4),evaluation=np.arange(10,34),audit={})) for i in range(5)]
    importance=summarize_policy_importance(folds)
    assert NOISE_NAMES[0] in importance['noise_control_failures']
    evidence=[]
    for block in range(1,6):
        replay,targets=execution_fixture(tmp_path,block,move_bps=30.)
        evidence.append(block_evidence(replay,targets,block=block))
    economics=economic_report(evidence,best_baseline_net_bps=-1e9,engineered_features=['signal'],importance=importance)
    assert economics['failures']==['credible_features']


def test_family_permutations_keep_all_markets_and_exclude_nonexecuted_delay_tail():
    base=packet()
    other=base.frame.assign(market='BTC',signal=base.frame.signal+1000.)
    value=replace(base,frame=pd.concat([base.frame,other]).sort_values(['timestamp','market']).reset_index(drop=True))
    config=replace(declaration(),markets=('ETH','BTC'))
    original=value.frame.copy()
    class MarketRunner:
        seed=17
        calls=0
        def predict_features(self,state,inputs,config):
            self.calls+=1
            assert inputs.frame[['timestamp','market']].equals(original[['timestamp','market']])
            for market in config.markets:
                part=inputs.frame[inputs.frame.market.eq(market)]
                assert sorted(part.signal)==sorted(original[original.market.eq(market)].signal)
            return inputs.frame[['timestamp','market']].assign(target_position=np.tanh(inputs.frame.signal/1000.))
    runner=MarketRunner()
    reference=runner.predict_features(None,value,config)
    replay=SimpleNamespace(markets=('BTC','ETH'),steps=24,delay=2,returns_bps=np.ones((24,2)))
    block=SimpleNamespace(block=0,train=np.arange(4),evaluation=np.arange(10,34),audit={})
    fold=policy_fold_importance(runner,None,value,config,reference,replay,block)
    assert fold['available'] and len(fold['scored_policy_rows'])==44
    assert fold['nonexecuted_rows']==[44,45,46,47] and runner.calls==11


def test_exposed_inputs_must_reproduce_actual_positions(tmp_path):
    code=POLICY.replace('return self.predict_features(self.feature_packet(history,times))',
        "return pd.DataFrame({'timestamp':times,'market':'ETH','target_position':0.})")
    data=market_data(32)
    times=pd.DatetimeIndex(pd.to_datetime(data.frame.available_at_ms.iloc[2:],unit='ms',utc=True))
    with StrategyRunner(candidate(tmp_path,code),tmp_path/'run',seed=17) as runner:
        config=runner.declare(); state=runner.fit(data)
        with pytest.raises(DataContractError,match='reproduce'):
            runner.predict(state,data,times,config)
