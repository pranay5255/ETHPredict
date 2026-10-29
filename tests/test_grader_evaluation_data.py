import hashlib
import json

import numpy as np
import pandas as pd
import pytest

from grader.contracts import Declaration
from grader.data import CANDLE_COLUMNS, FUNDING_COLUMNS, DataContractError, LakeLoader, LoadedDataset
from grader.evaluation_data import EvaluationData


def fixture_data(count=240):
    clock=pd.date_range('2025-09-01',periods=count,freq='h',tz='UTC')
    rng=np.random.default_rng(13)
    close=100*np.exp(np.cumsum(rng.normal(0,.003,count)))
    frame=pd.DataFrame({'timestamp_ms':clock.as_unit('ms').asi8,'timestamp':clock.tz_localize(None),
        'open':close*.999,'high':close*1.002,'low':close*.998,'close':close,'volume':100.+np.arange(count),
        'quote_asset_volume':(100.+np.arange(count))*close,'last_trade_id':np.arange(count),
        'sample_count':1,'open_raw':0.,'high_raw':0.,'low_raw':0.,'close_raw':0.,'market_id':0,'symbol':'ETH'})[CANDLE_COLUMNS]
    funding=pd.DataFrame({'timestamp_ms':clock.as_unit('ms').asi8,'timestamp':clock.tz_localize(None),
        'rate':.0008,'value':.01,'direction':'long','market_id':0,'symbol':'ETH'})[FUNDING_COLUMNS]
    candles=LoadedDataset(frame.assign(available_at_ms=frame.timestamp_ms+3600000),{'market_id':0,'synthetic':True})
    funds=LoadedDataset(funding.assign(available_at_ms=funding.timestamp_ms),{'market_id':0,'synthetic':True})
    return EvaluationData({'ETH':{'candles_1h':candles,'fundings_1h':funds}},'1h','dev',
                          int(clock.as_unit('ms').asi8[0]),int(clock.as_unit('ms').asi8[-1]+3600000))


def declaration():
    return Declaration.parse(dict(name='fixture',markets=['ETH'],resolution='1h',lookback_bars=2,
        max_horizon_bars=1,budget_tier='fast',engineered_features=[],pretrained_models=[]))


def test_last_execution_bar_is_reserved_before_tier_boundary_and_purge_is_exact(tmp_path):
    data=fixture_data()
    blocks=data.blocks(declaration())
    assert len(blocks)==5
    final=blocks[-1]
    times=data.candles.frame.timestamp_ms.to_numpy()
    assert final.evaluation[-1]==len(times)-2
    assert times[final.evaluation[-1]]+3600000<data.end_ms
    for block in blocks:
        assert block.train[-1]+3<block.evaluation[0]
        assert len(block.audit['purged_indices'])==3
        replay=data.replay(data.market_slice(np.r_[block.evaluation,block.evaluation[-1]+1]),tmp_path/f'replay-{block.block}')
        assert replay.steps==len(block.evaluation)
        assert replay.source_times_ms[-1]+3600000==times[block.evaluation[-1]+1]+3600000


def test_causal_view_replaces_full_month_with_only_already_available_rows(tmp_path):
    data=fixture_data()
    cutoff=int(data.candles.frame.available_at_ms.iloc[40])
    view=data.view(tmp_path/'view',cutoff)
    loader=LakeLoader(view)
    frame=loader.load('ETH','candles_1h','1h','2025-09-01',pd.Timestamp(cutoff,unit='ms',tz='UTC')).frame
    assert len(frame)==41 and frame.available_at_ms.le(cutoff).all()
    hashes=json.loads((view/'ETH/_manifest.json').read_text())['datasets']['candles_1h']['partitions']
    assert len(hashes)==1
    path=view/'ETH/candles_1h/2025-09.parquet'
    assert hashlib.sha256(path.read_bytes()).hexdigest()==next(iter(hashes.values()))
    data.view(view,int(data.candles.frame.available_at_ms.iloc[20]))
    assert len(pd.read_parquet(path))==21


def test_shadow_plan_has_dev_prelude_and_no_holdout_decisions():
    data=fixture_data()
    data.tier='shadow';data.start_ms=int(data.candles.frame.timestamp_ms.iloc[90])
    blocks=data.blocks(declaration())
    assert len(blocks)==5 and blocks[0].train[0]==0
    assert blocks[0].evaluation[0]==90 and blocks[-1].evaluation[-1]==238
    assert all(len(block.evaluation)>=20 for block in blocks)


def test_two_lake_shadow_loader_verifies_both_sources_and_never_opens_holdout(tmp_path,monkeypatch):
    from pathlib import Path
    from grader.tiers import load_tiers
    tiers=load_tiers();interval=3600000
    begin,end=tiers['dev'].start_ms,tiers['shadow'].end_ms
    synthetic=fixture_data((end-begin)//interval)
    offset=begin-synthetic.start_ms
    for item in synthetic.datasets['ETH'].values():
        item.frame['timestamp_ms']+=offset
        item.frame['available_at_ms']+=offset
        item.frame['timestamp']=pd.to_datetime(item.frame.timestamp_ms,unit='ms',utc=True).dt.tz_localize(None)
    public,private=tmp_path/'public',tmp_path/'private'
    for root,left,right in [(public,begin,tiers['dev'].end_ms),(private,tiers['shadow'].start_ms,end)]:
        manifest={'symbol':'ETH','market_id':0,'datasets':{}}
        for name,item in synthetic.datasets['ETH'].items():
            columns=FUNDING_COLUMNS if name=='fundings_1h' else CANDLE_COLUMNS
            selected=item.frame[(item.frame.timestamp_ms>=left)&(item.frame.timestamp_ms<right)][columns]
            if name=='fundings_1h' and root==public:
                selected=selected.iloc[1:]  # known collection starts at first payment
            target=root/'ETH'/name;target.mkdir(parents=True)
            hashes={}
            for month,part in selected.groupby(pd.to_datetime(selected.timestamp_ms,unit='ms').dt.strftime('%Y-%m')):
                path=target/f'{month}.parquet';part.to_parquet(path,index=False)
                hashes[path.name]=hashlib.sha256(path.read_bytes()).hexdigest()
            manifest['datasets'][name]={'expected_interval_ms':interval,'rows':len(selected),
                'first':pd.Timestamp(int(selected.timestamp_ms.iloc[0]),unit='ms',tz='UTC').isoformat(),
                'last':pd.Timestamp(int(selected.timestamp_ms.iloc[-1]),unit='ms',tz='UTC').isoformat(),
                'partitions':hashes}
            if root==private:
                canary=target/'2026-08.parquet';canary.write_bytes(b'forbidden synthetic holdout')
                hashes[canary.name]=hashlib.sha256(canary.read_bytes()).hexdigest()
        (root/'ETH/_manifest.json').write_text(json.dumps(manifest))
    original=Path.read_bytes
    def no_holdout(path):
        if path.name=='2026-08.parquet':
            raise AssertionError('Shadow loader attempted to open holdout')
        return original(path)
    monkeypatch.setattr(Path,'read_bytes',no_holdout)
    config=declaration()
    from dataclasses import replace
    config=replace(config,uses_dev_lake=False)
    loaded=EvaluationData.load(public,config,tier='shadow',private_root=private)
    assert len(loaded.candles.frame)==(end-begin)//interval
    assert len(loaded.candles.provenance['ETH']['sources'])==2
    assert loaded.blocks(config)[0].train[0]==0
    corrupted=private/'ETH/fundings_1h/2026-06.parquet'
    corrupted.write_bytes(original(corrupted)+b'corrupted')
    with pytest.raises(DataContractError,match='SHA-256'):
        EvaluationData.load(public,config,tier='shadow',private_root=private)
