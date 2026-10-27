import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

from grader.data import DataContractError, utc_ms
from grader.splits import overlaps, purge_training_spans
from grader.tiers import build_walk_forward_blocks, dev_blocks, load_tiers


def test_literal_half_open_tiers_and_monthly_internal_boundaries():
    tiers = load_tiers()
    assert tiers['dev'].start_ms == utc_ms('2025-06-04')
    assert tiers['dev'].end_ms == utc_ms('2026-05-01')
    assert tiers['shadow'].end_ms == utc_ms('2026-08-01')
    assert tiers['holdout'].end_ms == utc_ms('2026-10-16')
    for a, b in [('dev', 'shadow'), ('shadow', 'holdout')]:
        boundary = tiers[a].end_ms
        assert tiers[a].contains([boundary-1,boundary]).tolist() == [True,False]
        assert tiers[b].contains([boundary-1,boundary]).tolist() == [False,True]


def test_growing_training_blocks_are_contiguous_purged_and_nonoverlapping():
    clock = pd.date_range('2026-01-01',periods=600,freq='h',tz='UTC')
    horizon, lookback = 4, 7
    blocks = build_walk_forward_blocks(clock,resolution='1h',max_horizon_bars=horizon,lookback_bars=lookback)
    assert len(blocks) == 5
    assert np.array_equal(np.concatenate([b.evaluation for b in blocks]),np.arange(100,600))
    for i, block in enumerate(blocks):
        assert block.train[0] == 0
        assert block.train[-1] == block.evaluation[0] - lookback - horizon - 1
        assert len(block.audit['purged_indices']) == lookback+horizon
        assert block.audit['embargo_bars'] == lookback+horizon
        assert block.audit['embargoed_indices'] == []
        assert block.audit['overlap_count_after'] == 0
        assert not overlaps(block.train-lookback,block.train+horizon,
                            block.evaluation-lookback,block.evaluation+horizon).any()
        assert block.history[0] == block.evaluation[0]-lookback
        assert block.history[-1] == block.evaluation[-1]
        if i:
            assert set(blocks[i-1].train) < set(block.train)
            assert blocks[i-1].evaluation[-1]+1 == block.evaluation[0]
            assert block.audit['evaluation_start_ms'] == blocks[i-1].audit['evaluation_end_exclusive_ms']


def test_vendored_embargo_removes_future_side_population():
    anchors=np.arange(100)
    samples=pd.DataFrame({'label_span_start_idx':anchors,'label_span_end_idx':anchors+2})
    evaluation=np.arange(40,50)
    proposed=np.r_[np.arange(40),np.arange(50,100)]
    train,audit=purge_training_spans(samples,proposed,evaluation,
        {'embargo_mode':'sequence_plus_horizon'},max_horizon=2,sequence_length=3)
    assert audit['embargo_bars'] == 5
    assert audit['embargoed_indices'] == [52,53,54,55,56]
    assert not set(train)&set(audit['purged_indices']+audit['embargoed_indices'])
    assert min(train[train>49]) == 57


@pytest.mark.parametrize('resolution', ['1m','5m','1h'])
def test_real_dev_layout_stays_within_tier(resolution):
    blocks=dev_blocks(resolution,max_horizon_bars=12,lookback_bars=24)
    tier=load_tiers()['dev']
    assert blocks[0].audit['training_start_ms'] == tier.start_ms
    assert blocks[-1].audit['evaluation_end_exclusive_ms'] == tier.end_ms
    assert all(tier.start_ms <= b.audit['evaluation_start_ms'] < b.audit['evaluation_end_exclusive_ms'] <= tier.end_ms for b in blocks)


@pytest.mark.parametrize('mutation', ['overlap','gap','unknown','boundary','layout'])
def test_invalid_tier_configuration_fails_closed(tmp_path,mutation):
    path=Path(__file__).parents[1]/'grader/tiers.yaml'
    config=yaml.safe_load(path.read_text())
    if mutation=='overlap': config['tiers']['shadow']['start']='2026-04-01'
    if mutation=='gap': config['tiers']['shadow']['start']='2026-06-01'
    if mutation=='unknown': config['tiers']['dev']['unknown']=True
    if mutation=='boundary':
        config['tiers']['dev']['end']='2026-05-02'
        config['tiers']['shadow']['start']='2026-05-02'
    if mutation=='layout': config['block_layout']['evaluation_blocks']=4
    target=tmp_path/'bad.yaml';target.write_text(yaml.safe_dump(config))
    with pytest.raises(DataContractError): load_tiers(target)


def test_duplicate_tier_fields_fail_closed(tmp_path):
    original=(Path(__file__).parents[1]/'grader/tiers.yaml').read_text()
    path=tmp_path/'duplicate.yaml'
    path.write_text(original.replace('version: 1','version: 1\nversion: 1'))
    with pytest.raises(DataContractError,match='Duplicate'): load_tiers(path)


@pytest.mark.parametrize('kind',['gap','duplicate','reversed','off_grid','short','horizon','lookback'])
def test_invalid_clock_or_declarations_fail_closed(kind):
    clock=pd.date_range('2026-01-01',periods=120,freq='h',tz='UTC')
    kwargs={'resolution':'1h','max_horizon_bars':2,'lookback_bars':3}
    if kind=='gap': clock=clock.delete(10)
    if kind=='duplicate': clock=clock.insert(10,clock[10])
    if kind=='reversed': clock=clock[::-1]
    if kind=='off_grid': clock=clock+pd.Timedelta(minutes=1)
    if kind=='short': clock=clock[:10]
    if kind=='horizon': kwargs['max_horizon_bars']=0
    if kind=='lookback': kwargs['lookback_bars']=True
    with pytest.raises(DataContractError): build_walk_forward_blocks(clock,**kwargs)


def test_vendored_split_source_is_byte_identical():
    root=Path(__file__).parents[1]
    sources=json.loads((root/'grader/vendor_sources.json').read_text())
    record=sources['grader/splits.py']
    assert hashlib.sha256((root/record['source']).read_bytes()).hexdigest()==record['sha256']
    assert hashlib.sha256((root/'grader/splits.py').read_bytes()).hexdigest()==record['sha256']


def test_deny_rules_are_present():
    rules=json.loads((Path(__file__).parents[1]/'.claude/settings.json').read_text())['permissions']['deny']
    # Claude's Edit rule covers Edit and Write; Write(path) rules never match.
    # Absolute filesystem paths require //, while / anchors at the project.
    assert {'Edit(/grader/**)','Read(//srv/ethpredict-grader/**)','Read(/autoresearch/shadow*)'} <= set(rules)
