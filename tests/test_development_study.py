from copy import deepcopy
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

from src.experiments.development_study import opportunity_candidates, run_study
from src.experiments.local_batch import prepare_batch
from src.experiments.meta_labeling_mvp import build_multi_horizon_lighter_dataset
from tests.test_staged_trial import _v2_config, _write_5m_ohlcv


def recipe(tmp_path):
    source=tmp_path/'source';path=source/'raw'/'ETHUSDT-5m-lighter-fixture.csv'
    _write_5m_ohlcv(path,rows=400,step=.002)
    raw=pd.read_csv(path,header=None)
    raw[5]=10+np.sin(np.arange(len(raw)))+np.arange(len(raw))%7
    raw.to_csv(path,header=False,index=False)
    cfg=_v2_config(source,tmp_path/'runs')
    cfg['targets']['horizons']={'next_hour':{'bars':2}}
    cfg['labels']['meta_triple_barrier']['vertical_barriers']={'next_hour':2}
    cfg['validation'].update(method='exact_span_walk_forward',folds=3)
    cfg['training'].update(epochs=2,ensemble_seeds=[17],target_standardization='standard')
    cfg['alpha_backtest'].update(horizon='next_hour',edge_threshold_bps=0.,cooldown_bars=2)
    cfg['alpha_backtest']['meta_threshold_selection']={'enabled':False}
    cfg['model']['meta_labeler'].update(type='logistic',calibration={'method':'none'},selection={'enabled':False})
    cfg['research']={'final_test_guard':{'evaluate':False}}
    cfg['tracking']={'trackio':{'enabled':False,'allow_local_debug_without_trackio':True}}
    cfg['development_study']={'excluded_from':'2026-05-30','max_configurations':6,'seeds':[17],
        'forecast_models':['zero','mean','linear','lstm'],'opportunity':{'lookback_bars':12,'volume_z_min':[0.]},
        'economic_gate':{'enabled':True}}
    cfg['local_batch']={'source_candles':str(path.relative_to(source)),'rows':380,'warmup_bars':20,
                         'source_end_exclusive':'2026-05-30T00:00:00Z'}
    template=tmp_path/'template.yml';template.write_text(yaml.safe_dump(cfg))
    return source,cfg,template


def test_rule_uses_prior_completed_bars_and_has_no_fake_expected_return(tmp_path):
    _,cfg,_=recipe(tmp_path);dataset=build_multi_horizon_lighter_dataset(cfg)
    ids=np.arange(30,70)
    baseline=opportunity_candidates(dataset,ids,cfg,0.)
    assert baseline.expected_edge_bps.eq(0).all()
    assert baseline.edge_estimate_status.eq('not_estimated_rule_only').all()
    assert 'pred_return' not in baseline
    assert (pd.to_datetime(baseline.information_cutoff_timestamp)<pd.to_datetime(baseline.bar_close_timestamp)).all()
    changed=deepcopy(dataset)
    boundary=int(dataset['samples'].iloc[70].sample_index)
    changed['price_path'].loc[boundary:,['close','high','low','volume']]*=100
    result=opportunity_candidates(changed,ids,cfg,0.)
    cols=['side','is_candidate','opportunity_volume_z','opportunity_range_width','opportunity_breakout_distance']
    pd.testing.assert_frame_equal(baseline[cols],result[cols])


def test_three_block_development_study_never_calls_final_test(tmp_path,monkeypatch):
    source,cfg,template=recipe(tmp_path)
    saved=prepare_batch(template,source,tmp_path/'frozen')
    import src.experiments.meta_labeling_mvp as mvp
    def forbidden(*args,**kwargs):
        raise AssertionError('Development study cannot evaluate final test')
    monkeypatch.setattr(mvp,'_score_trial_on_test',forbidden)
    monkeypatch.setattr(mvp,'_record_final_test_evaluation',forbidden)
    output=tmp_path/'study';report=run_study(saved,output,device='cpu',allow_cpu=True)
    assert report['final_test_evaluations']==0
    assert len(report['forecast_comparisons'])==12
    assert {row['model'] for row in report['forecast_comparisons']}=={'zero','mean','linear','lstm'}
    assert all(row['checkpoint_roundtrip_exact'] for row in report['forecast_comparisons'])
    assert len(report['opportunity_trials'])==2
    assert all(len(row['blocks'])==3 and row['metrics']['test']=={} for row in report['opportunity_trials'])
    assert not list(output.rglob('*ledger*'))
    assert (output/'report.md').is_file()
    with pytest.raises(FileExistsError):
        run_study(saved,output,device='cpu',allow_cpu=True)


def test_exclusion_and_disabled_test_are_enforced_before_any_fit(tmp_path):
    source,cfg,template=recipe(tmp_path)
    cfg['research']['final_test_guard']['evaluate']=True
    template.write_text(yaml.safe_dump(cfg));saved=prepare_batch(template,source,tmp_path/'frozen')
    with pytest.raises(ValueError,match='evaluate=false'):
        run_study(saved,tmp_path/'study',device='cpu',allow_cpu=True)
