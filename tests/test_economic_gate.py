from copy import deepcopy
from pathlib import Path

import torch

from src.evaluation.economic_gate import economic_gate
from src.experiments.meta_labeling_mvp import _apply_selected_test_evaluations


def fixture():
    metrics={'net_pnl':10.,'trades':90}
    blocks=[{'net_pnl':value,'trades':30} for value in [5.,-1.,6.]]
    stress={'rows':[{'cost_multiple':2.,'net_pnl':1.}]}
    importance={'status':'completed','summary':[{'category':'noise_control','negative_log_loss_drop':.001},{'category':'positive','feature':'opportunity_volume_z','negative_log_loss_drop':.1,'folds':3}], 'noise_control_failures':[]}
    settings={'enabled':True,'feature_prefix':'opportunity_'}
    return metrics,blocks,stress,importance,settings


def test_gate_requires_cross_block_economics_stress_and_real_features():
    args=fixture();assert economic_gate(*args)['passed']
    for i,key,value in [(0,'net_pnl',-1.),(0,'trades',0),(2,'rows',[]),(3,'noise_control_failures',['diagnostic_noise_0'])]:
        changed=deepcopy(args);changed[i][key]=value;assert not economic_gate(*changed)['passed']
    changed=deepcopy(args);changed[1][:]=[{'net_pnl':10.,'trades':90}]
    assert not economic_gate(*changed)['passed']
    changed=deepcopy(args);changed[3]['summary'][1]['negative_log_loss_drop']=0.
    assert not economic_gate(*changed)['passed']


def test_economic_rejection_blocks_test_before_context_or_ledger(tmp_path,monkeypatch):
    import src.experiments.meta_labeling_mvp as mvp
    def forbidden(*args,**kwargs):
        raise AssertionError('Rejected economics must never reach final-test scorer')
    monkeypatch.setattr(mvp,'_score_trial_on_test',forbidden)
    trial={'trial_id':'classification_winner','status':'completed','metrics':{'validation':{'net_pnl':-1.,'trades':100},'test':{}}}
    selection={'raw_best_trial_id':'classification_winner','best_trading_trial_id':'classification_winner'}
    config={'research':{'economic_gate':{'enabled':True},'final_test_guard':{'evaluate':True,'evaluate_selection_roles':['trade_qualified_best']}}}
    result=_apply_selected_test_evaluations([trial],selection,{},config=config,run_dir=tmp_path,smoke=False,device=torch.device('cpu'))
    assert result[0]['test_evaluation']['status']=='economic_rejected'
    assert result[0]['metrics']['test']=={} and result[0]['final_test_evaluation'] is None
    assert not list(tmp_path.rglob('*ledger*'))


def test_direct_scorer_rejects_before_recording_a_test_use(tmp_path,monkeypatch):
    import pytest
    import src.experiments.meta_labeling_mvp as mvp
    def forbidden(*args,**kwargs): raise AssertionError('No ledger write before economic acceptance')
    monkeypatch.setattr(mvp,'_record_final_test_evaluation',forbidden)
    context={'config':{'research':{'economic_gate':{'enabled':True}}},'dataset':{},'splits':{}}
    with pytest.raises(ValueError,match='Economic acceptance failed'):
        mvp._score_trial_on_test({'trial_id':'loser','metrics':{'validation':{'net_pnl':-1.}}},context,
                                run_dir=tmp_path,smoke=False,device=torch.device('cpu'),selection_roles=['raw_best'])


def test_legacy_benchmark_cannot_bypass_economic_acceptance(tmp_path,monkeypatch):
    import pytest
    import src.experiments.forecast_benchmark as benchmark
    def forbidden(*args,**kwargs): raise AssertionError('No split/test access before acceptance')
    monkeypatch.setattr(benchmark,'build_validation_splits',forbidden)
    cfg={'research':{'economic_gate':{'enabled':True},'final_test_guard':{'evaluate':True}}}
    with pytest.raises(ValueError,match='Economic-gated research'):
        benchmark.run_forecast_benchmark_from_dataset(cfg,{},tmp_path)
