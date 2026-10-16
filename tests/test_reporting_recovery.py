import json
from pathlib import Path

import pytest
import yaml

from src.evaluation.evidence import assess_evidence
from src.experiments.resume_reporting import resume_reporting, _immutable_outputs
from src.experiments.staged_trial import run_staged_trial
from tests.test_staged_trial import _v2_config, _write_5m_ohlcv


def test_negative_outcomes_cannot_be_alpha_just_because_dsr_exists():
    outcome = {'net_pnl': -10., 'trades': 100., 'statistics': {'dsr': {'available':True,'value':.01}, 'psr':{'available':True,'value':.02}}}
    report = {'evidence_grade':'alpha_claim_grade','final_test_use_count':1,
              'best_trial': {'metrics':{'validation':outcome,'test':outcome}}, 'pbo':{'available':True,'value':.8}}
    assessment = assess_evidence(report, {'research':{'evidence_scope':'alpha_claim_evaluation'}})
    assert assessment['grade'] == 'debugging_only'
    assert assessment['gates']['dsr_at_least_095'] is False
    assert assess_evidence(report, {'research':{'evidence_scope':'first_bounded_backtest'}})['grade'] == 'exploratory_backtest'


def test_reporting_recovery_never_retrains_or_reuses_test_and_is_idempotent(tmp_path, monkeypatch):
    data = tmp_path/'data'
    _write_5m_ohlcv(data/'raw'/'ETHUSDT-5m-lighter-fixture.csv')
    cfg = _v2_config(data,tmp_path/'runs')
    cfg['benchmark']={'enabled':False}
    cfg['research']={'evidence_scope':'first_bounded_backtest','final_test_guard':{'do_not_reuse_test':True}}
    path = tmp_path/'cfg.yml';path.write_text(yaml.safe_dump(cfg))
    result = run_staged_trial(path,smoke=True,device='cpu',allow_cpu=True)
    manifest = Path(result['run_dir'])/'manifest.json'
    before = _immutable_outputs(result)
    import src.experiments.meta_labeling_mvp as mvp
    def forbidden(*args,**kwargs):
        raise AssertionError('Reporting recovery must not fit, predict or evaluate test')
    monkeypatch.setattr(mvp,'train_base_model',forbidden)
    monkeypatch.setattr(mvp,'predict_base_model',forbidden)
    monkeypatch.setattr(mvp,'_score_trial_on_test',forbidden)
    recovered = resume_reporting(manifest)
    assert recovered['evidence_grade'] == 'debugging_only'
    assert recovered['reporting_recovery']['test_evaluations_added'] == 0
    assert before == _immutable_outputs(recovered)
    assert resume_reporting(manifest)['reporting_recovery'] == recovered['reporting_recovery']
    protected = Path(next(iter(before)))
    protected.write_bytes(b'tampered')
    with pytest.raises(ValueError,match='Protected outcomes changed'):
        resume_reporting(manifest)
