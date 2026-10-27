from pathlib import Path
import json

import jsonschema
import numpy as np

from grader.baselines import materialize
from grader.artifacts import verify_artifacts
from grader.checks import CheckResult
from grader.grade import _evaluate
from grader.report import load_evidence, read_report
from test_grader_evaluation_data import fixture_data


def test_full_flat_grade_checks_two_seeds_all_blocks_and_records_zero_score(tmp_path,monkeypatch):
    import grader.grade as grading
    from grader.runner import StrategyRunner
    from grader.grade import data_identity
    runners=[]
    class TrackingRunner(StrategyRunner):
        def __init__(self,*args,**kwargs):
            super().__init__(*args,**kwargs);runners.append(self)
    monkeypatch.setattr(grading,'StrategyRunner',TrackingRunner)
    def baseline_provider(declaration,data):
        assert runners[0].process.poll() is not None and runners[0].guard is None
        return {'trial_count':6,'trial_sharpes':[-.2,.1],'best_net_by_seed':{'17':0.,'29':0.},
            'grading_fingerprint':'synthetic-fixture','data_sha256':data_identity(data)['sha256'],
            'markets':['ETH'],'resolution':'1h','tier':'dev','winner':'flat'}
    source=materialize('flat',tmp_path/'source',markets=['ETH'],resolution='1h')
    result=_evaluate(source,tmp_path/'grade',lambda declaration:fixture_data(),mode='fixture',tier='dev',
        frozen_check=CheckResult('grader_hash',True,{'synthetic_fixture':True}),
        baseline_provider=baseline_provider,
        compute_uncertainty=False,device='cpu')
    report=read_report(tmp_path/'grade/grade_report.json')
    manifest=verify_artifacts(tmp_path/'grade',report['artifacts']['manifest'])
    paths={row['path'] for row in manifest['files']}
    assert 'seed-17/block-1/positions.parquet' in paths
    assert 'worker/private-worker.trace' in paths
    assert 'baseline_bank.json' in paths
    assert any(name.endswith('/state.pkl') for name in paths)
    assert report['status']=='rejected' and report['score']==0.
    assert report['pre_gate_score']==.1
    assert len(report['metrics'])==2 and all(len(row['blocks'])==5 for row in report['metrics'])
    assert all(row['trial_count']==7 for row in report['metrics'])
    assert len([row for row in report['integrity'] if row['name']=='causality' and row['passed']])==10
    assert all(len(row['details']['cutoffs_ms'])==20 for row in report['integrity'] if row['name']=='causality')
    assert len(report['resources']['calls'])==225
    assert report['resources']['wall_seconds']<report['resources']['wall_limit_seconds']
    assert any(row['check']=='economic:enough_trades' for row in report['failures'])
    for paths,blocks in zip(report['artifacts']['seed_blocks'],result.seed_blocks):
        for path,expected in zip(paths,blocks):
            actual=load_evidence(path)
            np.testing.assert_array_equal(actual.cashflows['net_bps'],expected.cashflows['net_bps'])


def test_unfrozen_production_rejects_before_import_or_data_read(tmp_path):
    def forbidden(declaration):
        raise AssertionError('production data must not be loaded before freeze')
    result=_evaluate(tmp_path/'absent',tmp_path/'grade',forbidden,mode='production',tier='dev',
        frozen_check=CheckResult('grader_hash',False,{'unfrozen':True}),device='cpu')
    report=read_report(tmp_path/'grade/grade_report.json')
    assert report['score']==0. and not report['metrics'] and result.seed_blocks==[]
    assert any(row['check']=='grader_hash' for row in report['failures'])


def test_unknown_report_envelope_fields_are_rejected(tmp_path):
    result=_evaluate(tmp_path/'absent',tmp_path/'grade',None,mode='production',tier='dev',
        frozen_check=CheckResult('grader_hash',False,{}),device='cpu')
    from grader.report import write_report
    import pytest
    with pytest.raises(jsonschema.ValidationError):
        write_report({**result.report,'unrecognized':True},tmp_path/'invalid.json')
    from copy import deepcopy
    value=deepcopy(result.report)
    value['artifacts']['manifest']['unrecognized']=True
    with pytest.raises(jsonschema.ValidationError):
        write_report(value,tmp_path/'invalid-artifact.json')


def test_report_reader_rejects_nonfinite_numbers_at_any_depth(tmp_path):
    import pytest
    from grader.data import DataContractError
    result=_evaluate(tmp_path/'absent',tmp_path/'grade',None,mode='production',tier='dev',
        frozen_check=CheckResult('grader_hash',False,{}),device='cpu')
    result.report['resources']['nested']={'value':'numeric-sentinel'}
    content=json.dumps(result.report)
    for token in ['NaN','Infinity','1e1000']:
        path=tmp_path/f'{token}.json'
        path.write_text(content.replace('"numeric-sentinel"',token))
        with pytest.raises(DataContractError,match='Nonfinite'):
            read_report(path)


def test_full_engineered_grade_uses_actual_policy_importance_for_both_seeds(tmp_path):
    import json
    source=materialize('momentum_1h',tmp_path/'source',markets=['ETH'],resolution='1h')
    result=_evaluate(source,tmp_path/'grade',lambda declaration:fixture_data(),mode='fixture',tier='dev',
        frozen_check=CheckResult('grader_hash',True,{'synthetic_fixture':True}),
        baseline_count=6,baseline_net_by_seed={'17':0.,'29':0.},compute_uncertainty=False,device='cpu')
    report=read_report(tmp_path/'grade/grade_report.json')
    assert len(result.seed_blocks)==2 and all(len(blocks)==5 for blocks in result.seed_blocks)
    assert all(check['passed'] for check in report['integrity'])
    assert len(report['feature_importance'])==2
    for seed in report['feature_importance']:
        importance=seed['report']
        assert importance['status']=='completed' and len(importance['folds'])==5
        assert {row['feature'] for row in importance['summary']}=={'returns','diagnostic_noise_0','diagnostic_noise_1'}
        for block in range(1,6):
            saved=json.loads((tmp_path/f'grade/seed-{seed["seed"]}/block-{block}/feature_importance.json').read_text())
            assert saved['available'] and saved['permutation_scope'].startswith('within-market')
    assert sum(call['operation']=='predict_features' for call in report['resources']['calls'])==100
    assert all('credible_features' in row['gates'] for row in report['economics'])
