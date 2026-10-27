from pathlib import Path

import jsonschema
import numpy as np

from grader.baselines import materialize
from grader.checks import CheckResult
from grader.grade import _evaluate
from grader.report import load_evidence, read_report
from test_grader_evaluation_data import fixture_data


def test_full_flat_grade_checks_two_seeds_all_blocks_and_records_zero_score(tmp_path):
    source=materialize('flat',tmp_path/'source',markets=['ETH'],resolution='1h')
    result=_evaluate(source,tmp_path/'grade',lambda declaration:fixture_data(),mode='fixture',tier='dev',
        frozen_check=CheckResult('grader_hash',True,{'synthetic_fixture':True}),
        baseline_count=6,baseline_sharpes=[-.2,.1],baseline_net_by_seed={'17':0.,'29':0.},
        compute_uncertainty=False,device='cpu')
    report=read_report(tmp_path/'grade/grade_report.json')
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
