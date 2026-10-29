import json

import pytest

from grader.data import DataContractError
from grader.history import read_trials


def attempt(identity):
    return dict(version=1,event='attempt',trial_id=identity,timestamp_utc='2026-10-28T00:00:00Z',commit=identity,
                description='fixture',hypothesis_sha256='a'*64,candidate_sha256='b'*64)


def result(identity,**overrides):
    return dict(version=1,event='result',trial_id=identity,timestamp_utc='2026-10-28T00:00:00Z',status='rejected',
        score=0.,pre_gate_score=.1,sharpe_per_period=None,report_sha256='c'*64,
        report_path='report.json',trackio_run_id=None,**overrides)


def decision(identity,**overrides):
    value=dict(version=1,event='decision',trial_id=identity,timestamp_utc='2026-10-28T00:00:00Z',
        decision='discard',incumbent_commit='incumbent',incumbent_rescored_score=.2,
        candidate_score=0.,delta=.01,qualitative=None)
    return {**value,**overrides}


def test_crash_interrupted_and_completed_attempts_all_count_without_invented_sharpes(tmp_path):
    path=tmp_path/'ledger.jsonl'
    rows=[attempt('good'),dict(result('good'),sharpe_per_period=.2),
          attempt('crash'),dict(result('crash'),status='crash'),attempt('interrupted')]
    path.write_text('\n'.join(json.dumps(row) for row in rows)+'\n')
    history=read_trials(path)
    assert history['count']==3 and history['sharpes']==[.2]


@pytest.mark.parametrize('rows',[
    [result('unknown')],[attempt('twice'),attempt('twice')],
    [attempt('duplicate-result'),result('duplicate-result'),result('duplicate-result')],
    [dict(attempt('unknown-field'),secret_score=1.)],
    [attempt('nan'),dict(result('nan'),sharpe_per_period=float('nan'))],
])
def test_malformed_accounting_fails_closed(tmp_path,rows):
    path=tmp_path/'ledger.jsonl';path.write_text('\n'.join(json.dumps(row) for row in rows))
    with pytest.raises(DataContractError):read_trials(path)


@pytest.mark.parametrize('rows',[
    [attempt('trial'),decision('trial')],
    [attempt('trial'),dict(result('trial'),score=.1)],
    [attempt('trial'),dict(result('trial'),score=None)],
    [attempt('trial'),dict(result('trial'),pre_gate_score=1.1)],
    [dict(attempt('trial'),version=True)],
    [dict(attempt('trial'),timestamp_utc='2026-10-28T00:00:00')],
    [dict(attempt('trial'),candidate_sha256='unknown')],
    [attempt('trial'),result('trial'),decision('trial',delta=-.1)],
    [attempt('trial'),result('trial'),decision('trial',candidate_score=.9)],
    [attempt('trial'),result('trial'),decision('trial',decision='keep',delta=0.)],
    [attempt('trial'),result('trial'),decision('trial',decision='unknown')],
    [attempt('trial'),result('trial'),decision('trial',qualitative={'shadow_score':.9})],
])
def test_invalid_event_values_order_and_keep_decisions_fail_closed(tmp_path,rows):
    path=tmp_path/'ledger.jsonl';path.write_text('\n'.join(json.dumps(row) for row in rows))
    with pytest.raises(DataContractError):read_trials(path)


def test_keep_uses_rescored_incumbent_and_cannot_be_rescued_by_qualitative_ratings(tmp_path):
    ratings={name:1. for name in ['hypothesis','simplicity','mechanism_for_constants','no_hardcoded_regimes']}
    qualitative={**ratings,'evidence':{name:'fixture source:1' for name in ratings}}
    path=tmp_path/'ledger.jsonl'
    rows=[attempt('trial'),dict(result('trial'),status='accepted',score=.4),
          decision('trial',decision='keep',incumbent_rescored_score=.2,candidate_score=.4,delta=.1,qualitative=qualitative)]
    path.write_text('\n'.join(json.dumps(row) for row in rows))
    assert read_trials(path)['count']==1
    rows[-1]['delta']=.2
    path.write_text('\n'.join(json.dumps(row) for row in rows))
    with pytest.raises(DataContractError):read_trials(path)


def test_crash_can_record_explicitly_unavailable_artifacts_and_uncertainty(tmp_path):
    rows=[attempt('trial'),dict(result('trial'),status='crash',pre_gate_score=None,
        report_path=None,report_sha256=None),decision('trial',incumbent_rescored_score=None,delta=None)]
    path=tmp_path/'ledger.jsonl';path.write_text('\n'.join(json.dumps(row) for row in rows))
    assert read_trials(path)['count']==1 and read_trials(path)['sharpes']==[]


def test_numeric_overflow_is_rejected_even_in_nested_metadata(tmp_path):
    rows=[attempt('trial'),result('trial'),decision('trial',qualitative={'value':'overflow'})]
    path=tmp_path/'ledger.jsonl'
    path.write_text('\n'.join(json.dumps(row) for row in rows).replace('"overflow"','1e1000'))
    with pytest.raises(DataContractError):read_trials(path)
