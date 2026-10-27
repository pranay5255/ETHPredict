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
