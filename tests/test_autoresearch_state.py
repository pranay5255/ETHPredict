from copy import deepcopy
import csv
import json
import math
import os
from pathlib import Path
import subprocess
import sys

import pytest

from autoresearch.decisions import decision, monitor
from autoresearch.journal import Journal


def report(score=0., *, status='accepted', delta=.01):
    return {'score': score, 'status': status, 'uncertainty': {'standard_error': delta},
            'metrics': [{'blocks': [None]*5} for _ in range(2)],
            'integrity': [{'name': 'causality', 'passed': True}]}


def test_initial_measured_zero_is_control_not_an_improvement():
    control = report(status='rejected')
    assert decision(control)['decision'] == 'initialize'
    control['integrity'][0]['passed'] = False
    assert decision(control)['decision'] == 'discard'


@pytest.mark.parametrize('score,expected', [(.5, 'discard'), (.51, 'discard'), (.5100001, 'keep')])
def test_keep_is_strictly_above_rescored_incumbent_plus_delta(score, expected):
    assert decision(report(score), report(.5))['decision'] == expected


@pytest.mark.parametrize('delta', [None, -1, math.nan, math.inf, True])
def test_missing_or_invalid_uncertainty_cannot_keep(delta):
    assert decision(report(.9, delta=delta), report(.1))['decision'] == 'discard'


def test_economic_rejection_and_perfect_qualitative_ratings_cannot_rescue_a_tie():
    candidate = report(.5, status='rejected')
    candidate['qualitative'] = {'hypothesis': 1, 'simplicity': 1, 'mechanism_for_constants': 1,
                                'no_hardcoded_regimes': 1}
    assert decision(candidate, report(.1))['decision'] == 'discard'
    candidate['status'] = 'accepted'
    assert decision(candidate, report(.5))['decision'] == 'discard'


def test_three_increases_at_ten_trial_intervals_pause_and_flat_gap_resets_streak():
    state = monitor(None, trial_count=1, dev_score=.5, hidden_score=.5)
    for count, hidden in [(10, .4), (20, .3)]:
        state = monitor(state, trial_count=count, dev_score=.5, hidden_score=hidden)
        assert not state['alert']
    unchanged = monitor(state, trial_count=29, dev_score=.9, hidden_score=0.)
    assert unchanged == state
    state = monitor(state, trial_count=30, dev_score=.5, hidden_score=.2)
    assert state['alert'] and state['increases'] == 3
    other = monitor(None, trial_count=1, dev_score=.5, hidden_score=.5)
    other = monitor(other, trial_count=10, dev_score=.5, hidden_score=.4)
    other = monitor(other, trial_count=20, dev_score=.5, hidden_score=.4)
    assert other['increases'] == 0 and not other['alert']


def attempt(journal, identity):
    journal.append('attempt', trial_id=identity, commit='a'*40, description='fixture',
        hypothesis_sha256='b'*64, candidate_sha256='c'*64)


def test_append_only_journal_counts_crash_and_interrupted_attempt_without_invented_sharpe(tmp_path):
    journal = Journal(tmp_path/'private', tmp_path/'public')
    attempt(journal, 'first')
    prefix = journal.path.read_bytes()
    journal.append('result', trial_id='first', status='crash', score=0., pre_gate_score=None,
        sharpe_per_period=None, report_sha256=None, report_path=None, trackio_run_id='run-id')
    attempt(journal, 'interrupted')
    assert journal.path.read_bytes().startswith(prefix)
    assert journal.read()['count'] == 2 and journal.read()['sharpes'] == []
    journal.publish()
    assert (tmp_path/'public/ledger.jsonl').read_bytes() == journal.path.read_bytes()
    with (tmp_path/'public/results.tsv').open() as stream:
        rows = list(csv.DictReader(stream, delimiter='\t'))
    assert [row['status'] for row in rows] == ['crash', 'interrupted']
    assert (tmp_path/'public/ledger.jsonl').stat().st_mode & 0o777 == 0o444
    before = journal.path.read_bytes()
    with pytest.raises(Exception):
        attempt(journal, 'first')
    assert journal.path.read_bytes() == before


def test_private_history_cannot_be_published(tmp_path):
    journal = Journal(tmp_path/'hidden')
    attempt(journal, 'hidden')
    with pytest.raises(ValueError, match='private-only'):
        journal.publish()


def test_installed_trackio_persists_only_whitelisted_dev_metrics(tmp_path):
    payload = {'trial_id': 'f'*32, 'commit': 'a'*40, 'status': 'rejected', 'score': 0.,
               'pre_gate_score': .1, 'worker_seconds': 1., 'peak_vram_bytes': 0}
    source = tmp_path/'input.json'
    receipt = tmp_path/'receipt.json'
    source.write_text(json.dumps(payload))
    environment = {**os.environ, 'TRACKIO_DIR': str(tmp_path/'trackio'), 'HF_HUB_OFFLINE': '1'}
    process = subprocess.run([sys.executable, '-m', 'autoresearch.telemetry', str(source), str(receipt)],
        env=environment, capture_output=True, text=True, timeout=30)
    assert process.returncode == 0, process.stdout + process.stderr
    saved = json.loads(receipt.read_text())
    assert saved['project'] == 'ethpredict-autoresearch' and saved['delivery'] == 'local_verified'
    assert saved['metrics']['dev.score'] == 0.
    assert all(name.startswith('dev.') for name in saved['metrics'])
