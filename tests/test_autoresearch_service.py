"""Controller fixtures, not substitutes for the human-provisioned private run.

Scoring/evidence/schema/manifest operations are real. Worker execution is
replaced with explicitly labelled synthetic reports to exercise transactions.
"""
from copy import deepcopy
from dataclasses import asdict
import fcntl
import json
import os
from pathlib import Path
import socket
import threading
import time

import numpy as np
import pytest

from autoresearch.journal import Journal, write_json
from autoresearch.protocol import request, source_proof
from autoresearch.service import Controller, dispatch, serve
from grader.artifacts import capture_artifacts
from grader.checks import CheckResult
from grader.contracts import Declaration
from grader.grade import EvaluationResult, _evaluate, rescore
from grader.report import read_report, save_evidence, write_report
from grader.score import block_evidence
from test_autoresearch_protocol import repository, git, commit
from test_grader_score import execution_fixture


PRIVATE_CANARY = 'private-shadow-value-must-never-be-exported-6e86b7'


def fixture_report(directory, identity, tier, *, crash=False):
    result = _evaluate(directory/'unused', directory, lambda _: None, mode='fixture', tier=tier,
        frozen_check=CheckResult('grader_hash', False, {'profile': 'controller fixture; no worker'}),
        compute_uncertainty=False, device='cpu')
    report = result.report
    report['trial_id'] = identity
    report['strategy']['declaration'] = asdict(Declaration('controller-fixture', ('ETH',), '1h', 1, 1,
        'fast', (), (), False))
    report['integrity'] = [{'name': name, 'passed': True, 'details': {'profile': 'fixture'}}
        for name in ['grader_hash', 'causality', 'determinism', 'schema', 'tier_isolation', 'budget', 'artifact_integrity']]
    report['failures'] = []
    if crash:
        report['integrity'].append({'name': 'fit', 'passed': False, 'details': {'profile': 'fixture'}})
        report['failures'] = [{'check': 'crash', 'message': 'deliberate synthetic worker crash'}]
    else:
        evidence = []
        for number in range(1, 6):
            replay, _ = execution_fixture(directory, number)
            evidence.append(block_evidence(replay, np.zeros((replay.steps, 1)), block=number))
        result = EvaluationResult(report, [deepcopy(evidence), deepcopy(evidence)])
        rescore(result, trial_sharpes=[], trial_count=7, baseline_net_by_seed={'17': 0., '29': 0.},
            compute_uncertainty=False)
        for seed, blocks in zip([17, 29], result.seed_blocks):
            report['artifacts']['seed_blocks'].append([save_evidence(block,
                directory/f'seed-{seed}/block-{block.block}/evidence') for block in blocks])
        if tier == 'shadow':
            report['pre_gate_score'] = .789123456789
            for metric in report['metrics']:
                metric['daily_sharpe_per_period'].update(available=True, value=7.123456789, reason=None)
    if tier == 'shadow':
        (directory/'private-canary.txt').write_text(PRIVATE_CANARY)
    (directory/'artifact_manifest.json').unlink()
    report['artifacts']['manifest'] = capture_artifacts(directory)
    write_report(report, directory/'grade_report.json')
    return read_report(directory/'grade_report.json')


@pytest.fixture
def setup(repository, tmp_path, monkeypatch):
    repo, protected = repository
    git(repo, 'tag', 'grader-v1')
    gpu_lock = tmp_path/'gpu.lock'
    gpu_lock.touch()
    config = {'version': 1, 'grader_root': str(repo), 'private_root': str(tmp_path/'private'),
        'public_root': str(tmp_path/'public'), 'dev_lake': str(tmp_path/'dev'),
        'private_lake': str(tmp_path/'hidden'), 'model_paths': {}, 'protected_entries': protected,
        'agent_uid': os.getuid(), 'gpu_lock': str(gpu_lock)}
    # DSR/count calculations remain real. Bootstrap's numerical implementation
    # is covered by the frozen grader tests, not repeated in transaction tests.
    import autoresearch.decisions as decisions
    real_rescore = decisions.rescore
    def quick_rescore(*args, **kwargs):
        return real_rescore(*args, **kwargs, compute_uncertainty=False)
    monkeypatch.setattr(decisions, 'rescore', quick_rescore)

    class FixtureController(Controller):
        def __init__(self, config):
            self.rescored_counts = []
            super().__init__(config, production=False)

        def _grade(self, identity, tier, candidate):
            other = os.open(config['gpu_lock'], os.O_RDWR)
            try:
                with pytest.raises(BlockingIOError):
                    fcntl.flock(other, fcntl.LOCK_EX | fcntl.LOCK_NB)
            finally:
                os.close(other)
            crash = 'FIXTURE_CRASH' in (candidate/'strategy/strategy.py').read_text()
            return fixture_report(self._path(identity)/tier, identity, tier, crash=crash)

        def _bank(self, identity, tier):
            return {'trial_count': 6, 'trial_sharpes': [], 'best_net_by_seed': {'17': 0., '29': 0.}}

        def _rescore(self, identity, tier):
            result = super()._rescore(identity, tier)
            if result['metrics']:
                self.rescored_counts.append((tier, result['metrics'][0]['trial_count']))
            return result

        def _telemetry(self, identity, report, status):
            write_json(self._path(identity)/'trackio.json', {'project': 'ethpredict-autoresearch',
                'run_id': 'fixture-' + identity, 'metrics': {'dev.score': report['score'] if report else 0.}})
            return 'fixture-' + identity

    return repo, config, FixtureController


def finish(controller, identity):
    deadline = time.monotonic() + 15
    while time.monotonic() < deadline:
        result = controller.status(identity)
        if result['status'] == 'complete':
            # Completion includes public publication, not just report creation.
            while controller.active is not None and time.monotonic() < deadline:
                time.sleep(.01)
            return result
        time.sleep(.01)
    pytest.fail('controller fixture did not complete')


def assert_private_canary_absent(public):
    for path in public.rglob('*'):
        if path.is_file():
            body = path.read_bytes()
            assert PRIVATE_CANARY.encode() not in body
            assert b'0.789123456789' not in body and b'7.123456789' not in body


def test_three_transactions_include_crash_discard_independent_histories_and_dev_only_export(setup):
    repo, config, factory = setup
    controller = factory(config)
    try:
        original = (repo/'strategy/strategy.py').read_text()
        outcomes = []
        for number, content in enumerate([original, original + '\n# FIXTURE_CRASH\n', original + '\n# comment only\n']):
            if number:
                (repo/'strategy/strategy.py').write_text(content)
                commit(repo)
            identity = f'{number + 1:032x}'
            assert controller.submit(identity, f'controller fixture {number}', source_proof(repo))['status'] == 'queued'
            outcomes.append(finish(controller, identity))
        assert [row['decision'] for row in outcomes] == ['initialize', 'discard', 'discard']
        assert [row['grade_status'] for row in outcomes] == ['rejected', 'crash', 'rejected']
        assert controller.journal.read()['count'] == 3
        assert controller.journal.read()['sharpes'] == []
        assert controller.hidden_journal.read()['count'] == 3
        assert controller.hidden_journal.read()['sharpes'] == [7.123456789, 7.123456789]
        assert ('dev', 9) in controller.rescored_counts
        assert (Path(config['public_root'])/'ledger.jsonl').read_bytes() == controller.journal.path.read_bytes()
        assert (Path(config['public_root'])/'trials'/outcomes[0]['trial_id']/'source-proof.json').exists()
        assert_private_canary_absent(Path(config['public_root']))
        prior_count = controller.journal.read()['count']
        assert controller.submit(outcomes[-1]['trial_id'], 'controller fixture 2', source_proof(repo)) == outcomes[-1]
        assert controller.journal.read()['count'] == prior_count
    finally:
        controller.close()


def test_forbidden_proposal_is_recorded_and_never_evaluated(setup, monkeypatch):
    repo, config, factory = setup
    controller = factory(config)
    monkeypatch.setattr(controller, '_grade', lambda *a: pytest.fail('forbidden source must never run'))
    try:
        (repo/'grader/score.py').write_text('score=1\n')
        commit(repo)
        identity = 'a'*32
        controller.submit(identity, 'forbidden fixture', source_proof(repo))
        result = finish(controller, identity)
        assert result['score'] == 0 and result['decision'] == 'discard'
        assert controller.journal.read()['count'] == 1
        assert controller.hidden_journal.read()['count'] == 1
    finally:
        controller.close()


def test_restart_records_interrupted_attempt_without_relaunching_it(setup):
    _, config, factory = setup
    journal = Journal(config['private_root'], config['public_root'])
    identity = 'b'*32
    journal.append('attempt', trial_id=identity, commit='c'*40, description='interrupted fixture',
        hypothesis_sha256=None, candidate_sha256=None)
    controller = factory(config)
    try:
        assert controller.status(identity)['grade_status'] == 'crash'
        assert controller.journal.read()['count'] == 1
        assert controller.hidden_journal.read()['count'] == 1
        assert controller.active is None
    finally:
        controller.close()


@pytest.mark.parametrize('message', [
    {'version': 1, 'operation': 'shadow'},
    {'version': 1, 'operation': 'info', 'tier': 'shadow'},
    {'version': 1, 'operation': 'submit', 'request_id': 'f'*32, 'description': 'bad', 'proof': {}, 'command': 'cat /srv'},
])
def test_wire_has_no_hidden_query_or_path_command_escape(message):
    with pytest.raises(ValueError):
        dispatch(None, message)


def test_real_unix_peer_credentials_reject_wrong_uid_without_dispatch(tmp_path):
    class NeverDispatch:
        closed = False
        config = {'agent_uid': os.getuid() + 1}
    controller = NeverDispatch()
    listener = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    path = tmp_path/'control.sock'
    listener.bind(str(path))
    listener.listen(2)
    thread = threading.Thread(target=serve, args=(listener, controller), daemon=True)
    thread.start()
    try:
        result = request(path, {'version': 1, 'operation': 'info'})
        assert result == {'status': 'error', 'message': 'Peer identity is not authorized'}
    finally:
        controller.closed = True
        thread.join(timeout=2)
        listener.close()
