import json

import pytest

from autoresearch.run_experiment import Client, dry_run
from test_autoresearch_protocol import repository


def completed(identity, *, status='rejected', choice='discard'):
    return {'status': 'complete', 'trial_id': identity, 'score': 0., 'grade_status': status,
        'decision': choice, 'incumbent_commit': None, 'incumbent_trial': None,
        'report': None, 'trackio_run_id': 'fixture', 'human_alert': False}


def test_pending_request_reuses_saved_source_after_disconnect(repository, monkeypatch):
    repo, _ = repository
    monkeypatch.chdir(repo)
    identity = 'd'*32
    saved = {'request_id': identity, 'description': 'original', 'proof': {'original': True}}
    pending = repo/'.git/ethpredict-pending.json'
    pending.write_text(json.dumps(saved))
    import autoresearch.run_experiment as driver
    monkeypatch.setattr(driver, 'source_proof', lambda _: pytest.fail('must not rebuild a pending experiment'))
    client = Client('/unused', repo/'public', poll_seconds=0)
    monkeypatch.setattr(client, 'info', lambda: {})
    calls = []
    def call(operation, **fields):
        calls.append((operation, fields))
        assert operation == 'submit' and fields == saved
        return completed(identity)
    monkeypatch.setattr(client, 'call', call)
    assert client.run('new description ignored during resume')['trial_id'] == identity
    assert len(calls) == 1 and not pending.exists()


def test_dry_run_resumes_completed_request_without_creating_a_fourth_trial(repository, monkeypatch):
    repo, _ = repository
    monkeypatch.chdir(repo)
    class InterruptedClient:
        def __init__(self):
            self.results = {}
            self.count = 0
            self.interrupt = True

        def info(self):
            return {'trial_count': self.count}

        def call(self, operation, *, request_id):
            assert operation == 'status'
            return self.results.get(request_id, {'status': 'unknown', 'request_id': request_id})

        def observe(self, identity, *, response):
            return response

        def run(self, description, *, identity):
            self.count += 1
            self.results[identity] = completed(identity,
                status='crash' if self.count == 2 else 'rejected',
                choice='initialize' if self.count == 1 else 'discard')
            if self.interrupt:
                self.interrupt = False
                raise TimeoutError('Observation ended after the broker completed this identity')
            return self.results[identity]

    client = InterruptedClient()
    with pytest.raises(TimeoutError):
        dry_run(client)
    assert client.count == 1
    dry_run(client)
    assert client.count == 3
    state = json.loads((repo/'.git/ethpredict-dry-run.json').read_text())
    assert set(state['outcomes']) == set(state['requests']) and len(state['requests']) == 3
    dry_run(client)
    assert client.count == 3
