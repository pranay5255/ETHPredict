"""Confined loop client: committed source -> private broker -> keep/reset.

Run this through scripts.agent_sandbox. The client cannot access a GPU, hidden
data, private reports or the authoritative journal. It receives dev results.
"""
from __future__ import annotations

import argparse
import base64
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import time
import uuid

if __package__ in {None, ''}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from autoresearch.protocol import object_id, request, source_proof


def confined():
    status = dict(line.split(':', 1) for line in Path('/proc/self/status').read_text().splitlines() if ':' in line)
    if (Path.cwd() != Path('/workspace') or os.geteuid() == 0 or
            not Path('/.dockerenv').exists() or int(status['CapEff'].strip(), 16) != 0 or
            status['NoNewPrivs'].strip() != '1' or status['Seccomp'].strip() != '2'):
        raise RuntimeError('Run this client through scripts.agent_sandbox; its Git operations belong inside the isolated worker')


def git(*arguments, body=None):
    return subprocess.check_output(['git', '-c', 'core.fsmonitor=false', '-c', 'core.hooksPath=/dev/null',
        '-c', 'commit.gpgsign=false', *arguments], input=body, timeout=60).decode().strip()


def save_state(path, value):
    temporary = Path(str(path) + '.tmp')
    with temporary.open('w') as stream:
        json.dump(value, stream)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def restore(commit, trial, public):
    if not re.fullmatch('[a-f0-9]{40}', commit) or not re.fullmatch('[a-f0-9]{32}', trial):
        raise ValueError('Invalid incumbent identity')
    # A fresh code-only workspace may not yet have the incumbent's objects.
    # Write only raw objects from the broker's read-only dev bundle. This is
    # still client-side Git inside the sandbox, never host Git.
    proof = json.loads((public/'trials'/trial/'source-proof.json').read_text())
    if proof['commit'] != commit:
        raise ValueError('Incumbent bundle does not match its reported commit')
    for collection, kind in [('blobs', 'blob'), ('trees', 'tree'), ('commits', 'commit')]:
        for oid, encoded in proof[collection].items():
            body = base64.b64decode(encoded, validate=True)
            if object_id(kind, body) != oid or git('hash-object', '-w', '-t', kind, '--stdin', body=body) != oid:
                raise ValueError('Incumbent Git object changed')
    git('reset', '--hard', commit)


def verify_info(info):
    fields = {'status', 'grader_tag', 'grader_commit', 'grader_hashes', 'incumbent_commit',
              'incumbent_trial', 'trial_count', 'human_alert', 'busy'}
    if set(info) != fields or info['status'] != 'ready' or info['grader_tag'] != 'grader-v1':
        raise ValueError('Private broker is not the frozen grader-v1 service')
    for name, digest in info['grader_hashes'].items():
        path = Path(name)
        if (not name.startswith('grader/') or '..' in path.parts or path.is_absolute() or
                path.is_symlink() or hashlib.sha256(path.read_bytes()).hexdigest() != digest):
            raise ValueError('Local protected grader differs from the private frozen checkout')
    if info['human_alert']:
        raise RuntimeError('HUMAN_ALERT: paused for human review')


class Client:
    def __init__(self, path, public, *, poll_seconds=2):
        self.path, self.public, self.poll_seconds = path, Path(public), poll_seconds

    def call(self, operation, **fields):
        result = request(self.path, {'version': 1, 'operation': operation, **fields})
        if result.get('status') == 'error':
            raise RuntimeError(result['message'])
        return result

    def info(self):
        info = self.call('info')
        verify_info(info)
        return info

    def run(self, description, *, identity=None):
        self.info()
        # Persist the request before submission. If observation is interrupted,
        # resume this same request; never restart an experiment merely because
        # a connection or observation timed out.
        pending = Path('.git/ethpredict-pending.json')
        if pending.exists():
            saved = json.loads(pending.read_text())
            if identity is not None and saved['request_id'] != identity:
                raise RuntimeError('Observe the preceding pending request before starting another')
            identity = saved['request_id']
        else:
            identity = identity or uuid.uuid4().hex
            saved = {'request_id': identity, 'description': description, 'proof': source_proof(Path.cwd())}
            save_state(pending, saved)
        response = self.call('submit', **saved)
        return self.observe(identity, response=response)

    def observe(self, identity, *, response=None):
        response = response or self.call('status', request_id=identity)
        if response['status'] not in {'queued', 'running', 'complete'}:
            raise RuntimeError('Broker does not have this bounded trial')
        while response['status'] != 'complete':
            time.sleep(self.poll_seconds)
            response = self.call('status', request_id=identity)
        fields = {'status', 'trial_id', 'score', 'grade_status', 'decision', 'incumbent_commit',
                  'incumbent_trial', 'report', 'trackio_run_id', 'human_alert'}
        if set(response) != fields:
            raise ValueError('Unknown broker result fields')
        print(f"score: {response['score']:.12f}", flush=True)
        print(f"status: {response['grade_status']}", flush=True)
        print(f"decision: {response['decision']}", flush=True)
        print(f"trial: {response['trial_id']}", flush=True)
        if response['decision'] == 'discard' and response['incumbent_commit'] is not None:
            restore(response['incumbent_commit'], response['incumbent_trial'], self.public)
        pending = Path('.git/ethpredict-pending.json')
        if pending.exists():
            if json.loads(pending.read_text())['request_id'] != identity:
                raise RuntimeError('A different observation is still pending')
            pending.unlink()
        if response['human_alert']:
            raise RuntimeError('HUMAN_ALERT: paused for human review')
        return response


CONTROL = '''"""Untuned flat control for the bounded infrastructure dry run."""
import numpy as np
from grader.baselines.common import Baseline

class Strategy(Baseline):
    name = 'autoresearch-dry-run-flat'
    def fit(self, train, ctx):
        pass
    def predict(self, history, decision_times):
        return self.positions(history, decision_times, lambda data: np.zeros(len(data)))
'''


def dry_run(client):
    state_path = Path('.git/ethpredict-dry-run.json')
    if state_path.exists():
        state = json.loads(state_path.read_text())
    else:
        if Path('.git/ethpredict-pending.json').exists():
            raise RuntimeError('Finish observing the pending experiment before the dry run')
        state = {'before_count': client.info()['trial_count'],
                 'requests': [uuid.uuid4().hex for _ in range(3)], 'outcomes': {}}
        save_state(state_path, state)
    for number, identity in enumerate(state['requests']):
        if identity in state['outcomes']:
            continue
        current = client.call('status', request_id=identity)
        if current['status'] != 'unknown':
            state['outcomes'][identity] = client.observe(identity, response=current)
            save_state(state_path, state)
            continue
        code = CONTROL
        if number == 1:
            code = code.replace('        pass', "        raise RuntimeError('deliberate infrastructure dry-run crash')")
        elif number == 2:
            code += '\n# No behavior change: this proposal must be discarded.\n'
        Path('strategy/strategy.py').write_text(code)
        Path('strategy/README.md').write_text(
            '# Bounded infrastructure hypothesis\n\n'
            'Flat is the control, the second trial deliberately crashes, and the third '
            'changes a comment only. No score improvement is expected.\n'
            f'\nDry-run step: {number + 1}/3.\n')
        git('add', '--', 'src', 'strategy')
        git('commit', '--allow-empty', '-m', f'Infrastructure dry-run step {number + 1}')
        state['outcomes'][identity] = client.run(f'Infrastructure dry run {number + 1}/3', identity=identity)
        save_state(state_path, state)
    outcomes = [state['outcomes'][identity] for identity in state['requests']]
    after = client.info()['trial_count']
    if after != state['before_count'] + 3 or outcomes[1]['grade_status'] != 'crash' or outcomes[2]['decision'] != 'discard':
        raise RuntimeError('Three-attempt infrastructure acceptance check failed')
    print(f'dry_run_trials: 3; authoritative_total: {after}', flush=True)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--socket', default='/broker/control.sock')
    parser.add_argument('--results', type=Path, default=Path('/results'))
    parser.add_argument('--description', default='Committed strategy experiment')
    parser.add_argument('--dry-run', action='store_true')
    parser.add_argument('--setup', metavar='RUN_TAG')
    args = parser.parse_args(argv)
    confined()
    client = Client(args.socket, args.results)
    info = client.info()
    if args.setup:
        if not re.fullmatch('[A-Za-z0-9][A-Za-z0-9_.-]{0,63}', args.setup):
            raise ValueError('Use a simple run tag')
        git('checkout', '-b', 'autoresearch/' + args.setup)
        print(f"grader: {info['grader_tag']} {info['grader_commit']}")
        print(f"prior_trials: {info['trial_count']}")
        return
    if args.dry_run:
        dry_run(client)
    else:
        client.run(args.description)


if __name__ == '__main__':
    main()
