"""Private ethgrader coordinator. The socket exports dev information only.

The production launcher supplies a root-owned configuration and frozen grader
import path. No request accepts filesystem paths, commands, model IDs, tier
names, configuration overrides, or hidden-report queries.
"""
from __future__ import annotations

from copy import deepcopy
import fcntl
import hashlib
import json
import argparse
import os
from pathlib import Path
import pwd
import queue
import re
import shutil
import signal
import socket
import struct
import subprocess
import sys
import threading
import traceback

from grader.artifacts import verify_artifacts
from grader.baseline_bank import read_bank
from grader.checks import verify_frozen_grader
from grader.models import sha256_file
from grader.report import read_report

from .decisions import decision, monitor, rescore_saved
from .journal import Journal, atomic, write_json
from .protocol import encode, receive, send, unique, verify_source


IDENTITY = re.compile('[a-f0-9]{32}')
CONFIG_FIELDS = {'version', 'grader_root', 'private_root', 'public_root', 'dev_lake',
                 'private_lake', 'model_paths', 'protected_entries', 'agent_uid', 'gpu_lock'}
ALERT = b'Autoresearch paused. Human review of the private service is required.\n'


def validate_config(config, *, production=True):
    if set(config) != CONFIG_FIELDS or type(config['version']) is not int or config['version'] != 1:
        raise ValueError('Unknown service configuration')
    for name in ['grader_root', 'private_root', 'public_root', 'dev_lake', 'private_lake', 'gpu_lock']:
        if not isinstance(config[name], str) or not Path(config[name]).is_absolute():
            raise ValueError('Service paths must be explicit absolute paths')
    if type(config['agent_uid']) is not int or config['agent_uid'] <= 0:
        raise ValueError('Provision one unprivileged loop identity')
    if production:
        identity = pwd.getpwnam('ethgrader')
        if os.geteuid() != identity.pw_uid:
            raise PermissionError('The production service must run as ethgrader')
        private = Path('/srv/ethpredict-grader')
        for name in ['grader_root', 'private_root', 'dev_lake', 'private_lake']:
            if not Path(config[name]).resolve().is_relative_to(private):
                raise ValueError('All grader runtime/data paths must remain private')
        if Path(config['public_root']).resolve().is_relative_to(private):
            raise ValueError('The dev export must be outside the sealed root')


def grade_status(report):
    return 'crash' if any(row['check'] == 'crash' for row in report['failures']) else report['status']


class Controller:
    def __init__(self, config, *, production=True):
        validate_config(config, production=production)
        self.config = deepcopy(config)
        self.root, self.public = Path(config['private_root']), Path(config['public_root'])
        self.root.mkdir(parents=True, exist_ok=True, mode=0o700)
        self.guard = os.open(self.root/'controller.lock', os.O_CREAT | os.O_RDWR | os.O_NOFOLLOW, 0o600)
        try:
            fcntl.flock(self.guard, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BaseException:
            os.close(self.guard)
            raise
        self.journal = Journal(self.root, self.public)
        self.hidden_journal = Journal(self.root/'shadow-history')
        self.frozen = verify_frozen_grader(Path(config['grader_root']))
        if not self.frozen.passed:
            os.close(self.guard)
            raise ValueError('Private grader bytes differ from grader-v1')
        self.tag = subprocess.check_output(['git', '-C', config['grader_root'],
            'rev-parse', 'grader-v1^{commit}'], text=True).strip()
        head = subprocess.check_output(['git', '-C', config['grader_root'], 'rev-parse', 'HEAD'], text=True).strip()
        if head != self.tag:
            os.close(self.guard)
            raise ValueError('Private grader checkout is not pinned to the tag')
        self.lock = threading.RLock()
        self.jobs = queue.Queue(maxsize=1)
        self.active = None
        self.process = None
        self.closed = False
        self.incumbent = None
        self._recover()
        self.thread = threading.Thread(target=self._work, name='private-grader', daemon=True)
        self.thread.start()

    def _path(self, identity):
        if not isinstance(identity, str) or not IDENTITY.fullmatch(identity):
            raise ValueError('Invalid request identity')
        return self.root/'trials'/identity

    def alert(self):
        atomic(self.root/'HUMAN_ALERT', ALERT)
        atomic(self.public/'HUMAN_ALERT', ALERT, mode=0o444)

    def _recover(self):
        # An interrupted evaluation is counted and terminated, never silently
        # rerun. systemd KillMode=control-group kills its old child processes.
        for path in sorted((self.root/'trials').glob('*/request.json')):
            identity = path.parent.name
            self._path(identity)
            if identity not in self.journal.read()['trials']:
                stored = json.loads(path.read_text())
                proof = stored['proof']
                try:
                    files = verify_source(proof, self.config['protected_entries'])
                    hypothesis = hashlib.sha256(files['strategy/README.md'][1]).hexdigest()
                except Exception:
                    hypothesis = None
                commit = proof.get('commit') if isinstance(proof, dict) else None
                if not isinstance(commit, str) or not re.fullmatch('[a-f0-9]{40}', commit):
                    commit = None
                self.journal.append('attempt', trial_id=identity, commit=commit,
                    description=stored['description'], hypothesis_sha256=hypothesis,
                    candidate_sha256=hashlib.sha256(encode(proof)).hexdigest())
        for identity, trial in self.journal.read()['trials'].items():
            if identity not in self.hidden_journal.read()['trials']:
                fields = {key: value for key, value in trial['attempt'].items()
                          if key not in {'version', 'event', 'timestamp_utc'}}
                self.hidden_journal.append('attempt', **fields)
            if 'result' not in self.hidden_journal.read()['trials'][identity]:
                self._record_result(self.hidden_journal, identity, None, 'shadow', 'crash', None)
            if trial.get('decision', {}).get('decision') in {'initialize', 'keep'}:
                self.incumbent = {'trial_id': identity, 'commit': trial['attempt']['commit']}
            if 'result' not in trial:
                self.journal.append('result', trial_id=identity, status='crash', score=0.,
                    pre_gate_score=None, sharpe_per_period=None, report_sha256=None,
                    report_path=None, trackio_run_id=None)
            if 'decision' not in trial:
                self.journal.append('decision', trial_id=identity, decision='discard',
                    incumbent_commit=self.incumbent['commit'] if self.incumbent else None,
                    incumbent_rescored_score=None, candidate_score=trial.get('result', {}).get('score', 0.),
                    delta=None, qualitative=None)
        self.journal.publish()
        for identity in self.journal.read()['trials']:
            directory = self._path(identity)
            if not (directory/'response.json').exists():
                write_json(directory/'response.json', self._response(identity))
                self._export(identity)
        if (self.root/'HUMAN_ALERT').exists():
            atomic(self.public/'HUMAN_ALERT', ALERT, mode=0o444)

    def info(self):
        with self.lock:
            return {'status': 'ready', 'grader_tag': 'grader-v1', 'grader_commit': self.tag,
                'grader_hashes': self.frozen.details['sha256'],
                'incumbent_commit': self.incumbent['commit'] if self.incumbent else None,
                'incumbent_trial': self.incumbent['trial_id'] if self.incumbent else None,
                'trial_count': self.journal.read()['count'],
                'human_alert': (self.root/'HUMAN_ALERT').exists(), 'busy': self.active is not None}

    def submit(self, identity, description, proof):
        directory = self._path(identity)
        if (not isinstance(description, str) or not 0 < len(description) <= 512 or
                any(ord(character) < 32 or ord(character) == 127 for character in description)):
            raise ValueError('Use a bounded single-line description')
        payload = {'description': description, 'proof': proof}
        digest = hashlib.sha256(encode(payload)).hexdigest()
        with self.lock:
            stored = directory/'request.json'
            if stored.exists():
                prior = json.loads(stored.read_text())
                if prior['sha256'] != digest:
                    raise ValueError('Request identity was reused for different source')
                return self.status(identity)
            if (self.root/'HUMAN_ALERT').exists():
                raise RuntimeError('HUMAN_ALERT: paused for human review')
            if self.active is not None:
                raise RuntimeError('An experiment is active; observe its existing request')
            valid, hypothesis = True, None
            try:
                files = verify_source(proof, self.config['protected_entries'])
                hypothesis = hashlib.sha256(files['strategy/README.md'][1]).hexdigest()
            except Exception:
                valid = False
            commit = proof.get('commit') if isinstance(proof, dict) else None
            if not isinstance(commit, str) or not re.fullmatch('[a-f0-9]{40}', commit):
                commit = None
            directory.mkdir(parents=True, mode=0o700)
            write_json(stored, {**payload, 'sha256': digest, 'valid_source': valid})
            self.journal.append('attempt', trial_id=identity, commit=commit, description=description,
                hypothesis_sha256=hypothesis, candidate_sha256=hashlib.sha256(encode(proof)).hexdigest())
            self.hidden_journal.append('attempt', trial_id=identity, commit=commit, description=description,
                hypothesis_sha256=hypothesis, candidate_sha256=hashlib.sha256(encode(proof)).hexdigest())
            self.journal.publish()
            self.active = identity
            self.jobs.put_nowait(identity)
            return {'status': 'queued', 'request_id': identity}

    def status(self, identity):
        directory = self._path(identity)
        with self.lock:
            if (directory/'response.json').exists() and self.active != identity:
                return json.loads((directory/'response.json').read_text())
            if identity not in self.journal.read()['trials']:
                return {'status': 'unknown', 'request_id': identity}
            return {'status': 'running', 'request_id': identity}

    def _grade(self, identity, tier, candidate):
        if self.closed:
            raise RuntimeError('Private service is stopping')
        directory = self._path(identity)
        output = directory/tier
        command = [sys.executable, '-I', '-c',
            f"import sys;sys.path.insert(0,{self.config['grader_root']!r});from grader.cli import main;raise SystemExit(main())",
            'grade', '--strategy', str(candidate), '--tier', tier, '--lake', self.config['dev_lake'],
            '--private-lake', self.config['private_lake'], '--output', str(output),
            '--baseline-cache', str(self.root/'baselines'/tier), '--ledger',
            str(self.journal.path if tier == 'dev' else self.hidden_journal.path),
            '--trial-id', identity]
        for model, path in self.config['model_paths'].items():
            command += ['--model-path', f'{model}={path}']
        environment = {'PATH': '/usr/bin:/bin', 'HOME': str(self.root), 'PYTHONNOUSERSITE': '1',
            'PYTHONDONTWRITEBYTECODE': '1', 'OMP_NUM_THREADS': '1', 'OPENBLAS_NUM_THREADS': '1',
            'MKL_NUM_THREADS': '1', 'HF_HUB_OFFLINE': '1', 'HF_DATASETS_OFFLINE': '1'}
        environment.update(GIT_CONFIG_NOSYSTEM='1',
            GIT_CONFIG_GLOBAL=str(Path(self.config['grader_root']).parent/'gitconfig'))
        with (directory/f'{tier}.log').open('wb') as log:
            self.process = subprocess.Popen(command, cwd=self.config['grader_root'], env=environment,
                stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
            try:
                self.process.wait(timeout=6*3600)
                if self.process.returncode:
                    raise RuntimeError('Private grading command failed')
            finally:
                if self.process.poll() is None:
                    os.killpg(self.process.pid, signal.SIGKILL)
                    self.process.wait()
                self.process = None
        report = read_report(output/'grade_report.json')
        if report['mode'] != 'production' or report['tier'] != tier or report['trial_id'] != identity:
            raise ValueError('Grade report does not match the request')
        if report['artifacts']['manifest'] is not None:
            verify_artifacts(output, report['artifacts']['manifest'])
        return report

    def _bank(self, identity, tier):
        return read_bank(self._path(identity)/tier/'baseline_bank.json')

    def _telemetry(self, identity, report, status):
        # Execute a separate clean process after hidden work has stopped. It
        # receives only a bounded whitelist of dev values, no service state.
        directory = self._path(identity)
        payload = {'trial_id': identity, 'commit': self.journal.read()['trials'][identity]['attempt']['commit'],
            'status': status, 'score': report['score'] if report else 0.,
            'pre_gate_score': report['pre_gate_score'] if report else None,
            'worker_seconds': report.get('resources', {}).get('wall_seconds') if report else None,
            'peak_vram_bytes': report.get('resources', {}).get('peak_vram_bytes') if report else None}
        write_json(directory/'telemetry-input.json', payload)
        environment = {'PATH': '/usr/bin:/bin', 'HOME': str(self.root),
            'TRACKIO_DIR': str(self.public/'trackio'), 'HF_HUB_OFFLINE': '1',
            'PYTHONNOUSERSITE': '1', 'PYTHONDONTWRITEBYTECODE': '1'}
        module_root = str(Path(__file__).resolve().parents[1])
        command = [sys.executable, '-I', '-c',
            f"import sys;sys.path.insert(0,{module_root!r});from autoresearch.telemetry import main;main()",
            str(directory/'telemetry-input.json'), str(directory/'trackio.json')]
        with (directory/'trackio.log').open('wb') as log:
            subprocess.run(command, env=environment, cwd=directory, stdout=log,
                stderr=subprocess.STDOUT, timeout=90, check=True)
        return json.loads((directory/'trackio.json').read_text())['run_id']

    def _run(self, identity):
        descriptor = os.open(self.config['gpu_lock'], os.O_RDWR | os.O_NOFOLLOW)
        try:
            fcntl.flock(descriptor, fcntl.LOCK_EX)
            self._perform(identity)
        finally:
            os.close(descriptor)

    def _perform(self, identity):
        directory = self._path(identity)
        saved = json.loads((directory/'request.json').read_text())
        dev, status, run_id = None, 'rejected', None
        if saved['valid_source']:
            candidate = directory/'candidate'
            for name, (mode, body) in verify_source(saved['proof'], self.config['protected_entries']).items():
                target = candidate/name
                target.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
                target.write_bytes(body)
                target.chmod(mode & 0o777)
            atomic(directory/'source-proof.json', encode(saved['proof']))
            # _run holds the external OS lock through both grades, all banks,
            # telemetry and the statistical comparison.
            dev = self._grade(identity, 'dev', candidate)
            hidden = self._grade(identity, 'shadow', candidate)
            self._record_result(self.hidden_journal, identity, hidden, 'shadow', grade_status(hidden), None)
            status = grade_status(dev)
        else:
            self._record_result(self.hidden_journal, identity, None, 'shadow', 'rejected', None)
        run_id = self._telemetry(identity, dev, status)
        with self.lock:
            self._finish_result(identity, dev, status, run_id)

    def _record_result(self, journal, identity, report, tier, status, run_id):
        directory = self._path(identity)
        available = [row['daily_sharpe_per_period']['value'] for row in report['metrics']
                     if row['daily_sharpe_per_period']['available']] if report else []
        journal.append('result', trial_id=identity, status=status,
            score=report['score'] if report else 0., pre_gate_score=report['pre_gate_score'] if report else None,
            sharpe_per_period=sum(available)/len(available) if available else None,
            report_sha256=sha256_file(directory/tier/'grade_report.json') if report else None,
            report_path=f'trials/{identity}/{tier}/grade_report.json' if report else None, trackio_run_id=run_id)

    def _rescore(self, identity, tier):
        path = self._path(identity)/tier/'grade_report.json'
        report = read_report(path)
        if len(report['metrics']) != 2 or any(len(row['blocks']) != 5 for row in report['metrics']):
            return report
        history = self.journal.read() if tier == 'dev' else self.hidden_journal.read()
        return rescore_saved(path, history, self._bank(identity, tier))

    def _finish_result(self, identity, report, status, run_id):
        directory = self._path(identity)
        self._record_result(self.journal, identity, report, 'dev', status, run_id)
        previous = None
        if self.incumbent:
            previous = self._rescore(self.incumbent['trial_id'], 'dev')
        choice = decision(report, previous) if report else {'decision': 'discard',
            'incumbent_rescored_score': previous['score'] if previous else None, 'candidate_score': 0., 'delta': None}
        if choice['decision'] in {'initialize', 'keep'}:
            self.incumbent = {'trial_id': identity, 'commit': self.journal.read()['trials'][identity]['attempt']['commit']}
        self.journal.append('decision', trial_id=identity, **choice,
            incumbent_commit=self.incumbent['commit'] if self.incumbent else None, qualitative=None)
        if self.incumbent:
            count = self.journal.read()['count']
            monitor_path = self.root/'monitor.json'
            if not monitor_path.exists() or count % 10 == 0:
                visible = self._rescore(self.incumbent['trial_id'], 'dev')
                hidden = self._rescore(self.incumbent['trial_id'], 'shadow')
                state = monitor(json.loads(monitor_path.read_text()) if monitor_path.exists() else None,
                    trial_count=count, dev_score=visible['score'], hidden_score=hidden['score'])
                write_json(monitor_path, state)
                if state['alert']:
                    self.alert()
        write_json(directory/'response.json', self._response(identity))
        self._export(identity)
        self.journal.publish()

    def _response(self, identity):
        trial = self.journal.read()['trials'][identity]
        return {'status': 'complete', 'trial_id': identity, 'score': trial['result']['score'],
            'grade_status': trial['result']['status'], 'decision': trial['decision']['decision'],
            'incumbent_commit': self.incumbent['commit'] if self.incumbent else None,
            'incumbent_trial': self.incumbent['trial_id'] if self.incumbent else None,
            'report': trial['result']['report_path'], 'trackio_run_id': trial['result']['trackio_run_id'],
            'human_alert': (self.root/'HUMAN_ALERT').exists()}

    def _export(self, identity):
        source = self._path(identity)
        target = self.public/'trials'/identity
        if target.exists():
            return
        temporary = self.public/'trials'/('.' + identity)
        if temporary.exists():
            if temporary.is_symlink() or temporary.stat().st_uid != os.geteuid():
                raise ValueError('Untrusted unfinished export directory')
            shutil.rmtree(temporary)
        temporary.mkdir(parents=True, mode=0o700)
        # There is intentionally no recursive copy of the trial directory.
        # Only the independently verified DEV manifest may select artifacts.
        report_path = source/'dev/grade_report.json'
        trial = self.journal.read()['trials'][identity]
        if trial['result']['report_path'] is not None:
            report = read_report(report_path)
            if sha256_file(report_path) != trial['result']['report_sha256']:
                raise ValueError('Dev export report hash changed')
            files = []
            if report['artifacts']['manifest'] is not None:
                manifest = verify_artifacts(report_path.parent, report['artifacts']['manifest'])
                files = [row['path'] for row in manifest['files']] + ['artifact_manifest.json']
            for name in [*files, 'grade_report.json']:
                destination = temporary/'dev'/name
                destination.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
                shutil.copyfile(report_path.parent/name, destination)
        for name in ['source-proof.json', 'trackio.json']:
            if (source/name).exists():
                shutil.copyfile(source/name, temporary/name)
        atomic(temporary/'run.log', f"score: {trial['result']['score']:.12f}\nstatus: {trial['result']['status']}\n".encode())
        for path in temporary.rglob('*'):
            path.chmod(0o755 if path.is_dir() else 0o444)
        temporary.chmod(0o755)
        os.replace(temporary, target)

    def _work(self):
        while True:
            identity = self.jobs.get()
            if identity is None:
                return
            try:
                self._run(identity)
            except BaseException:
                directory = self._path(identity)
                atomic(directory/'controller-error.log', traceback.format_exc().encode())
                with self.lock:
                    trial = self.journal.read()['trials'][identity]
                    if 'result' not in self.hidden_journal.read()['trials'][identity]:
                        self._record_result(self.hidden_journal, identity, None, 'shadow', 'crash', None)
                    if 'result' not in trial:
                        self.journal.append('result', trial_id=identity, status='crash', score=0.,
                            pre_gate_score=None, sharpe_per_period=None, report_sha256=None,
                            report_path=None, trackio_run_id=None)
                    if 'decision' not in trial:
                        self.journal.append('decision', trial_id=identity, decision='discard',
                            incumbent_commit=self.incumbent['commit'] if self.incumbent else None,
                            incumbent_rescored_score=None, candidate_score=trial.get('result', {}).get('score', 0.),
                            delta=None, qualitative=None)
                    self.alert()
                    write_json(directory/'response.json', self._response(identity))
                    self._export(identity)
                    self.journal.publish()
            finally:
                with self.lock:
                    self.active = None
                self.jobs.task_done()

    def close(self):
        self.closed = True
        process = self.process
        if process is not None and process.poll() is None:
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
        self.jobs.put(None)
        self.thread.join(timeout=30)
        if self.thread.is_alive():
            raise RuntimeError('Private worker has not terminated')
        os.close(self.guard)


def dispatch(controller, message):
    if not isinstance(message, dict) or type(message.get('version')) is not int or message['version'] != 1:
        raise ValueError('Unknown protocol version')
    fields = {'info': {'version', 'operation'}, 'status': {'version', 'operation', 'request_id'},
        'submit': {'version', 'operation', 'request_id', 'description', 'proof'}}
    operation = message.get('operation')
    if operation not in fields or set(message) != fields[operation]:
        raise ValueError('Unknown request fields or operation')
    if operation == 'info':
        return controller.info()
    if operation == 'status':
        return controller.status(message['request_id'])
    return controller.submit(message['request_id'], message['description'], message['proof'])


def serve(listener, controller):
    listener.settimeout(1)
    while not controller.closed:
        try:
            stream, _ = listener.accept()
        except socket.timeout:
            continue
        with stream:
            stream.settimeout(30)
            _, uid, _ = struct.unpack('3i', stream.getsockopt(socket.SOL_SOCKET, socket.SO_PEERCRED, 12))
            if uid != controller.config['agent_uid']:
                send(stream, {'status': 'error', 'message': 'Peer identity is not authorized'})
                continue
            try:
                send(stream, dispatch(controller, receive(stream)))
            except Exception:
                # Never serialize an exception from a private operation.
                send(stream, {'status': 'error', 'message': 'Request rejected; inspect the current request or contact the human operator'})


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True, type=Path)
    args = parser.parse_args(argv)
    descriptor = os.open(args.config, os.O_RDONLY | os.O_NOFOLLOW)
    with os.fdopen(descriptor) as source:
        metadata = os.fstat(source.fileno())
        if metadata.st_uid != 0 or metadata.st_mode & 0o022 or metadata.st_nlink != 1:
            raise PermissionError('Use the root-owned reviewed service configuration')
        config = json.load(source, object_pairs_hook=unique)
    if os.environ.get('LISTEN_PID') != str(os.getpid()) or os.environ.get('LISTEN_FDS') != '1':
        raise RuntimeError('Start the service through its reviewed systemd socket unit')
    controller = Controller(config)
    listener = socket.socket(fileno=3)
    if listener.family != socket.AF_UNIX or listener.getsockname() != '/run/ethpredict-grader/control.sock':
        controller.close()
        raise ValueError('Unexpected inherited service socket')
    def stopping(signum, frame):
        controller.closed = True
    signal.signal(signal.SIGTERM, stopping)
    signal.signal(signal.SIGINT, stopping)
    try:
        serve(listener, controller)
    finally:
        controller.close()
        listener.close()
