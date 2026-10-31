"""Service-owned append-only event log; public files are read-only dev views."""
from __future__ import annotations

from datetime import datetime, timezone
import io
import csv
import json
import os
from pathlib import Path
import tempfile

from grader.history import read_trials


def timestamp():
    return datetime.now(timezone.utc).isoformat().replace('+00:00', 'Z')


def atomic(path, body, *, mode=0o600):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix='.' + path.name, dir=path.parent)
    try:
        with os.fdopen(descriptor, 'wb') as stream:
            stream.write(body)
            stream.flush()
            os.fsync(stream.fileno())
            os.fchmod(stream.fileno(), mode)
        os.replace(temporary, path)
        directory = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def write_json(path, value, *, mode=0o600):
    atomic(path, (json.dumps(value, indent=2, allow_nan=False) + '\n').encode(), mode=mode)


class Journal:
    def __init__(self, private_root, public_root=None):
        self.root = Path(private_root)
        self.public = Path(public_root) if public_root is not None else None
        self.path = self.root/'ledger.jsonl'
        self.root.mkdir(parents=True, exist_ok=True, mode=0o700)
        if self.public is not None:
            self.public.mkdir(parents=True, exist_ok=True, mode=0o755)

    def read(self):
        return read_trials(self.path)

    def append(self, event, **fields):
        value = {'version': 1, 'event': event, 'timestamp_utc': timestamp(), **fields}
        line = (json.dumps(value, separators=(',', ':'), allow_nan=False) + '\n').encode()
        # Validate the complete transition before appending irreversibly. This
        # temporary file is private and never contains a shadow value either.
        prior = self.path.read_bytes() if self.path.exists() else b''
        descriptor, temporary = tempfile.mkstemp(dir=self.root, prefix='.validate-ledger-')
        try:
            with os.fdopen(descriptor, 'wb') as stream:
                stream.write(prior + line)
            read_trials(temporary)
        finally:
            os.unlink(temporary)
        descriptor = os.open(self.path, os.O_WRONLY | os.O_APPEND | os.O_CREAT | os.O_NOFOLLOW, 0o600)
        with os.fdopen(descriptor, 'ab') as stream:
            stream.write(line)
            stream.flush()
            os.fsync(stream.fileno())
        return value

    def publish(self):
        if self.public is None:
            raise ValueError('A private-only journal cannot be published')
        history = self.read()
        output = io.StringIO(newline='')
        writer = csv.writer(output, delimiter='\t', lineterminator='\n')
        writer.writerow(['trial_id', 'commit', 'score', 'gates', 'status', 'decision', 'description', 'trackio_run_id'])
        for identity, trial in history['trials'].items():
            result, choice = trial.get('result', {}), trial.get('decision', {})
            writer.writerow([identity, trial['attempt']['commit'], result.get('score', ''),
                'pass' if result.get('status') == 'accepted' else 'fail',
                result.get('status', 'interrupted'), choice.get('decision', ''),
                trial['attempt']['description'], result.get('trackio_run_id', '')])
        # The authoritative journal stays private. The public mirror contains
        # exactly its dev-only schema; no monitor/hidden file is ever copied.
        atomic(self.public/'ledger.jsonl', self.path.read_bytes() if self.path.exists() else b'', mode=0o444)
        atomic(self.public/'results.tsv', output.getvalue().encode(), mode=0o444)
