"""Local Trackio delivery of an explicit dev-only metric whitelist."""
from __future__ import annotations

import json
import math
import os
from pathlib import Path
import re
import sys

PROJECT = 'ethpredict-autoresearch'
FIELDS = {'trial_id', 'commit', 'status', 'score', 'pre_gate_score', 'worker_seconds', 'peak_vram_bytes'}


def log(payload, receipt):
    if not isinstance(payload, dict) or set(payload) != FIELDS:
        raise ValueError('Unknown telemetry fields')
    if not isinstance(payload['trial_id'], str) or not re.fullmatch('[a-f0-9]{32}', payload['trial_id']):
        raise ValueError('Invalid telemetry trial identity')
    if payload['commit'] is not None and not re.fullmatch('[a-f0-9]{40}', payload['commit']):
        raise ValueError('Invalid telemetry commit')
    if payload['status'] not in {'accepted', 'rejected', 'crash', 'timeout', 'nondeterministic'}:
        raise ValueError('Invalid telemetry status')
    metrics = {}
    for name in ['score', 'pre_gate_score', 'worker_seconds', 'peak_vram_bytes']:
        value = payload[name]
        if value is None and name != 'score':
            continue
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
            raise ValueError('Invalid telemetry metric')
        if name in {'score', 'pre_gate_score'} and value > 1:
            raise ValueError('Invalid telemetry score')
        metrics['dev.' + name] = value
    import trackio
    from trackio.sqlite_storage import SQLiteStorage
    run = trackio.init(project=PROJECT, name=payload['trial_id'], resume='allow',
        config={'trial_id': payload['trial_id'], 'commit': payload['commit'],
                'status': payload['status'], 'tier': 'dev'},
        auto_log_gpu=False, auto_log_cpu=False, embed=False)
    try:
        trackio.log(metrics)
    finally:
        trackio.finish()
    records = SQLiteStorage.get_run_records(PROJECT)
    if not any(row['id'] == run.id and row['name'] == payload['trial_id'] for row in records):
        raise RuntimeError('Trackio run identity was not persisted')
    rows = SQLiteStorage.get_logs(PROJECT, run_id=run.id)
    if not any(row.get('dev.score') == payload['score'] for row in rows):
        raise RuntimeError('Trackio score readback failed')
    content = {'project': PROJECT, 'run_name': payload['trial_id'], 'run_id': run.id,
               'delivery': 'local_verified', 'metrics': metrics}
    temporary = Path(str(receipt) + '.tmp')
    with temporary.open('w') as stream:
        json.dump(content, stream, allow_nan=False)
        stream.write('\n')
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, receipt)
    return run.id


def main():
    log(json.loads(Path(sys.argv[1]).read_text()), Path(sys.argv[2]))


if __name__ == '__main__':
    main()
