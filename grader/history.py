"""Strict append-only trial-event accounting, including interrupted attempts."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from .data import DataContractError, _json_unique


EVENT_FIELDS = {
    'attempt': {'version','event','trial_id','timestamp_utc','commit','description','hypothesis_sha256','candidate_sha256'},
    'result': {'version','event','trial_id','timestamp_utc','status','score','pre_gate_score','sharpe_per_period','report_sha256','report_path','trackio_run_id'},
    'decision': {'version','event','trial_id','timestamp_utc','decision','incumbent_commit','incumbent_rescored_score','candidate_score','delta','qualitative'},
}


def read_trials(path):
    path = Path(path)
    if not path.exists():
        return {'count': 0, 'sharpes': [], 'trials': {}}
    trials = {}
    for line in path.read_text().splitlines():
        try:
            event = json.loads(line, object_pairs_hook=_json_unique,
                parse_constant=lambda value: (_ for _ in ()).throw(ValueError('Nonfinite JSON')))
            kind = event['event']
            if kind not in EVENT_FIELDS or set(event) != EVENT_FIELDS[kind] or event['version'] != 1:
                raise ValueError('Unknown event fields/version')
            identity = event['trial_id']
            if not isinstance(identity,str) or not identity:
                raise ValueError('Invalid trial identity')
            if kind == 'attempt':
                if identity in trials:
                    raise ValueError('Repeated trial attempt')
                trials[identity] = {'attempt': event}
            elif identity not in trials or kind in trials[identity]:
                raise ValueError('Result/decision lacks a unique preceding attempt')
            else:
                if kind == 'result':
                    for name in ['score','pre_gate_score','sharpe_per_period']:
                        value = event[name]
                        if value is not None and (isinstance(value,bool) or not isinstance(value,(int,float)) or not np.isfinite(value)):
                            raise ValueError('Invalid numeric trial result')
                    if event['status'] not in {'accepted','rejected','nondeterministic','timeout','crash'}:
                        raise ValueError('Unknown trial result status')
                trials[identity][kind] = event
        except (ValueError, KeyError, TypeError, json.JSONDecodeError) as exc:
            raise DataContractError('Malformed or inconsistent trial ledger') from exc
    sharpes = [trial['result']['sharpe_per_period'] for trial in trials.values()
               if trial.get('result',{}).get('sharpe_per_period') is not None]
    return {'count': len(trials), 'sharpes': sharpes, 'trials': trials}
