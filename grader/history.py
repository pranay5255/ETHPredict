"""Strict append-only trial-event accounting, including interrupted attempts."""
from __future__ import annotations

import json
from datetime import datetime, timedelta
from pathlib import Path
import re

import numpy as np

from .data import DataContractError, _json_unique


EVENT_FIELDS = {
    'attempt': {'version','event','trial_id','timestamp_utc','commit','description','hypothesis_sha256','candidate_sha256'},
    'result': {'version','event','trial_id','timestamp_utc','status','score','pre_gate_score','sharpe_per_period','report_sha256','report_path','trackio_run_id'},
    'decision': {'version','event','trial_id','timestamp_utc','decision','incumbent_commit','incumbent_rescored_score','candidate_score','delta','qualitative'},
}


def _finite_number(value, *, nullable=False, score=False):
    if value is None and nullable:
        return
    if isinstance(value,bool) or not isinstance(value,(int,float)) or not np.isfinite(value):
        raise ValueError('Invalid numeric trial field')
    if score and not 0 <= value <= 1:
        raise ValueError('Trial score outside [0,1]')


def _hash(value):
    if value is not None and (not isinstance(value,str) or not re.fullmatch('[a-f0-9]{64}',value)):
        raise ValueError('Invalid trial artifact hash')


def _json_number(token):
    value = float(token)
    _finite_number(value)
    return value


def read_trials(path):
    path = Path(path)
    if not path.exists():
        return {'count': 0, 'sharpes': [], 'trials': {}}
    trials = {}
    for line in path.read_text().splitlines():
        try:
            event = json.loads(line, object_pairs_hook=_json_unique,
                parse_float=_json_number,
                parse_constant=lambda value: (_ for _ in ()).throw(ValueError('Nonfinite JSON')))
            kind = event['event']
            if kind not in EVENT_FIELDS or set(event) != EVENT_FIELDS[kind] or type(event['version']) is not int or event['version'] != 1:
                raise ValueError('Unknown event fields/version')
            identity = event['trial_id']
            if not isinstance(identity,str) or not identity:
                raise ValueError('Invalid trial identity')
            stamp = datetime.fromisoformat(event['timestamp_utc'].replace('Z','+00:00'))
            if stamp.utcoffset() != timedelta(0):
                raise ValueError('Trial event timestamp must be explicit UTC')
            if kind == 'attempt':
                if identity in trials:
                    raise ValueError('Repeated trial attempt')
                if (event['commit'] is not None and not isinstance(event['commit'],str)
                        or not isinstance(event['description'],str)):
                    raise ValueError('Invalid attempt metadata')
                _hash(event['hypothesis_sha256']);_hash(event['candidate_sha256'])
                trials[identity] = {'attempt': event}
            elif identity not in trials or kind in trials[identity]:
                raise ValueError('Result/decision lacks a unique preceding attempt')
            else:
                if kind == 'result':
                    _finite_number(event['score'],score=True)
                    _finite_number(event['pre_gate_score'],nullable=True,score=True)
                    _finite_number(event['sharpe_per_period'],nullable=True)
                    if event['status'] not in {'accepted','rejected','nondeterministic','timeout','crash'}:
                        raise ValueError('Unknown trial result status')
                    if event['status'] != 'accepted' and event['score'] != 0:
                        raise ValueError('Failed trial has a nonzero final score')
                    _hash(event['report_sha256'])
                    for field in ['report_path','trackio_run_id']:
                        if event[field] is not None and not isinstance(event[field],str):
                            raise ValueError('Invalid result artifact metadata')
                else:
                    if 'result' not in trials[identity]:
                        raise ValueError('Decision precedes its trial result')
                    if event['decision'] not in {'initialize','keep','discard'}:
                        raise ValueError('Invalid trial decision')
                    if event['incumbent_commit'] is not None and not isinstance(event['incumbent_commit'],str):
                        raise ValueError('Invalid incumbent commit')
                    for field in ['incumbent_rescored_score','candidate_score']:
                        _finite_number(event[field],nullable=True,score=True)
                    _finite_number(event['delta'],nullable=True)
                    if event['delta'] is not None and event['delta'] < 0:
                        raise ValueError('Negative score uncertainty')
                    if event['candidate_score'] is not None and event['candidate_score'] != trials[identity]['result']['score']:
                        raise ValueError('Decision score differs from trial result')
                    if event['decision']=='keep' and (trials[identity]['result']['status']!='accepted' or
                            any(event[field] is None for field in ['incumbent_rescored_score','candidate_score','delta']) or
                            not event['candidate_score'] > event['incumbent_rescored_score']+event['delta']):
                        raise ValueError('Keep decision violates the strict numeric rule')
                    qualitative = event['qualitative']
                    if qualitative is not None:
                        ratings = {'hypothesis','simplicity','mechanism_for_constants','no_hardcoded_regimes'}
                        if not isinstance(qualitative,dict) or set(qualitative) != ratings|{'evidence'}:
                            raise ValueError('Unknown qualitative decision fields')
                        for field in ratings:
                            _finite_number(qualitative[field],score=True)
                        notes = qualitative['evidence']
                        if not isinstance(notes,dict) or set(notes)!=ratings or any(not isinstance(note,str) for note in notes.values()):
                            raise ValueError('Invalid qualitative decision evidence')
                trials[identity][kind] = event
        except (ValueError, KeyError, TypeError, AttributeError, json.JSONDecodeError) as exc:
            raise DataContractError('Malformed or inconsistent trial ledger') from exc
    sharpes = [trial['result']['sharpe_per_period'] for trial in trials.values()
               if trial.get('result',{}).get('sharpe_per_period') is not None]
    return {'count': len(trials), 'sharpes': sharpes, 'trials': trials}
