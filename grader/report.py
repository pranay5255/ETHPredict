"""Strict report envelope and numeric evidence persistence for later rescoring."""
from __future__ import annotations

from dataclasses import asdict
import json
from pathlib import Path

import jsonschema
import numpy as np
import pandas as pd

from .contracts import regular_bytes
from .data import DataContractError, _json_unique
from .models import sha256_file
from .score import BlockEvidence


FIELDS = {'version','trial_id','mode','tier','status','score','pre_gate_score','started_at_utc','finished_at_utc','feature_importance',
          'grader','strategy','data','seeds','integrity','economics','metrics','resources','uncertainty',
          'failures','artifacts','qualitative'}
SCHEMA = {'type':'object','additionalProperties':False,'required':sorted(FIELDS),'properties':{
    'version':{'const':1}, 'trial_id':{'type':'string','minLength':1},
    'mode':{'enum':['production','baseline_calibration','fixture']}, 'tier':{'enum':['dev','shadow']},
    'status':{'enum':['accepted','rejected','nondeterministic','timeout']},
    'score':{'type':'number','minimum':0,'maximum':1}, 'pre_gate_score':{'type':'number','minimum':0,'maximum':1},
    'started_at_utc':{'type':'string'},'finished_at_utc':{'type':'string'},
    'grader':{'type':'object'},'strategy':{'type':'object'},'data':{'type':'object'},
    'seeds':{'type':'array','items':{'type':'integer'},'minItems':2,'maxItems':2,'uniqueItems':True},
    'integrity':{'type':'array','items':{'type':'object','additionalProperties':False,
        'required':['name','passed','details'],'properties':{'name':{'type':'string'},'passed':{'type':'boolean'},'details':{'type':'object'}}}},
    'economics':{'type':'array','items':{'type':'object'}},'metrics':{'type':'array','items':{'type':'object'}},
    'feature_importance':{'type':'array','items':{'type':'object'}},
    'resources':{'type':'object'},'uncertainty':{'type':['object','null']},
    'failures':{'type':'array','items':{'type':'object','additionalProperties':False,'required':['check','message'],
        'properties':{'check':{'type':'string'},'message':{'type':'string'}}}},
    'artifacts':{'type':'object','additionalProperties':False,'required':['seed_blocks','manifest'],'properties':{
        'seed_blocks':{'type':'array','maxItems':2,'items':{'type':'array','maxItems':5,'items':{'type':'string'}}},
        'manifest':{'oneOf':[{'type':'null'},{'type':'object','additionalProperties':False,
            'required':['path','sha256','files','bytes'],'properties':{'path':{'const':'artifact_manifest.json'},
            'sha256':{'type':'string','pattern':'^[a-f0-9]{64}$'},'files':{'type':'integer','minimum':0},
            'bytes':{'type':'integer','minimum':0}}}]}}},
    'qualitative':{'type':'object','additionalProperties':False,'required':['status','tie_breaker_only'],
        'properties':{'status':{'const':'not_run'},'tie_breaker_only':{'const':True}}}}}


def write_report(value, path):
    jsonschema.validate(value, SCHEMA)
    # JSON, unlike Python float, has no NaN/Infinity. Fail instead of sanitizing.
    content = json.dumps(value, indent=2, sort_keys=True, allow_nan=False)
    Path(path).write_text(content+'\n')


def read_report(path):
    def finite_number(token):
        value = float(token)
        if not np.isfinite(value):
            raise DataContractError('Nonfinite report JSON')
        return value
    value = json.loads(regular_bytes(Path(path), maximum=32*1024**2), object_pairs_hook=_json_unique,
        parse_float=finite_number,
        parse_constant=lambda token: (_ for _ in ()).throw(DataContractError('Nonfinite report JSON')))
    jsonschema.validate(value, SCHEMA)
    if not np.isfinite([value['score'],value['pre_gate_score']]).all():
        raise DataContractError('Nonfinite report score')
    if value['status'] != 'accepted' and value['score'] != 0:
        raise DataContractError('Rejected report has a nonzero final score')
    return value


def save_evidence(block, directory):
    directory = Path(directory);directory.mkdir(parents=True)
    block.daily.to_parquet(directory/'daily.parquet')
    np.savez(directory/'cashflows.npz', **block.cashflows, cashflow_dates=block.cashflow_dates)
    metadata = {name:getattr(block,name) for name in ['block','markets','net_1x_bps','net_2x_bps','turnover','trades','daily_sharpe']}
    metadata['files'] = {name:sha256_file(directory/name) for name in ['daily.parquet','cashflows.npz']}
    (directory/'evidence.json').write_text(json.dumps(metadata, allow_nan=False))
    return str(directory)


def load_evidence(directory):
    directory = Path(directory)
    metadata = json.loads(regular_bytes(directory/'evidence.json', maximum=65_536), object_pairs_hook=_json_unique)
    fields = {'block','markets','net_1x_bps','net_2x_bps','turnover','trades','daily_sharpe','files'}
    if set(metadata) != fields or set(metadata['files']) != {'daily.parquet','cashflows.npz'}:
        raise DataContractError('Unknown evidence fields')
    for name, digest in metadata.pop('files').items():
        if (directory/name).is_symlink() or sha256_file(directory/name) != digest:
            raise DataContractError('Evidence hash mismatch')
    daily = pd.read_parquet(directory/'daily.parquet')
    if set(daily) != {'net_1x_bps','net_2x_bps','turnover','trades'} or not np.isfinite(daily.to_numpy()).all():
        raise DataContractError('Invalid daily evidence')
    with np.load(directory/'cashflows.npz', allow_pickle=False) as arrays:
        cashflows = {name:arrays[name] for name in arrays.files if name != 'cashflow_dates'}
        dates = arrays['cashflow_dates']
    if not cashflows or any(not np.isfinite(array).all() or len(array) != len(dates) for array in cashflows.values()):
        raise DataContractError('Invalid cashflow evidence')
    metadata['markets'] = tuple(metadata['markets'])
    return BlockEvidence(**metadata, daily=daily, cashflows=cashflows, cashflow_dates=dates)
