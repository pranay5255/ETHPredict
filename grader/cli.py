"""Frozen grader entry point. Only production grades are loop-eligible."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import pwd
import sys
import uuid

from .baseline_bank import build_bank, read_bank
from .baselines import NAMES, materialize
from .data import DataContractError
from .evaluation_data import EvaluationData
from .grade import _evaluate
from .history import read_trials
from .models import cached_model_paths


def _models(values):
    result = {}
    for value in values:
        key,separator,path = value.partition('=')
        if not separator or not key or not path or key in result:
            raise DataContractError('Use a unique --model-path MODEL_ID=DIRECTORY')
        result[key] = path
    return result


def _private(args):
    if args.tier=='shadow':
        try:
            uid = pwd.getpwnam('ethgrader').pw_uid
        except KeyError as exc:
            raise DataContractError('Shadow grading requires the provisioned private service') from exc
        root = Path('/srv/ethpredict-grader')
        if os.geteuid()!=uid or not Path(args.output).resolve().is_relative_to(root):
            raise DataContractError('Shadow inputs and every artifact must remain inside the ethgrader service')


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command',required=True)
    for name in ['grade','baseline','baselines']:
        command = commands.add_parser(name)
        command.add_argument('--lake',default='data/lake/lighter')
        command.add_argument('--output')
        command.add_argument('--tier',choices=['dev','shadow'],default='dev')
        command.add_argument('--private-lake')
        command.add_argument('--model-path',action='append',default=[])
        command.add_argument('--device',choices=['cpu','cuda'],default='cuda')
        if name=='grade':
            command.add_argument('--strategy',required=True)
            command.add_argument('--baseline-bank',default='artifacts/grader-baselines/baseline_bank.json')
            command.add_argument('--ledger',default='autoresearch/ledger.jsonl')
            command.add_argument('--trial-id')
        else:
            command.add_argument('--markets',nargs='+',default=['ETH'])
            command.add_argument('--resolution',choices=['1m','5m','1h'],default='1h')
            if name=='baseline':
                command.add_argument('--name',choices=NAMES,required=True)
    args = parser.parse_args(argv)
    if args.output is None:
        root = '/srv/ethpredict-grader/runs' if args.tier=='shadow' else 'artifacts/grader'
        args.output = str(Path(root)/str(uuid.uuid4()))
    _private(args)
    models = {**cached_model_paths(),**_models(args.model_path)}
    output = Path(args.output).resolve()
    if args.command=='baselines':
        bank = build_bank(args.lake,output,markets=args.markets,resolution=args.resolution,
            model_paths=models,tier=args.tier,private_root=args.private_lake,device=args.device,
            progress=lambda line:print(line,flush=True))
        print(json.dumps({'baseline_bank':str(output/'baseline_bank.json'),'complete':bank['complete']}))
        return 0 if bank['complete'] else 1
    if args.command=='baseline':
        source = materialize(args.name,output/'baseline-source',markets=args.markets,resolution=args.resolution)
        result = _evaluate(source,output/'grade',lambda declaration:EvaluationData.load(args.lake,declaration,
            tier=args.tier,private_root=args.private_lake),mode='baseline_calibration',tier=args.tier,
            model_paths=models,device=args.device)
    else:
        # Missing/bad bank or ledger is an integrity failure inside the report
        # path, not permission to compare against an invented zero baseline.
        bank_holder = {}
        preflight_error = None
        def factory(declaration):
            if preflight_error is not None:
                raise DataContractError(preflight_error)
            bank = read_bank(args.baseline_bank);bank_holder['bank'] = bank
            data = EvaluationData.load(args.lake,declaration,tier=args.tier,private_root=args.private_lake)
            from .grade import data_identity
            if bank['data_sha256']!=data_identity(data)['sha256'] or bank['resolution']!=declaration.resolution or set(bank['markets'])!=set(declaration.markets):
                raise DataContractError('Baseline bank does not match every declared market, clock and source block')
            return data
        try:
            bank = read_bank(args.baseline_bank)
            history = read_trials(args.ledger)
        except Exception as exc:
            preflight_error = f'Baseline/ledger preflight failed: {type(exc).__name__}: {exc}'
            bank,history = {},None
        result = _evaluate(args.strategy,output,factory,mode='production',tier=args.tier,
            model_paths=models,trial_id=args.trial_id,history=history,device=args.device,
            baseline_count=bank.get('trial_count',0),baseline_sharpes=bank.get('trial_sharpes',()),
            baseline_net_by_seed=bank.get('best_net_by_seed',{}))
    print(f'score: {result.report["score"]:.12f}')
    print(f'status: {result.report["status"]}')
    return 0


if __name__=='__main__':
    raise SystemExit(main())
