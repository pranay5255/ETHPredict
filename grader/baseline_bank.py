"""Comparable baseline bank built exclusively by the trusted grader."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import subprocess

import numpy as np

from .baselines import NAMES, materialize
from .artifacts import verify_artifacts
from .data import DataContractError, _json_unique
from .evaluation_data import EvaluationData
from .grade import _evaluate, rescore
from .models import sha256_file
from .report import read_report, write_report
from .score import load_rubric


def grading_fingerprint():
    root = Path(__file__).resolve().parent
    files = {str(path.relative_to(root)):sha256_file(path) for path in sorted(root.rglob('*'))
             if path.is_file() and '__pycache__' not in path.parts and path.name != 'rubric.yaml'}
    config = load_rubric()
    options = {key:value for key,value in config.items() if key not in {'components','calibration'}}
    repository = root.parent
    # Candidate baselines import the source library only in their subprocess;
    # its immutable Git tree identity is part of the frozen comparison contract.
    reference = 'grader-v1' if subprocess.run(['git','-C',str(repository),'rev-parse','--verify','grader-v1'],
        capture_output=True).returncode == 0 else 'HEAD'
    source_tree = subprocess.check_output(['git','-C',str(repository),'rev-parse',f'{reference}:src'],text=True).strip()
    return hashlib.sha256(json.dumps({'files':files,'options':options,'src_tree':source_tree},sort_keys=True).encode()).hexdigest()


def build_bank(lake, directory, *, markets, resolution, model_paths=None, tier='dev', private_root=None,
               device='cuda', progress=None):
    directory = Path(directory).resolve();directory.mkdir(parents=True)
    results = {}
    for name in NAMES:
        if progress:
            progress(f'baseline {name}: starting')
        source = materialize(name,directory/name/'source',markets=markets,resolution=resolution)
        result = _evaluate(source,directory/name/'grade',
            lambda declaration:EvaluationData.load(lake,declaration,tier=tier,private_root=private_root),
            mode='baseline_calibration',tier=tier,model_paths=model_paths or {},device=device,
            compute_uncertainty=False)
        results[name] = result
        if progress:
            progress(f'baseline {name}: {result.report["status"]}; {len(result.seed_blocks)} complete seed paths')
    complete = all(len(result.seed_blocks)==2 and all(len(blocks)==5 for blocks in result.seed_blocks)
        and all(row['passed'] for row in result.report['integrity'] if row['name']!='grader_hash') for result in results.values())
    identities = {result.report['data'].get('sha256') for result in results.values()}
    if len(identities) != 1 or None in identities:
        complete = False
    known_sharpes = []
    by_baseline = {}
    for name,result in results.items():
        if len(result.report['metrics']) != 2:
            continue
        by_baseline[name] = {str(row['seed']):row['net_1x_bps'] for row in result.report['metrics']}
        available = [row['daily_sharpe_per_period']['value'] for row in result.report['metrics'] if row['daily_sharpe_per_period']['available']]
        if available:
            known_sharpes.append(float(np.mean(available)))
    winner = max(by_baseline,key=lambda name:np.mean(list(by_baseline[name].values()))) if complete else None
    threshold = by_baseline[winner] if winner is not None else {}
    records = []
    for name,result in results.items():
        if len(result.seed_blocks)==2:
            rescore(result,trial_sharpes=known_sharpes,trial_count=len(NAMES),baseline_net_by_seed=threshold)
            write_report(result.report,directory/name/'grade/grade_report.json')
        path = directory/name/'grade/grade_report.json'
        records.append({'name':name,'report':str(path.relative_to(directory)),'sha256':sha256_file(path)})
    bank = {'version':1,'complete':complete,'data_sha256':next(iter(identities)) if len(identities)==1 else None,
            'grading_fingerprint':grading_fingerprint(),'markets':list(markets),'resolution':resolution,'tier':tier,
            'trial_count':len(NAMES),'trial_sharpes':known_sharpes,'winner':winner,
            'best_net_by_seed':threshold,'records':records}
    (directory/'baseline_bank.json').write_text(json.dumps(bank,indent=2,allow_nan=False)+'\n')
    return bank


def read_bank(path):
    path = Path(path).resolve(strict=True)
    value = json.loads(path.read_text(),object_pairs_hook=_json_unique)
    fields = {'version','complete','data_sha256','grading_fingerprint','markets','resolution','tier',
              'trial_count','trial_sharpes','winner','best_net_by_seed','records'}
    if set(value)!=fields or type(value['version']) is not int or value['version']!=1 or value['complete'] is not True:
        raise DataContractError('Baseline bank is incomplete or has unknown fields')
    if value['grading_fingerprint']!=grading_fingerprint() or value['trial_count']!=len(NAMES):
        raise DataContractError('Baseline bank was produced by a different grading contract')
    if {row['name'] for row in value['records']}!=set(NAMES) or len(value['records'])!=len(NAMES):
        raise DataContractError('Baseline bank omits required comparisons')
    by_baseline,known_sharpes = {},{}
    seeds = load_rubric()['seeds']
    for row in value['records']:
        if set(row)!={'name','report','sha256'}:
            raise DataContractError('Unknown baseline record fields')
        report_path = (path.parent/row['report']).resolve(strict=True)
        if not report_path.is_relative_to(path.parent) or sha256_file(report_path)!=row['sha256']:
            raise DataContractError('Baseline report escapes bank or has changed')
        report = read_report(report_path)
        verify_artifacts(report_path.parent,report['artifacts']['manifest'])
        if report['data']['sha256']!=value['data_sha256'] or not all(item['passed'] for item in report['integrity'] if item['name']!='grader_hash'):
            raise DataContractError('Baseline report has different data or failed integrity')
        declaration = report['strategy']['declaration']
        required = {'grader_hash','causality','determinism','schema','tier_isolation','budget','artifact_integrity'}
        if (report['mode']!='baseline_calibration' or report['tier']!=value['tier']
                or declaration['resolution']!=value['resolution'] or set(declaration['markets'])!=set(value['markets'])
                or not required.issubset({item['name'] for item in report['integrity']})
                or len(report['metrics'])!=2 or [item['seed'] for item in report['metrics']]!=seeds):
            raise DataContractError('Baseline report lacks the required complete comparison contract')
        for metric in report['metrics']:
            if (metric['trial_count']!=value['trial_count'] or len(metric['blocks'])!=5
                    or {block['block'] for block in metric['blocks']}!=set(range(1,6))):
                raise DataContractError('Baseline metrics omit blocks or use different trial accounting')
            for key in ['net_1x_bps','net_2x_bps','turnover','trades']:
                if not np.isclose(sum(block[key] for block in metric['blocks']),metric[key],rtol=0,atol=1e-9):
                    raise DataContractError('Baseline aggregate disagrees with its block evidence')
        by_baseline[row['name']] = {str(item['seed']):item['net_1x_bps'] for item in report['metrics']}
        available = [item['daily_sharpe_per_period']['value'] for item in report['metrics'] if item['daily_sharpe_per_period']['available']]
        if available:
            known_sharpes[row['name']] = float(np.mean(available))
    winner = max(NAMES,key=lambda name:np.mean(list(by_baseline[name].values())))
    sharpes = [known_sharpes[name] for name in NAMES if name in known_sharpes]
    if value['winner']!=winner or value['best_net_by_seed']!=by_baseline[winner] or value['trial_sharpes']!=sharpes:
        raise DataContractError('Baseline winner, threshold or Sharpe history disagrees with its reports')
    return value
