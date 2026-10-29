"""Full evaluation orchestration. Candidate imports stay in the worker process."""
from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess
import uuid

import numpy as np
import pandas as pd

from .checks import CheckResult, causality_check, determinism_check, verify_frozen_grader
from .contracts import regular_bytes
from .data import DataContractError, RESOLUTIONS
from .evaluation_data import EvaluationData
from .history import read_trials
from .models import block_contamination, contamination, stage_models
from .report import save_evidence, write_report
from .runner import StrategyRunner, WorkerFailure
from .score import block_evidence, component_score, economic_report, load_rubric
from .tiers import load_tiers
from .uncertainty import score_standard_error
from .policy_importance import policy_fold_importance, summarize_policy_importance
from .artifacts import capture_artifacts


def now():
    return datetime.now(timezone.utc).isoformat()


def source_identity(root):
    root = Path(root)
    records = {}
    for parent in ['src','strategy']:
        if not (root/parent).exists():
            continue
        for path in sorted((root/parent).rglob('*')):
            if path.is_file() and '__pycache__' not in path.parts:
                if path.is_symlink():
                    raise DataContractError('Candidate identity contains symlinks')
                records[str(path.relative_to(root))] = hashlib.sha256(regular_bytes(path,maximum=32*1024**2)).hexdigest()
    content = json.dumps(records,sort_keys=True).encode()
    return {'sha256':hashlib.sha256(content).hexdigest(),'files':records}


def data_identity(data):
    # Side-data opt-in does not change the common execution/baseline population.
    identity = {market:{key:datasets[key].provenance for key in [f'candles_{data.resolution}','fundings_1h']}
                for market,datasets in data.datasets.items()}
    return {'sha256':hashlib.sha256(json.dumps(identity,sort_keys=True).encode()).hexdigest(),
            'tier':data.tier,'start_ms':data.start_ms,'end_ms':data.end_ms,
            'resolution':data.resolution,'markets':list(data.datasets),'provenance':identity,
            'input_provenance':{market:{key:item.provenance for key,item in datasets.items()}
                                for market,datasets in data.datasets.items()},
            'execution_tail':'final source bar reserved for liquidation before the tier endpoint'}


@dataclass
class EvaluationResult:
    report: dict
    seed_blocks: list


def rescore(result, *, trial_sharpes, trial_count, baseline_net_by_seed, compute_uncertainty=True):
    report, blocks = result.report, result.seed_blocks
    rubric = load_rubric()
    report['metrics'],report['economics'] = [],[]
    if len(blocks) != 2 or any(len(seed) != 5 for seed in blocks):
        return report
    for seed, evidence in zip(report['seeds'],blocks):
        metrics = component_score(evidence,trial_sharpes=trial_sharpes,trial_count=trial_count,rubric=rubric)
        metrics['seed'] = seed
        metrics['blocks'] = [{name:getattr(block,name) for name in ['block','net_1x_bps','net_2x_bps','turnover','trades','daily_sharpe']} for block in evidence]
        report['metrics'].append(metrics)
        economics = economic_report(evidence,best_baseline_net_bps=baseline_net_by_seed.get(str(seed)),
            engineered_features=report['strategy']['declaration']['engineered_features'],
            importance=next((row['report'] for row in report['feature_importance'] if row['seed']==seed),{}),rubric=rubric)
        economics['seed'] = seed
        report['economics'].append(economics)
    report['pre_gate_score'] = float(np.mean([row['pre_gate_score'] for row in report['metrics']]))
    # Rescoring can replace economic/statistical failures, never integrity ones.
    report['failures'] = [item for item in report['failures'] if not item['check'].startswith(('economic:','statistics:'))]
    for row in report['economics']:
        report['failures'] += [{'check':'economic:'+name,'message':f'seed {row["seed"]}: {name}'} for name in row['failures']]
    for row in report['metrics']:
        if row['unavailable_components']:
            report['failures'].append({'check':'statistics:available','message':f'seed {row["seed"]}: unavailable {row["unavailable_components"]}'})
    if compute_uncertainty:
        report['uncertainty'] = score_standard_error(blocks,trial_sharpes=trial_sharpes,trial_count=trial_count,rubric=rubric)
    passed = not report['failures'] and all(item['passed'] for item in report['integrity'])
    report['score'] = report['pre_gate_score'] if passed else 0.
    if report['status'] not in {'nondeterministic','timeout'}:
        report['status'] = 'accepted' if passed else 'rejected'
    return report


def _evaluate(candidate, directory, data_factory, *, mode, tier, model_paths=None, frozen_check=None,
              trial_id=None, history=None, baseline_count=0, baseline_sharpes=(), baseline_net_by_seed=None,
              baseline_provider=None, compute_uncertainty=True, device='cuda'):
    """Shared production/calibration/fixture path; the CLI controls the mode.

    Only production is eligible for the loop. Calibration materializes known
    baseline source; fixture calls use explicitly synthetic data in tests.
    """
    directory = Path(directory).resolve();directory.mkdir(parents=True)
    rubric = load_rubric()
    report = {'version':1,'trial_id':trial_id or str(uuid.uuid4()),'mode':mode,'tier':tier,'status':'rejected',
        'score':0.,'pre_gate_score':0.,'started_at_utc':now(),'finished_at_utc':'',
        'grader':{'tag':rubric['frozen_tag'],'calibration':rubric['calibration']},'strategy':{},'data':{},
        'seeds':rubric['seeds'],'integrity':[],'economics':[],'metrics':[],'feature_importance':[],'resources':{},'uncertainty':None,
        'failures':[],'artifacts':{'seed_blocks':[],'manifest':None},'qualitative':{'status':'not_run','tie_breaker_only':True}}
    result = EvaluationResult(report,[])
    frozen_check = frozen_check or verify_frozen_grader(Path(__file__).resolve().parent.parent)
    report['integrity'].append(asdict(frozen_check))
    if not frozen_check.passed:
        report['failures'].append({'check':'grader_hash','message':'Frozen grader hashes do not match grader-v1'})
    stage,runner = 'setup',None
    try:
        if mode == 'production' and (not frozen_check.passed or rubric['calibration'] != 'complete'):
            raise WorkerFailure('Production grading requires the frozen calibrated tag',reasons=['grader_hash'])
        data_holder = {}
        def view(cutoff):
            return data_holder['data'].view(directory/'causal-lake-view',cutoff)
        with StrategyRunner(candidate,directory/'worker',seed=rubric['seeds'][0],device=device,
                            registered_model_paths=model_paths or {},lake_view=view) as runner:
            report['strategy']['source'] = source_identity(runner.candidate)
            readme = runner.candidate/'strategy/README.md'
            report['strategy']['pre_run_hypothesis'] = regular_bytes(readme,maximum=1_048_576).decode() if readme.exists() else None
            stage = 'declaration'
            declaration = runner.declare()
            models = stage_models(declaration.pretrained_models,model_paths or {},directory/'registered-models')
            runner.model_paths = {key:str(path) for key,path in models.items()}
            report['strategy']['declaration'] = asdict(declaration)
            stage = 'data'
            data = data_factory(declaration);data_holder['data'] = data
            report['data'] = data_identity(data)
            report['strategy']['models'] = contamination(declaration.pretrained_models,start_ms=data.start_ms,
                end_ms=data.end_ms,holdout_start_ms=load_tiers()['holdout'].start_ms)
            if not declaration.pretrained_models:
                report['strategy']['models'] = [{'initialization':'from scratch or analytic rule',
                    'external_checkpoint':False,'license':'project source; no external model-weight license asserted'}]
            blocks = data.blocks(declaration)
            clock = next(iter(data.candles.datasets.values())).frame.timestamp_ms.to_numpy()
            report['strategy']['models_by_block'] = block_contamination(declaration.pretrained_models,blocks,clock,
                resolution=declaration.resolution,max_horizon_bars=declaration.max_horizon_bars,
                holdout_start_ms=load_tiers()['holdout'].start_ms)
            for seed in rubric['seeds']:
                runner.seed = seed
                evidence,paths,importance_folds = [],[],[]
                for block in blocks:
                    block_number = block.block+1
                    block_dir = directory/f'seed-{seed}'/f'block-{block_number}'
                    raw_train = data.market_slice(np.arange(block.train[-1]+declaration.max_horizon_bars+1))
                    replay_train = data.market_slice(np.arange(block.train[-1]+2))
                    train_replay = data.replay(replay_train,block_dir/'train-replay',training=True)
                    permitted = data.market_slice(block.history)
                    times = pd.DatetimeIndex(pd.to_datetime(clock[block.evaluation]+RESOLUTIONS[declaration.resolution],unit='ms',utc=True))
                    stage = 'fit'
                    state = runner.fit(raw_train,replay=train_replay,train_anchor_end_ms=int(clock[block.train[-1]]))
                    stage = 'schema'
                    positions = runner.predict(state,permitted,times,declaration)
                    packet = runner.last_features
                    if block_number == 1:
                        stage = 'determinism'
                        check = determinism_check(runner,raw_train,permitted,times,declaration,positions,
                            replay=train_replay,train_anchor_end_ms=int(clock[block.train[-1]]))
                        report['integrity'].append(asdict(check))
                        if not check.passed:
                            raise WorkerFailure('Same-seed repeat fit/predict differs',status='nondeterministic',reasons=['determinism'])
                    stage = 'causality'
                    check = causality_check(runner,state,permitted,times,declaration,positions,block=block_number,seed=seed,
                        cutoffs=rubric['integrity']['causality_cutoffs_per_block'],tolerance=rubric['integrity']['position_tolerance'],
                        reference_features=packet)
                    report['integrity'].append(asdict(check))
                    if not check.passed:
                        raise WorkerFailure('Proper-prefix predictions differ',reasons=['causality'])
                    stage = 'accounting'
                    replay = data.replay(data.market_slice(np.r_[block.evaluation,block.evaluation[-1]+1]),block_dir/'score-replay')
                    targets = positions.pivot(index='timestamp',columns='market',values='target_position').reindex(index=times,columns=replay.markets).to_numpy()
                    scored = block_evidence(replay,targets,block=block_number)
                    if declaration.engineered_features:
                        stage = 'feature_importance'
                        importance = policy_fold_importance(runner,state,packet,declaration,positions,replay,block)
                        importance_folds.append(importance)
                        (block_dir/'feature_importance.json').write_text(json.dumps(importance,allow_nan=False))
                    evidence.append(scored)
                    positions.to_parquet(block_dir/'positions.parquet',index=False)
                    (block_dir/'split_audit.json').write_text(json.dumps(block.audit,allow_nan=False))
                    paths.append(save_evidence(scored,block_dir/'evidence'))
                result.seed_blocks.append(evidence)
                report['artifacts']['seed_blocks'].append(paths)
                if declaration.engineered_features:
                    report['feature_importance'].append({'seed':seed,'report':summarize_policy_importance(importance_folds)})
            report['integrity'] += [asdict(CheckResult(name,True,{'calls':len(runner.calls)})) for name in ['schema','tier_isolation','budget']]
        if baseline_provider is not None:
            # Release the candidate's CUDA reservation before trusted baseline
            # initialization. These independent comparisons are not candidate
            # fit/predict calls and cannot consume or reduce its VRAM budget.
            stage = 'baseline'
            bank = baseline_provider(declaration,data)
            baseline_count,baseline_sharpes = bank['trial_count'],bank['trial_sharpes']
            baseline_net_by_seed = bank['best_net_by_seed']
            (directory/'baseline_bank.json').write_text(json.dumps(bank,sort_keys=True,allow_nan=False)+'\n')
            report['grader']['baseline'] = {key:bank[key] for key in ['grading_fingerprint','data_sha256',
                'markets','resolution','tier','trial_count','winner','best_net_by_seed']}
        history = history or {'count':0,'sharpes':[],'trials':{}}
        existing = history['trials'].get(report['trial_id'],{})
        count = baseline_count+history['count']+(0 if existing else 1)
        current = [component_score(seed,trial_sharpes=[],trial_count=1)['daily_sharpe_per_period'] for seed in result.seed_blocks]
        known_current = [row['value'] for row in current if row['available']]
        sharpes = list(baseline_sharpes)+list(history['sharpes'])
        if known_current and 'result' not in existing:
            sharpes.append(float(np.mean(known_current)))
        rescore(result,trial_sharpes=sharpes,trial_count=count,
                baseline_net_by_seed=baseline_net_by_seed or {},compute_uncertainty=compute_uncertainty)
    except Exception as exc:
        status = getattr(exc,'status','rejected')
        report['status'] = status if status in {'nondeterministic','timeout'} else 'rejected'
        report['score'] = 0.
        reasons = getattr(exc,'reasons',()) or (stage,)
        report['failures'] += [{'check':str(reason),'message':f'{type(exc).__name__}: {exc}'} for reason in reasons
                               if not any(item['check']==str(reason) for item in report['failures'])]
        if not any(item['name']==stage and not item['passed'] for item in report['integrity']):
            report['integrity'].append(asdict(CheckResult(stage,False,{'error':str(exc)})))
    finally:
        if runner is not None:
            report['resources'] = {'wall_seconds':runner.elapsed,'wall_limit_seconds':runner.limit,
                'device':runner.device,'vram_reservation':runner.gpu_reservation,
                'peak_rss_bytes':max([0,*[call.peak_rss_bytes for call in runner.calls]]),
                'peak_vram_bytes':max([0,*[call.peak_vram_bytes for call in runner.calls]]),
                'calls':[{'operation':call.operation,'status':call.status,'duration_seconds':call.duration_seconds,
                          'peak_rss_bytes':call.peak_rss_bytes,'peak_vram_bytes':call.peak_vram_bytes,
                          'audit':asdict(call.audit)} for call in runner.calls],
                'scope':'one combined wall budget across both seeds, all blocks, declaration and integrity reruns'}
        try:
            manifest = capture_artifacts(directory)
            report['artifacts']['manifest'] = manifest
            report['integrity'].append(asdict(CheckResult('artifact_integrity',True,manifest)))
        except Exception as exc:
            report['score'] = 0.
            if report['status'] == 'accepted':
                report['status'] = 'rejected'
            report['failures'].append({'check':'artifact_integrity','message':f'{type(exc).__name__}: {exc}'})
            report['integrity'].append(asdict(CheckResult('artifact_integrity',False,{'error':str(exc)})))
        report['finished_at_utc'] = now()
        write_report(report,directory/'grade_report.json')
    return result
