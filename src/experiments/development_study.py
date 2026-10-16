"""Frozen development-only forecast repair and range-break continuation study.

No final-test scorer is imported or called. All outcomes are validation evidence.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import pandas as pd
import torch
import yaml

from src.evaluation.cost_stress import cost_stress_report
from src.evaluation.economic_gate import economic_gate
from src.evaluation.feature_importance import permutation_meta, summarize_importance
from src.evaluation.forecast_diagnostics import calibration_summary
from src.experiments.local_batch import verify_prepared
from src.experiments.meta_labeling_mvp import (build_multi_horizon_lighter_dataset, build_validation_splits,
    train_base_model, predict_base_model, add_meta_labels, fit_meta_labeler, add_meta_probabilities,
    run_alpha_backtest, _write_json)
from src.simulator.costs import dataset_cost_model, configured_notional
from src.training.checkpoints import load_base_checkpoint
from src.training.devices import resolve_training_device
from src.validation.splits import purge_training_spans
from src.utils.trackio_logging import enforce_trackio_policy, log_trackio_run, _verify_local_run


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def forecast_scores(frame, horizon, prior):
    y = frame[f'{horizon}_true_return'].to_numpy(dtype=float)
    p = frame[f'{horizon}_pred_return'].to_numpy(dtype=float)
    direction = frame[f'{horizon}_direction_prob'].to_numpy(dtype=float)
    baseline = float(np.mean(y*y))
    mse = float(np.mean((y-p)**2))
    rank = pd.Series(y).corr(pd.Series(p), method='spearman') if np.ptp(p) and np.ptp(y) else None
    slope = float(np.cov(p,y,ddof=0)[0,1]/np.var(p)) if np.std(p)>1e-12 else None
    return {'rows':len(y), 'unit':'log_fraction', 'mse':mse, 'zero_mse':baseline,
        'mse_vs_zero':mse/baseline if baseline else None, 'mae':float(np.mean(abs(y-p))),
        'median_abs_prediction_bps':float(np.median(abs(p))*10000),
        'median_abs_realized_bps':float(np.median(abs(y))*10000),
        'return_rank_correlation':float(rank) if rank is not None and np.isfinite(rank) else None,
        'calibration_slope':slope,
        'direction':calibration_summary(y>0,direction),
        'training_prior_baseline':calibration_summary(y>0,np.full(len(y),prior))}


def opportunity_candidates(dataset, indices, config, volume_z_min):
    """Price breaks past a trailing range with volume surprise, using prior bars.

    expected_edge_bps=0 is a neutral scheduling score, NOT a return estimate.
    No arbitrary price forecast or synthetic confidence is used by the rule.
    """
    settings = config['development_study']['opportunity']
    lookback = int(settings['lookback_bars'])
    price = dataset['price_path'].copy()
    prior = price.shift(1)  # Most recent bar used ends before the entry bar.
    upper = price.high.rolling(lookback,min_periods=lookback).max().shift(2)
    lower = price.low.rolling(lookback,min_periods=lookback).min().shift(2)
    vol_mean = price.volume.rolling(lookback,min_periods=lookback).mean().shift(2)
    vol_std = price.volume.rolling(lookback,min_periods=lookback).std().shift(2)
    volume_z = (prior.volume-vol_mean)/vol_std.replace(0,np.nan)
    sides = np.where(prior.close>upper,1,np.where(prior.close<lower,-1,0))
    horizon = config['alpha_backtest']['horizon']
    bars = dataset['horizon_bars'][horizon]
    model = dataset_cost_model(dataset,config)
    frame = dataset['samples'].iloc[indices].reset_index(drop=True).copy()
    rows=[]
    for _,sample in frame.iterrows():
        i=int(sample.sample_index); side=int(sides[i]); z=float(volume_z.iloc[i])
        state=sample.to_dict()
        state.update(horizon=horizon,horizon_bars=bars,side=side,position_notional=configured_notional(config))
        quote=model.quote(state)
        state.update(quote)
        state.update(is_candidate=bool(side and np.isfinite(z) and z>=volume_z_min),
            expected_edge_bps=0., edge_estimate_status='not_estimated_rule_only',
            meta_prob=1., prediction_role='untrained_baseline',
            opportunity_volume_z=z if np.isfinite(z) else 0.,
            opportunity_range_width=float(np.log(upper.iloc[i]/lower.iloc[i])) if np.isfinite(upper.iloc[i]) else 0.,
            opportunity_breakout_distance=float(np.log(prior.close.iloc[i]/(upper.iloc[i] if side>=0 else lower.iloc[i]))) if np.isfinite(upper.iloc[i]) else 0.,
            opportunity_prior_return=float(np.log(prior.close.iloc[i]/price.close.iloc[max(0,i-13)])),
            opportunity_information_cutoff=str(pd.to_datetime(price.close_time.iloc[max(0,i-1)],unit='ms')) if 'close_time' in price else str(price.timestamp.iloc[max(0,i-1)]))
        # Timestamp strings are audit columns, never numeric meta-model features.
        state['information_cutoff_timestamp']=state.pop('opportunity_information_cutoff')
        for col in dataset.get('feature_columns',[]):
            if col.startswith('diagnostic_noise_'):
                state[col]=float(dataset['feature_frame'].iloc[max(0,i-1)][col])
        rows.append(state)
    return add_meta_labels(pd.DataFrame(rows),dataset,config)


def _write_progress(path, journal):
    _write_json(path, journal)
    print(json.dumps(journal[-1],default=str),flush=True)


def _run_study(config_path, output, *, device='cuda:0', allow_cpu=False):
    config_path, output=Path(config_path).resolve(),Path(output).resolve()
    frozen=verify_prepared(config_path)
    config=yaml.safe_load(config_path.read_text())
    settings=config['development_study']
    enforce_trackio_policy(config)
    if config.get('research',{}).get('final_test_guard',{}).get('evaluate') is not False:
        raise ValueError('Development study requires final_test_guard.evaluate=false')
    if output.exists():
        raise FileExistsError('Study output exists; inspect saved outcomes rather than silently rerunning')
    cutoff=pd.Timestamp(settings['excluded_from'],tz='UTC')
    raw_end=pd.Timestamp(frozen['source_end'])
    if raw_end>=cutoff:
        raise ValueError('Data overlaps excluded previous-study window')
    proposals=[{'id':f'volume_{z:g}_{"meta" if meta else "base"}', 'volume_z_min':z, 'meta':meta}
               for z in settings['opportunity']['volume_z_min'] for meta in [False,True]]
    if len(proposals)>settings['max_configurations']:
        raise ValueError('Declared signal configuration budget exceeded')
    seeds=list(settings['seeds'])
    if not seeds or len(set(seeds))!=len(seeds):
        raise ValueError('Seeds must be nonempty and unique')
    guard={str(Path(path).resolve()):sha(path) for path in settings.get('protected_artifacts',[])}
    git_status=__import__('subprocess').check_output(['git','status','--porcelain'],text=True)
    if git_status and not config.get('tracking',{}).get('trackio',{}).get('allow_local_debug_without_trackio',False):
        raise ValueError('Commit source changes before a real frozen study')
    output.mkdir(parents=True)
    _write_json(output/'protocol.json',{'config_sha256':sha(config_path),'configuration':config,
        'proposals':proposals,'seeds':seeds,'excluded_from':str(cutoff),'protected_hashes':guard,
        'git_commit':__import__('subprocess').check_output(['git','rev-parse','HEAD'],text=True).strip(),
        'git_status':git_status,'source_frozen_manifest':frozen,'final_test_evaluations':0})
    selected_device=resolve_training_device(device,allow_cpu=allow_cpu)
    dataset=build_multi_horizon_lighter_dataset(config)
    if pd.to_datetime(dataset['price_path'].timestamp,utc=True).max()>=cutoff:
        raise ValueError('Feature/label source clock overlaps excluded period')
    splits=build_validation_splits(dataset,config)
    folds=splits['folds']
    if config['validation']['method'] != 'exact_span_walk_forward':
        raise ValueError('Study requires chronological exact-span walk-forward blocks')
    if len(folds)<3:
        raise ValueError('Need at least three purged chronological blocks')
    _write_json(output/'splits.json',splits)
    journal=[];forecast_rows=[];trials=[];fit_counts={'forecast_members':0,'meta_fit_calls':0,'logistic_fits':0,'calibrators':0}
    # All comparators see identical outer populations, including no-skill baselines.
    for kind in settings['forecast_models']:
        for seed in seeds:
            local=deepcopy(config);local['training']['ensemble_seeds']=[seed]
            local['model']['base']['type']=kind
            for fold in folds:
                name=f'{kind}_seed{seed}_fold{fold["fold"]}';folder=output/'forecasts'/name;folder.mkdir(parents=True)
                started=time.perf_counter()
                journal.append({'stage':'forecast','id':name,'status':'running'})
                _write_progress(output/'attempts.json',journal)
                try:
                    model,history=train_base_model(dataset,fold['train'],local,selected_device)
                    fit_counts['forecast_members']+=1
                    pred=predict_base_model(model,dataset,fold['validation'],selected_device)
                    pred['prediction_role']='base_oof'
                    pred.to_parquet(folder/'predictions.parquet',index=False)
                    torch.save(model.state_dict(),folder/'model.pt')
                    restored=load_base_checkpoint(folder/'model.pt',local,dataset,selected_device)
                    roundtrip=predict_base_model(restored,dataset,fold['validation'],selected_device)
                    for horizon in dataset['horizon_names']:
                        for suffix in ['pred_return','direction_prob']:
                            np.testing.assert_allclose(pred[f'{horizon}_{suffix}'],roundtrip[f'{horizon}_{suffix}'],rtol=0,atol=0)
                    prior=dataset['y_dir'][fold['train']].mean(0).numpy()
                    scores={h:forecast_scores(pred,h,float(prior[i])) for i,h in enumerate(dataset['horizon_names'])}
                    row={'model':kind,'seed':seed,'fold':fold['fold'],'status':'completed',
                         'scores':scores,'checkpoint_roundtrip_exact':True,'path':str(folder),
                         'elapsed_seconds':time.perf_counter()-started}
                    _write_json(folder/'history.json',history);_write_json(folder/'metrics.json',row)
                    forecast_rows.append(row);journal[-1].update(status='completed',elapsed_seconds=row['elapsed_seconds'])
                    del model,restored
                except Exception as exc:
                    journal[-1].update(status='failed',error=str(exc));_write_progress(output/'attempts.json',journal)
                    raise
                _write_progress(output/'attempts.json',journal)
    candidate_cache = {z: opportunity_candidates(dataset, np.arange(splits['development_end']), config, z)
                       for z in settings['opportunity']['volume_z_min']}
    for proposal in proposals:
        for seed in seeds:
            local=deepcopy(config);local['training']['ensemble_seeds']=[seed];local['experiment']['seed']=seed
            torch.manual_seed(seed);np.random.seed(seed)
            name=f'{proposal["id"]}_seed{seed}';folder=output/'opportunities'/name;folder.mkdir(parents=True)
            journal.append({'stage':'opportunity','id':name,'status':'running'});_write_progress(output/'attempts.json',journal)
            candidates=candidate_cache[proposal['volume_z_min']]
            block_metrics=[];parts=[];importance_folds=[];audits=[]
            for fold in folds:
                train=candidates.iloc[fold['train']].copy();held=candidates.iloc[fold['validation']].copy()
                combined=pd.concat([train,held],ignore_index=True)
                retained,audit=purge_training_spans(combined,np.arange(len(train)),np.arange(len(train),len(combined)),
                    config['validation'],clock=dataset['price_path'].timestamp,max_horizon=dataset['max_horizon_bars'])
                train=train.iloc[retained]
                if proposal['meta']:
                    bundle=fit_meta_labeler(train,local,dataset['horizon_names'],selected_device,dataset=dataset)
                    fit_counts['meta_fit_calls']+=1
                    fit_counts['logistic_fits']+=int(bundle.kind=='logistic')
                    fit_counts['calibrators']+=int(bundle.manifest().get('calibrator') is not None)
                    held=add_meta_probabilities(held,bundle)
                    labeled=held[held.is_candidate & held.meta_label.notna()]
                    permutation=permutation_meta(bundle,labeled,repeats=3,seed=seed+fold['fold'])
                    importance_folds.append({'fold':fold['fold'],'permutation_rows':[{**row,'fold':fold['fold']} for row in permutation['rows']],
                                             'auxiliary_comparisons':0})
                    _write_json(folder/f'meta_fold{fold["fold"]}.json',bundle.manifest())
                held['meta_evaluation_status']='held_out';held['prediction_role']='base_oof';held['fold']=fold['fold']
                metrics,trades=run_alpha_backtest(held,local,cost_model=dataset_cost_model(dataset,local))
                block_metrics.append({'fold':fold['fold'],**metrics});parts.append(held)
                audits.append({'fold':fold['fold'],'train_row_ids':train.sample_index.tolist(),
                               'validation_row_ids':held.sample_index.tolist(),'purge':audit})
            combined=pd.concat(parts,ignore_index=True)
            metrics,trades=run_alpha_backtest(combined,local,cost_model=dataset_cost_model(dataset,local))
            stress=cost_stress_report(combined,local)
            importance=summarize_importance(importance_folds)
            acceptance=economic_gate(metrics,block_metrics,stress,importance,settings['economic_gate'])
            combined.to_parquet(folder/'candidates_validation.parquet',index=False);trades.to_parquet(folder/'trades_validation.parquet',index=False)
            row={'trial_id':name,'proposal':proposal,'seed':seed,'status':'completed','metrics':{'validation':metrics,'test':{}},
                 'blocks':block_metrics,'cost_stress':stress,'feature_importance':importance,'economic_gate':acceptance,
                 'test_evaluation':{'status':'disabled','reason':'development-only frozen protocol'},'artifacts':str(folder)}
            _write_json(folder/'purge_audit.json',audits);_write_json(folder/'manifest.json',row)
            trials.append(row);journal[-1].update(status='completed',trades=metrics['trades'],net_pnl=metrics['net_pnl'],acceptance=acceptance['status'])
            _write_progress(output/'attempts.json',journal)
    for path,digest in guard.items():
        if sha(path)!=digest:
            raise AssertionError('Protected earlier evidence changed: '+path)
    eligible=[t for t in trials if t['economic_gate']['passed']]
    # One configuration must pass consistently across all seeds, not win via a lucky seed.
    approved=[p['id'] for p in proposals if all(any(t['proposal']['id']==p['id'] and t['seed']==seed and t['economic_gate']['passed'] for t in trials) for seed in seeds)]
    report={'status':'completed','scope':'development_only','device':str(selected_device),
        'forecast_comparisons':forecast_rows,'opportunity_trials':trials,'approved_configurations':approved,
        'next_action':'review frozen survivors before any separate untouched confirmation' if approved else 'reject current opportunity hypothesis; no final test authorized',
        'budgets':{'declared_configurations':len(proposals),'seeds':seeds,'signal_attempts':len(trials),
                   'forecast_attempts':len(forecast_rows),'actual_fit_counts':fit_counts,
                   'permutation_evaluations':sum(len(f['permutation_rows'])*3 for t in trials for f in t['feature_importance']['folds'])},
        'final_test_evaluations':0,'previous_artifact_hashes_unchanged':True,
        'note':'Seed repetitions and folds are dependent evidence; no automatic promotion. Predictions from rule are not return forecasts.'}
    _write_json(output/'report.json',report)
    write_report(report,output)
    log_study_report(output/'report.json')
    return report


def run_study(config_path, output, *, device='cuda:0', allow_cpu=False):
    try:
        return _run_study(config_path, output, device=device, allow_cpu=allow_cpu)
    except Exception as exc:
        journal_path=Path(output)/'attempts.json'
        if journal_path.is_file():
            journal=json.loads(journal_path.read_text())
            if journal and journal[-1].get('status')=='running':
                journal[-1].update(status='failed',error=str(exc))
                _write_json(journal_path,journal)
        raise


def write_report(report,output):
    lines=['# Development-only forecast and opportunity study','',report['next_action'],'',
           f"Device: {report['device']}. Final-test evaluations: 0. Previous evidence unchanged.",'',
           '## Forecast validity','']
    for kind in sorted({r['model'] for r in report['forecast_comparisons']}):
        rows=[next(iter(r['scores'].values())) for r in report['forecast_comparisons'] if r['model']==kind]
        lines.append(f"- {kind}: median MSE/zero {np.median([r['mse_vs_zero'] for r in rows]):.4f}; median absolute forecast {np.median([r['median_abs_prediction_bps'] for r in rows]):.3f} bps; {len(rows)} seed/fold fits.")
    lines+=['','## Opportunity configurations','']
    for proposal in sorted({t['proposal']['id'] for t in report['opportunity_trials']}):
        rows=[t for t in report['opportunity_trials'] if t['proposal']['id']==proposal]
        lines.append(f"- {proposal}: seed net PnL {[round(t['metrics']['validation']['net_pnl'],3) for t in rows]}; trades {[int(t['metrics']['validation']['trades']) for t in rows]}; gates {[t['economic_gate']['status'] for t in rows]}.")
    lines+=['','## Budget',json.dumps(report['budgets'],indent=2),'',
            'All results are development evidence. No claim of independent seed/fold samples or live profitability.',
            'Use per-trial manifests for block economics, permutation/noise diagnostics and cost stress.']
    (output/'report.md').write_text('\n'.join(lines)+'\n')


def log_study_report(report_path):
    """Idempotent logging from saved evidence; never fits or evaluates a model."""
    report_path=Path(report_path);output=report_path.parent
    report=json.loads(report_path.read_text());protocol=json.loads((output/'protocol.json').read_text())
    config=protocol['configuration']
    prefix='development/'+config['experiment']['id']+'/'+output.name
    entries=[]
    for row in report['forecast_comparisons']:
        name=f"{row['model']}_seed{row['seed']}_fold{row['fold']}"
        entries.append((name,row,Path(row['path'])))
    for row in report['opportunity_trials']:
        entries.append((row['trial_id'],{'metrics':row['metrics']['validation'],'acceptance':row['economic_gate'],
                                        'blocks':row['blocks']},Path(row['artifacts'])))
    entries.append(('summary',{'budgets':report['budgets'],'final_test_evaluations':0,
                               'approved_configurations':len(report['approved_configurations'])},output))
    for name,metrics,folder in entries:
        receipt=folder/'trackio_receipt.json'
        if receipt.is_file():
            saved=json.loads(receipt.read_text())
            if saved.get('delivery')=='logged' and saved.get('verification')=='local_readback':
                _verify_local_run(saved['project'],saved['run'],saved['status'])
                continue
        log_trackio_run(config,name=prefix+'/'+name,group='development_study',
            run_config={'protocol_sha256':sha(output/'protocol.json'),'scope':'development_only'},
            metrics=metrics,artifacts={'folder':str(folder),'report':str(report_path)},
            status='completed',receipt_path=receipt)
    _write_json(output/'tracking_complete.json',{'status':'completed','records':len(entries),
                'fits_added':0,'final_test_evaluations_added':0})


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--prepared-config',type=Path)
    parser.add_argument('--log-report',type=Path)
    parser.add_argument('--output',type=Path)
    parser.add_argument('--device',default='cuda:0');parser.add_argument('--allow-cpu',action='store_true')
    args=parser.parse_args()
    if args.log_report:
        log_study_report(args.log_report)
        return
    if args.prepared_config is None or args.output is None:
        parser.error('Provide --prepared-config and --output, or --log-report')
    report=run_study(args.prepared_config,args.output,device=args.device,allow_cpu=args.allow_cpu)
    print(json.dumps({'report':str(args.output/'report.json'),'approved':report['approved_configurations'],'next_action':report['next_action']}))


if __name__=='__main__':
    main()
