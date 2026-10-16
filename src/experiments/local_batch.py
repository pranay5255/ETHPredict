"""Freeze a bounded local historical study, then run it once on the chosen device."""
from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import yaml

from src.data.lighter_client import LighterDataClient
from src.experiments.staged_trial import run_staged_trial


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def prepare_batch(template, source_data, output_root, *, funding_fetcher=None):
    template, source_data, output_root = Path(template), Path(source_data), Path(output_root).resolve()
    if output_root.exists():
        raise FileExistsError(f'Frozen study directory already exists: {output_root}; use its saved config rather than replacing it')
    config = yaml.safe_load(template.read_text())
    recipe = config.pop('local_batch')
    source = source_data / recipe['source_candles']
    raw = pd.read_csv(source, header=None).tail(int(recipe.get('rows', 8192))).copy()
    if len(raw) < int(recipe.get('rows', 8192)):
        raise ValueError('Insufficient source candles for the predeclared row budget')
    if not raw[0].is_unique or (raw[0].diff().dropna() != 300000).any():
        raise ValueError('Frozen batch requires complete unique five-minute source candles')
    warmup = int(recipe.get('warmup_bars', 2016))
    if len(raw) <= warmup + 100:
        raise ValueError('Frozen batch has insufficient rows after feature warmup')
    output_root.mkdir(parents=True)
    data = output_root / 'data'
    (data / 'raw').mkdir(parents=True)
    candles = data / 'raw' / 'ETHUSDT-5m-lighter-frozen.csv'
    raw.to_csv(candles, header=False, index=False)
    sources = [{'original_path': str(source.resolve()), 'original_sha256': sha(source),
                'frozen_path': str(candles), 'sha256': sha(candles), 'rows': len(raw)}]
    funding_repair = []
    if recipe.get('source_funding'):
        source_funding = source_data / recipe['source_funding']
        funding = pd.read_csv(source_funding)
        funding['timestamp'] = pd.to_datetime(funding.timestamp, utc=True)
        start = pd.to_datetime(raw.iloc[0, 0], unit='ms', utc=True).floor('h') - pd.Timedelta('1h')
        end = pd.to_datetime(raw.iloc[-1, 0], unit='ms', utc=True).floor('h')
        funding = funding[funding.timestamp.between(start, end)].copy()
        missing = pd.date_range(start, end, freq='h').difference(pd.DatetimeIndex(funding.timestamp))
        if len(missing) and not recipe.get('repair_funding_gaps', False):
            raise ValueError(f'Funding cashflows missing at {missing.tolist()}')
        fetch = funding_fetcher or LighterDataClient(timeout=20).fundings
        for timestamp in missing:
            params = dict(market_id=0, resolution='1h', start_timestamp=int((timestamp-pd.Timedelta('1h')).timestamp()*1000),
                          end_timestamp=int((timestamp+pd.Timedelta('1h')).timestamp()*1000), count_back=4)
            response = fetch(**params).copy()
            response['timestamp'] = pd.to_datetime(response.timestamp, utc=True)
            response_hash = hashlib.sha256(response.to_json(orient='records', date_format='iso').encode()).hexdigest()
            recovered = response[response.timestamp == timestamp]
            if len(recovered) != 1:
                raise ValueError(f'Could not recover the exact missing funding cashflow {timestamp}')
            funding = pd.concat([funding, recovered], ignore_index=True)
            funding_repair.append({'timestamp': timestamp.isoformat(), 'endpoint': '/api/v1/fundings', 'parameters': params,
                                   'retrieved_at': datetime.now(timezone.utc).isoformat(), 'response_sha256': response_hash,
                                   'recovered_row': json.loads(recovered.to_json(orient='records', date_format='iso'))[0]})
        funding = funding.sort_values('timestamp')
        if not funding.timestamp.is_unique or (funding.timestamp.diff().dropna() != pd.Timedelta('1h')).any():
            raise ValueError('Frozen funding cashflows are not complete and unique')
        (data / 'lighter').mkdir()
        frozen_funding = data / 'lighter' / 'funding_payments.csv'
        funding.to_csv(frozen_funding, index=False)
        config['costs']['dynamic_funding']['paths'] = ['lighter/funding_payments.csv']
        sources.append({'original_path': str(source_funding.resolve()), 'original_sha256': sha(source_funding),
                        'frozen_path': str(frozen_funding), 'sha256': sha(frozen_funding), 'rows': len(funding)})
    config['data']['dir'] = str(data)
    config['data']['sample_start'] = pd.to_datetime(raw.iloc[warmup, 0], unit='ms', utc=True).isoformat()
    config['pipeline']['artifact_root'] = str(output_root / 'runs')
    config.setdefault('research', {}).setdefault('final_test_guard', {})['ledger_path'] = str(output_root / '_final_test_reuse_ledger.json')
    frozen_config = output_root / 'frozen_config.yml'
    frozen_config.write_text(yaml.safe_dump(config, sort_keys=False))
    manifest = {'protocol': 'first_bounded_fixed_notional_v1', 'template': str(template.resolve()), 'template_sha256': sha(template),
                'frozen_config': str(frozen_config), 'frozen_config_sha256': sha(frozen_config), 'sources': sources,
                'funding_repairs': funding_repair, 'source_start': str(pd.to_datetime(raw.iloc[0, 0], unit='ms', utc=True)),
                'source_end': str(pd.to_datetime(raw.iloc[-1, 0], unit='ms', utc=True)), 'warmup_bars': warmup,
                'sample_start': config['data']['sample_start'], 'selection_budget': config['search'],
                'freeze_note': 'Fixed chronological tail chosen before training; no outcome-dependent window or budget changes'}
    (output_root / 'freeze_manifest.json').write_text(json.dumps(manifest, indent=2, sort_keys=True)+'\n')
    return frozen_config


def verify_prepared(config_path):
    config_path = Path(config_path).resolve()
    manifest = json.loads((config_path.parent / 'freeze_manifest.json').read_text())
    if sha(config_path) != manifest['frozen_config_sha256']:
        raise ValueError('Frozen configuration changed after preparation')
    for source in manifest['sources']:
        if sha(source['frozen_path']) != source['sha256']:
            raise ValueError(f"Frozen data changed after preparation: {source['frozen_path']}")
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--template', type=Path, default=Path('configs/first_experiment_batch.yml'))
    parser.add_argument('--source-data', type=Path)
    parser.add_argument('--output-root', type=Path)
    parser.add_argument('--prepared-config', type=Path)
    parser.add_argument('--prepare-only', action='store_true')
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--allow-cpu', action='store_true')
    args = parser.parse_args()
    config = args.prepared_config
    if config is None:
        if args.source_data is None or args.output_root is None:
            parser.error('Provide --prepared-config or both --source-data and --output-root')
        config = prepare_batch(args.template, args.source_data, args.output_root)
    if args.prepare_only:
        print(config)
        return
    verify_prepared(config)
    report = run_staged_trial(config, device=args.device, allow_cpu=args.allow_cpu)
    print(json.dumps({'manifest': str(Path(report['run_dir'])/'manifest.json'), 'dashboard': report.get('dashboard_path'),
                      'selection': report.get('selection'), 'evidence_grade': report.get('evidence_grade')}, indent=2))


if __name__ == '__main__':
    main()
