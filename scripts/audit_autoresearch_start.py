"""Read-only, pre-sealing structural audit; never evaluates a trading policy.

Run against the original local checkout before P2 removes private partitions.
Do not use this command after sealing or publish raw private-tier observations.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd


def sha256(path: Path) -> str:
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def audit(repo: Path) -> dict:
    lake = repo / 'data/lake/lighter'
    recovery = json.loads((repo / 'context/funding_gap_recovery_20261027.json').read_text())
    protected = recovery['protected_file_hashes']
    for name, expected in protected.items():
        if sha256(repo / name) != expected:
            raise ValueError(f'Protected historical file changed: {name}')
    datasets = []
    total_bytes = 0
    for market in ['ETH', 'BTC', 'SOL', 'HYPE', 'NEAR']:
        directory = lake / market
        manifest = json.loads((directory / '_manifest.json').read_text())
        assert manifest['symbol'] == market
        for dataset in ['candles_1m', 'candles_5m', 'candles_1h', 'fundings_1h', 'mark_price_5m']:
            contract = manifest['datasets'][dataset]
            partitions = contract['partitions']
            actual_files = {p.name for p in (directory / dataset).glob('*.parquet')}
            assert actual_files == set(partitions), (market, dataset, 'partition set differs')
            for name, expected in partitions.items():
                path = directory / dataset / name
                assert sha256(path) == expected, path
                total_bytes += path.stat().st_size
            # Deliberately exercise the directory-level reader required by #54.
            frame = pd.read_parquet(directory / dataset).sort_values('timestamp_ms')
            stamps = frame.timestamp_ms.to_numpy()
            assert len(frame) == contract['rows'], (market, dataset, 'row count')
            assert frame.timestamp_ms.is_unique, (market, dataset, 'duplicate timestamp')
            assert frame.symbol.eq(market).all() and frame.market_id.eq(manifest['market_id']).all()
            interval = contract['expected_interval_ms']
            assert np.isfinite(stamps).all() and (np.diff(stamps) == interval).all(), (market, dataset, 'gap')
            assert pd.Timestamp(stamps[0], unit='ms') == pd.Timestamp(contract['first'])
            assert pd.Timestamp(stamps[-1], unit='ms') == pd.Timestamp(contract['last'])
            if dataset == 'fundings_1h':
                assert np.isfinite(frame.rate).all()
                assert frame.direction.isin(['long', 'short']).all()
                assert manifest['funding_recovery']['recovered_rows'] == 31
                assert not manifest['funding_recovery']['unresolved_timestamp_ms']
            datasets.append({'market': market, 'market_id': manifest['market_id'], 'dataset': dataset,
                             'rows': len(frame), 'partitions_verified': len(partitions),
                             'duplicates': 0, 'internal_missing_intervals': 0})
    eth = next(row for row in datasets if row['market'] == 'ETH' and row['dataset'] == 'candles_1m')
    assert eth['rows'] == 719697
    import torch
    return {'audited_at_utc': datetime.now(timezone.utc).isoformat(),
            'scope': 'Pre-sealing structural checks only; no strategy, test or holdout evaluation',
            'dataset_directories_loaded': len(datasets),
            'partition_hashes_verified': sum(row['partitions_verified'] for row in datasets),
            'parquet_bytes': total_bytes, 'protected_historical_files_verified': len(protected),
            'recovered_payments': 155, 'remaining_internal_funding_gaps': 0,
            'versions': {name: importlib.metadata.version(name) for name in ['torch', 'timesfm', 'pandas']},
            'gpu': torch.cuda.get_device_name(0),
            'gpu_total_bytes': torch.cuda.get_device_properties(0).total_memory,
            'datasets': datasets}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repo', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    report = audit(args.repo.resolve())
    args.output.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps({key: value for key, value in report.items() if key != 'datasets'}, indent=2))


if __name__ == '__main__':
    main()
