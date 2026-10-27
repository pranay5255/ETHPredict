"""Pre-sealing ETH archive/lake acceptance check; no policy or test evaluation.

Run as: uv run python -m scripts.verify_lake_csv_parity --repo <original checkout>
Both CSV formats use price units and base/quote asset volumes without scaling.
The training export has no header; its open_time equals timestamp_ms and its
inclusive close_time is timestamp_ms + resolution - 1ms. float_precision='round_trip'
recovers the float64 represented by CSV text rather than a lossy parser estimate.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd

from grader.data import DataContractError, LakeLoader


OHLCV = ['open', 'high', 'low', 'close', 'volume', 'quote_asset_volume']
MODEL_COLUMNS = ['open_time', 'open', 'high', 'low', 'close', 'volume', 'close_time',
                 'quote_asset_volume', 'number_of_trades', 'taker_buy_base_asset_volume',
                 'taker_buy_quote_asset_volume', 'ignore']


def verify(repo: Path) -> dict:
    archive = repo / 'data/lighter/candles_ETH_perp_5m_20260328_20260628.csv'
    model_csv = repo / 'data/raw/ETHUSDT-5m-lighter-20260328-20260628.csv'
    source = pd.read_csv(archive, float_precision='round_trip')
    model = pd.read_csv(model_csv, header=None, names=MODEL_COLUMNS, float_precision='round_trip')
    if source.timestamp_ms.duplicated().any() or model.open_time.duplicated().any():
        raise DataContractError('CSV contains duplicate timestamps')
    if not (model.close_time == model.open_time + 300_000 - 1).all():
        raise DataContractError('Model CSV has invalid inclusive close_time')
    source = source.set_index('timestamp_ms').sort_index()
    model = model.set_index('open_time').sort_index()
    pd.testing.assert_index_equal(source.index, model.index, check_names=False)
    pd.testing.assert_frame_equal(source[OHLCV], model[OHLCV], check_names=False, check_exact=True)
    begin, finish = int(source.index.min()), int(source.index.max()) + 300_000
    loaded = LakeLoader(repo / 'data/lake/lighter').load('ETH', 'candles_5m', '5m',
        pd.Timestamp(begin, unit='ms', tz='UTC'), pd.Timestamp(finish, unit='ms', tz='UTC'))
    lake = loaded.frame.set_index('timestamp_ms')
    pd.testing.assert_index_equal(source.index, lake.index)
    pd.testing.assert_frame_equal(source[OHLCV], lake[OHLCV], check_exact=True)
    return {'rows': len(source), 'different_cells': 0,
            'window': {'start_ms': begin, 'end_ms': finish},
            'columns': OHLCV, 'csv_sha256': hashlib.sha256(archive.read_bytes()).hexdigest(),
            'model_csv_sha256': hashlib.sha256(model_csv.read_bytes()).hexdigest(),
            'units': 'OHLC quote/base; volume base asset; quote_asset_volume quote asset; no scaling',
            'timestamp_conversion': 'model open_time = timestamp_ms; inclusive close_time = timestamp_ms + 300000 - 1',
            'csv_parser': 'float_precision=round_trip (no price/volume transformations)',
            'provenance': loaded.provenance}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repo', type=Path, required=True)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    report = verify(args.repo)
    if args.output:
        args.output.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps({key: value for key, value in report.items() if key != 'provenance'}, indent=2))


if __name__ == '__main__':
    main()
