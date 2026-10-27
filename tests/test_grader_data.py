import ast
import hashlib
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from grader.data import (CANDLE_COLUMNS, FUNDING_COLUMNS, DataContractError,
                         LakeLoader, availability_join, utc_ms)


def write_lake(root, market='ETH', market_id=0, dataset='candles_5m', frame=None):
    resolution = dataset.rsplit('_', 1)[1]
    interval = {'1m': 60_000, '5m': 300_000, '1h': 3_600_000}[resolution]
    if frame is None:
        clock = pd.date_range('2025-08-25T15:25Z', periods=6, freq='5min') if dataset != 'fundings_1h' else pd.date_range('2025-08-25', periods=6, freq='h', tz='UTC')
        frame = pd.DataFrame({'timestamp_ms': clock.as_unit('ms').asi8,
                              'timestamp': clock.tz_localize(None), 'market_id': market_id, 'symbol': market})
        if dataset == 'fundings_1h':
            frame = frame.assign(value=.01, rate=.0008, direction='long')[FUNDING_COLUMNS]
        else:
            frame = frame.assign(open=100., high=101., low=99., close=100.5,
                                 volume=2., quote_asset_volume=200., last_trade_id=1,
                                 sample_count=0, open_raw=0., high_raw=0., low_raw=0., close_raw=0.)[CANDLE_COLUMNS]
    directory = root / market / dataset
    directory.mkdir(parents=True, exist_ok=True)
    partitions = {}
    # Include empty early mark-price months exactly as the real lake does.
    for month in ['2025-06', '2025-07', '2025-08']:
        part = frame[pd.to_datetime(frame.timestamp_ms, unit='ms').dt.strftime('%Y-%m') == month]
        path = directory / f'{month}.parquet'
        part.to_parquet(path, index=False)
        partitions[path.name] = hashlib.sha256(path.read_bytes()).hexdigest()
    manifest = {'symbol': market, 'market_id': market_id, 'datasets': {
        dataset: {'expected_interval_ms': interval, 'first': str(pd.Timestamp(int(frame.timestamp_ms.min()), unit='ms')),
                  'last': str(pd.Timestamp(int(frame.timestamp_ms.max()), unit='ms')),
                  'partitions': partitions, 'rows': len(frame)}}}
    (root / market / '_manifest.json').write_text(json.dumps(manifest))
    return frame


@pytest.fixture
def lake(tmp_path):
    frame = write_lake(tmp_path)
    return tmp_path, frame, LakeLoader(tmp_path)


def load_fixture(loader, **kwargs):
    params = dict(market='ETH', dataset='candles_5m', resolution='5m',
                  start='2025-08-25T15:25Z', end='2025-08-25T15:55Z')
    return loader.load(**(params | kwargs))


def mutate(lake, transform):
    root, frame, loader = lake
    path = root / 'ETH/candles_5m/2025-08.parquet'
    transform(frame).to_parquet(path, index=False)
    manifest_path = root / 'ETH/_manifest.json'
    manifest = json.loads(manifest_path.read_text())
    manifest['datasets']['candles_5m']['partitions'][path.name] = hashlib.sha256(path.read_bytes()).hexdigest()
    manifest_path.write_text(json.dumps(manifest))


def test_half_open_utc_window_provenance_and_no_extra_month_read(lake):
    root, frame, loader = lake
    result = load_fixture(loader, start='2025-08-25T20:55:00+05:30', end='2025-08-25T15:35Z')
    assert result.frame.timestamp_ms.tolist() == frame.timestamp_ms.iloc[:2].tolist()
    assert result.frame.available_at_ms.tolist() == (frame.timestamp_ms.iloc[:2] + 300_000).tolist()
    report = result.provenance
    assert report['rows'] == 2 and report['market_id'] == 0 and report['complete']
    assert report['gaps'] == [] and len(report['manifest_sha256']) == 64
    assert report['first_ms'] == int(frame.timestamp_ms.iloc[0])
    assert report['last_ms'] == int(frame.timestamp_ms.iloc[1])
    assert report['partitions'] == [{'file': 'ETH/candles_5m/2025-08.parquet',
        'sha256': hashlib.sha256((root / 'ETH/candles_5m/2025-08.parquet').read_bytes()).hexdigest(),
        'rows': 6, 'selected_rows': 2, 'first_ms': int(frame.timestamp_ms.min()), 'last_ms': int(frame.timestamp_ms.max())}]
    assert load_fixture(loader, start='2025-08-31T23:55Z', end='2025-09-01', require_complete=False).provenance['rows'] == 0


def test_hash_is_verified_before_parquet_deserialization(lake, monkeypatch):
    root, _, loader = lake
    path = root / 'ETH/candles_5m/2025-08.parquet'
    path.write_bytes(path.read_bytes() + b'tampered')
    def forbidden(*args, **kwargs):
        pytest.fail('Deserialization happened before hash verification')
    monkeypatch.setattr(pd, 'read_parquet', forbidden)
    with pytest.raises(DataContractError, match='SHA-256'):
        load_fixture(loader)


def test_verified_bytes_are_the_deserialized_bytes(lake, monkeypatch):
    root, frame, loader = lake
    read = pd.read_parquet
    def read_then_tamper(source):
        (root / 'ETH/candles_5m/2025-08.parquet').write_bytes(b'changed after hash')
        return read(source)
    monkeypatch.setattr(pd, 'read_parquet', read_then_tamper)
    assert load_fixture(loader).frame.close.tolist() == frame.close.tolist()


@pytest.mark.parametrize('conflict', [False, True])
def test_duplicates_must_agree_in_every_value(lake, conflict):
    def change(frame):
        duplicate = frame.iloc[[1]].copy()
        if conflict:
            duplicate['volume'] += 1
        return pd.concat([frame, duplicate])
    mutate(lake, change)
    if conflict:
        with pytest.raises(DataContractError, match='Conflicting duplicate'):
            load_fixture(lake[2])
    else:
        assert len(load_fixture(lake[2]).frame) == 6


@pytest.mark.parametrize('column,value,reason', [
    ('close', np.nan, 'Nonfinite'), ('volume', -1., 'Negative'),
    ('high', 1., 'OHLC'), ('symbol', 'BTC', 'identity'),
    ('market_id', 1, 'identity'), ('timestamp_ms', 1, 'grid'),
    ('timestamp', pd.Timestamp('2025-08-25'), 'disagrees')])
def test_invalid_source_values_fail_closed(lake, column, value, reason):
    def change(frame):
        frame = frame.copy()
        frame.loc[0, column] = value
        return frame
    mutate(lake, change)
    with pytest.raises(DataContractError, match=reason):
        load_fixture(lake[2])


def test_unknown_column_is_not_ignored(lake):
    mutate(lake, lambda frame: frame.assign(future_close=9.))
    with pytest.raises(DataContractError, match='columns'):
        load_fixture(lake[2])


@pytest.mark.parametrize('kwargs', [dict(market='../ETH'), dict(dataset='candles_2m'),
    dict(resolution='1h'), dict(start=None), dict(start=123), dict(start='not-a-date'),
    dict(end='2025-08-25T15:25Z'), dict(start='2025-08-25T15:25:00.000001Z')])
def test_invalid_selections_fail_closed(lake, kwargs):
    with pytest.raises(DataContractError):
        load_fixture(lake[2], **kwargs)


def test_missing_initial_internal_and_final_rows_are_not_filled(lake):
    for removed in [0, 2, 5]:
        mutate(lake, lambda frame: frame.drop(index=removed))
        with pytest.raises(DataContractError, match='Missing required'):
            load_fixture(lake[2])
        result = load_fixture(lake[2], require_complete=False)
        assert len(result.frame) == 5
        gap, = result.provenance['gaps']
        assert gap['start_ms'] == int(lake[1].timestamp_ms.iloc[removed])
        assert gap['end_ms'] - gap['start_ms'] == 300_000 and gap['count'] == 1


def test_candle_available_at_close_and_no_stale_forward_fill(lake):
    mutate(lake, lambda frame: frame.drop(index=1))
    source = load_fixture(lake[2], require_complete=False)
    times = pd.DatetimeIndex(['2025-08-25T15:29:59.999Z', '2025-08-25T15:30Z',
                              '2025-08-25T15:34:59.999Z', '2025-08-25T15:35Z',
                              '2025-08-25T15:40Z'])
    joined = availability_join(times, source, ['close'])
    assert joined.available.tolist() == [False, True, True, False, True]
    assert pd.isna(joined.close.iloc[0]) and pd.isna(joined.close.iloc[3])
    assert joined.source_timestamp_ms.iloc[1] == lake[1].timestamp_ms.iloc[0]
    assert joined.source_timestamp_ms.iloc[4] == lake[1].timestamp_ms.iloc[2]


def test_funding_usable_only_at_payment_and_missing_payment_not_carried(tmp_path):
    frame = write_lake(tmp_path, dataset='fundings_1h')
    source = LakeLoader(tmp_path).load('ETH', 'fundings_1h', '1h', '2025-08-25', '2025-08-25T06:00Z')
    assert source.frame.available_at_ms.equals(source.frame.timestamp_ms)
    source.frame.drop(index=1, inplace=True)
    times = pd.DatetimeIndex(['2025-08-24T23:59:59.999Z','2025-08-25T00:00Z',
                              '2025-08-25T01:00Z','2025-08-25T02:00Z'])
    result = availability_join(times, source, ['rate'])
    assert result.available.tolist() == [False, True, False, True]
    assert np.isnan(result.rate.iloc[2])


def test_mark_price_before_actual_first_bar_is_unavailable(tmp_path):
    write_lake(tmp_path, dataset='mark_price_5m')
    source = LakeLoader(tmp_path).load('ETH', 'mark_price_5m', '5m', '2025-06-04', '2025-08-25T15:35Z')
    assert source.provenance['unavailable']
    gap, = source.provenance['gaps']
    assert gap['kind'] == 'before_source_start' and gap['end_ms'] == utc_ms('2025-08-25T15:25Z')
    times = pd.DatetimeIndex(['2025-06-04T00:00Z','2025-08-25T00:00Z','2025-08-25T15:29:59.999Z','2025-08-25T15:30Z'])
    result = availability_join(times, source, ['close'])
    assert result.available.tolist() == [False, False, False, True]
    assert result.close.iloc[:3].isna().all()
    empty = LakeLoader(tmp_path).load('ETH', 'mark_price_5m', '5m', '2025-06-04', '2025-06-05')
    assert not availability_join(times, empty, ['close']).available.any()


def test_multiple_markets_keep_individual_hashes_and_provenance(tmp_path):
    write_lake(tmp_path)
    write_lake(tmp_path, 'BTC', 1)
    result = LakeLoader(tmp_path).load_markets(['ETH','BTC'], 'candles_5m','5m','2025-08-25T15:25Z','2025-08-25T15:55Z')
    assert len(result.frame) == 12 and set(result.frame.market) == {'ETH', 'BTC'}
    assert result.provenance['BTC']['market_id'] == 1
    assert result.provenance['BTC']['partitions'][0]['sha256'] != result.provenance['ETH']['partitions'][0]['sha256']
    with pytest.raises(DataContractError, match='unique'):
        LakeLoader(tmp_path).load_markets(['ETH','ETH'], 'candles_5m','5m','2025-08-25','2025-08-26')


@pytest.mark.parametrize('kind', ['missing_file', 'missing_hash', 'invalid_hash', 'traversal', 'duplicate_key', 'resolution'])
def test_bad_manifest_and_missing_files_fail_closed(lake, kind):
    root, _, loader = lake
    path = root / 'ETH/_manifest.json'
    data = json.loads(path.read_text())
    part = data['datasets']['candles_5m']['partitions']
    if kind == 'missing_file': (root / 'ETH/candles_5m/2025-08.parquet').unlink()
    if kind == 'missing_hash': del part['2025-08.parquet']
    if kind == 'invalid_hash': part['2025-08.parquet'] = 'invalid'
    if kind == 'traversal': part['../../escape.parquet'] = '0'*64
    if kind == 'resolution': data['datasets']['candles_5m']['expected_interval_ms'] = 1
    path.write_text(json.dumps(data))
    if kind == 'duplicate_key': path.write_text(path.read_text().replace('"symbol": "ETH"', '"symbol": "ETH", "symbol": "BTC"'))
    with pytest.raises(DataContractError):
        load_fixture(loader)


def test_symlink_cannot_escape_root(lake, tmp_path):
    root, _, loader = lake
    path = root / 'ETH/candles_5m/2025-08.parquet'
    outside = root.parent / f'{root.name}-outside.parquet'
    outside.write_bytes(path.read_bytes())
    path.unlink()
    path.symlink_to(outside)
    with pytest.raises(DataContractError, match='escapes'):
        load_fixture(loader)


@pytest.mark.parametrize('columns', [['unknown'], ['close','close'], []])
def test_join_requires_known_columns(lake, columns):
    with pytest.raises(DataContractError):
        availability_join(pd.date_range('2025-08-25', periods=1, tz='UTC'), load_fixture(lake[2]), columns)


def test_grader_has_no_candidate_imports():
    for path in (Path(__file__).parents[1] / 'grader').rglob('*.py'):
        for node in ast.walk(ast.parse(path.read_text())):
            imports = [x.name for x in node.names] if isinstance(node, ast.Import) else [node.module or ''] if isinstance(node, ast.ImportFrom) else []
            assert all(name.split('.')[0] not in {'src','strategy'} for name in imports), path


def test_real_lake_csv_parity_and_five_market_provenance():
    root = os.environ.get('ETHPREDICT_SOURCE_REPO')
    if not root:
        pytest.skip('Set ETHPREDICT_SOURCE_REPO for the original pre-sealing lake/archive acceptance check')
    from scripts.verify_lake_csv_parity import verify
    report = verify(Path(root))
    assert report['rows'] == 26496 and report['different_cells'] == 0
    loaded = LakeLoader(Path(root)/'data/lake/lighter').load_markets(
        ['ETH','BTC','SOL','HYPE','NEAR'], 'candles_5m', '5m', '2026-03-28', '2026-03-29')
    assert all(item['rows'] == 288 and item['complete'] for item in loaded.provenance.values())
