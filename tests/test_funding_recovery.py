import hashlib
import json

import pandas as pd
import pytest

from scripts.lighter_backfill_lake import conform, recover_funding_gaps, summarize, write_atomic
from src.data.lighter_client import LighterApiError, LighterDataClient, normalize_fundings, to_epoch_ms
from src.data.side_data import DataCoverageError
from src.simulator.costs import CostModel

HOUR = 3_600_000


class FundingSource(LighterDataClient):
    def __init__(self, rows, *, exclusive_start=False, duplicate=False):
        super().__init__(request_pause_seconds=0)
        self.rows, self.exclusive_start, self.duplicate, self.calls = rows, exclusive_start, duplicate, []

    def _get(self, path, params=None):
        assert path == '/api/v1/fundings'
        self.calls.append(dict(params))
        rows = [row for row in self.rows
                if (row['timestamp']*1000 > params['start_timestamp'] if self.exclusive_start
                    else row['timestamp']*1000 >= params['start_timestamp'])
                and row['timestamp']*1000 <= params['end_timestamp']]
        # Reproduce the live endpoint's reserved endpoint slot: N slots can
        # deliver N-1 hourly records, dropping the start of an unpadded window.
        rows = rows[-max(0, params['count_back']-1):] if params['count_back'] > 1 else []
        return {'code': 200, 'fundings': rows + rows[:1] if self.duplicate else rows}


def source_rows(start, count):
    return [{'timestamp': (start+i*HOUR)//1000, 'rate': str(i/1e6), 'value': str(i/1000),
             'direction': 'long' if i % 2 else 'short'} for i in range(count)]


@pytest.mark.parametrize('exclusive_start', [False, True])
@pytest.mark.parametrize('budget', [1, 500])
def test_funding_windows_preserve_month_start_and_500_hour_boundary(exclusive_start, budget):
    start = to_epoch_ms('2026-04-01')
    count = 503 if budget == 500 else 3
    rows = source_rows(start-HOUR, count+3)
    client = FundingSource(rows, exclusive_start=exclusive_start, duplicate=True)
    old = client.fundings(0, '1h', start, start+500*HOUR, 500)
    if budget == 500:
        assert start not in set(old.timestamp_ms)  # Original failing behavior.
    client.calls.clear()
    result = client.fetch_fundings_range(0, '1h', start, start+count*HOUR, budget)
    assert result.timestamp_ms.tolist() == list(range(start, start+count*HOUR, HOUR))
    assert result.timestamp_ms.is_unique
    assert max(call['count_back'] for call in client.calls) <= 502
    assert len(client.calls) == (2 if budget == 500 else 3)


def test_funding_missing_responses_stay_missing_and_duplicates_must_agree():
    start = to_epoch_ms('2026-04-01')
    assert FundingSource([]).fetch_fundings_range(0, '1h', start, start+HOUR).empty
    row = source_rows(start, 1)[0]
    with pytest.raises(LighterApiError, match='Conflicting'):
        normalize_fundings({'fundings': [row, {**row, 'rate': '9'}]})
    with pytest.raises(LighterApiError, match='refusing to substitute zero'):
        normalize_fundings({'fundings': [{**row, 'rate': None}]})
    assert normalize_fundings({'fundings': [row]}).iloc[0].rate == 0.


def lake_fixture(tmp_path):
    root = tmp_path / 'lake'
    market = root / 'ETH'
    start = to_epoch_ms('2026-03-31T22:00:00Z')
    rows = source_rows(start, 7)
    missing = [start+HOUR, start+2*HOUR]
    frame = conform(normalize_fundings({'fundings': rows}), 'fundings', 0, 'ETH')
    retained = frame[~frame.timestamp_ms.isin(missing)]
    for month, part in retained.groupby(retained.timestamp.dt.strftime('%Y-%m')):
        write_atomic(part, market / 'fundings_1h' / f'{month}.parquet')
    # Original mainnet manifests predate the explicit network field.
    manifest = {'symbol': 'ETH', 'market_id': 0, 'source': FundingSource([]).base_url,
                'datasets': {'fundings_1h': summarize(market / 'fundings_1h', '1h'), 'untouched': {'identity': 'frozen'}}}
    (market / '_manifest.json').write_text(json.dumps(manifest))
    return root, rows, frame, missing


def test_targeted_closed_month_recovery_preserves_originals_and_is_idempotent(tmp_path):
    root, rows, expected, missing = lake_fixture(tmp_path)
    market = root / 'ETH'
    before = {path.name: path.read_bytes() for path in (market / 'fundings_1h').glob('*.parquet')}
    old_manifest = (market / '_manifest.json').read_bytes()
    client = FundingSource(rows)
    audit = tmp_path / 'audit'
    result = recover_funding_gaps(client, root, 'ETH', audit)
    assert result['recovered_timestamp_ms'] == missing
    assert result['status'] == 'complete' and result['after']['missing_intervals'] == 0
    actual = pd.read_parquet(market / 'fundings_1h').sort_values('timestamp_ms').reset_index(drop=True)
    pd.testing.assert_frame_equal(actual, expected)
    for name, content in before.items():
        assert (audit / 'ETH' / 'before' / name).read_bytes() == content
    assert (audit / 'ETH' / 'before' / '_manifest.json').read_bytes() == old_manifest
    assert result['before_manifest_sha256'] == hashlib.sha256(old_manifest).hexdigest()
    manifest = json.loads((market / '_manifest.json').read_text())
    assert manifest['datasets']['untouched'] == {'identity': 'frozen'}
    assert manifest['datasets']['fundings_1h'] == summarize(market / 'fundings_1h', '1h')
    requests = len(client.calls)
    hashes = (market / '_manifest.json').read_bytes()
    assert recover_funding_gaps(client, root, 'ETH', audit)['status'] == 'already_complete'
    assert len(client.calls) == requests and (market / '_manifest.json').read_bytes() == hashes


def test_unavailable_payments_are_journaled_without_zero_fill(tmp_path):
    root, rows, expected, missing = lake_fixture(tmp_path)
    rows = [row for row in rows if row['timestamp']*1000 not in missing]
    result = recover_funding_gaps(FundingSource(rows), root, 'ETH', tmp_path / 'audit')
    assert result['status'] == 'incomplete' and result['unresolved_timestamp_ms'] == missing
    assert result['before']['partitions'] == result['after']['partitions']
    assert len(result['requests']) == 2 and all('normalized_response' in row for row in result['requests'])


def test_source_failure_does_not_partially_rewrite_a_market(tmp_path):
    root, rows, expected, missing = lake_fixture(tmp_path)
    client = FundingSource(rows)
    original_get = client._get
    def fail_second(path, params=None):
        if client.calls:
            raise LighterApiError('source unavailable')
        return original_get(path, params)
    client._get = fail_second
    market = root / 'ETH'
    before = summarize(market / 'fundings_1h', '1h')
    manifest_bytes = (market / '_manifest.json').read_bytes()
    with pytest.raises(LighterApiError, match='source unavailable'):
        recover_funding_gaps(client, root, 'ETH', tmp_path / 'audit')
    assert summarize(market / 'fundings_1h', '1h') == before
    assert (market / '_manifest.json').read_bytes() == manifest_bytes
    receipt = json.loads((tmp_path / 'audit/ETH/receipt.json').read_text())
    assert receipt['status'] == 'fetch_failed_no_partitions_changed'
    assert receipt['requests'][-1]['error'] == 'source unavailable'


def test_recovery_rejects_changed_partition_hash_before_querying_source(tmp_path):
    root, rows, expected, missing = lake_fixture(tmp_path)
    path = next((root / 'ETH/fundings_1h').glob('*.parquet'))
    frame = pd.read_parquet(path)
    frame.loc[frame.index[0], 'rate'] = 99.
    write_atomic(frame, path)
    client = FundingSource(rows)
    with pytest.raises(ValueError, match='changed since the manifest'):
        recover_funding_gaps(client, root, 'ETH', tmp_path / 'audit')
    assert client.calls == []


@pytest.mark.parametrize('missing', [0, 3, 7])
def test_dynamic_grading_rejects_missing_payment_even_with_relaxed_gap_setting(tmp_path, missing):
    stamps = pd.date_range('2026-04-01', periods=8, freq='h', tz='UTC')
    source = tmp_path / 'funding.csv'
    pd.DataFrame({'timestamp': stamps.delete(missing), 'rate': .01}).to_csv(source, index=False)
    config = {'mode': 'dynamic', 'dynamic_funding': {'enabled': True, 'historical': True,
              'paths': [str(source)], 'rate_column': 'rate', 'rate_unit': 'bps', 'max_gap': '2h'}}
    path = pd.DataFrame({'timestamp': stamps})
    with pytest.raises(DataCoverageError):
        CostModel(config, path)
    pd.DataFrame({'timestamp': stamps, 'rate': .01}).to_csv(source, index=False)
    model = CostModel(config, path)
    quote = model.quote({'sample_index': 0, 'entry_idx': 0, 'horizon_bars': 7, 'side': 1}, exit_index=7)
    assert quote['funding_bps'] == pytest.approx(.07)
