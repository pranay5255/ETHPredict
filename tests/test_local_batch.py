import json
from pathlib import Path

import pandas as pd
import pytest
import yaml

from src.experiments.local_batch import prepare_batch, verify_prepared
from tests.test_staged_trial import _v2_config, _write_5m_ohlcv


def test_frozen_batch_repairs_only_missing_cashflows_and_detects_mutation(tmp_path):
    source = tmp_path / 'source'
    candles = source / 'raw' / 'ETHUSDT-5m-lighter-fixture.csv'
    _write_5m_ohlcv(candles, rows=160)
    raw = pd.read_csv(candles, header=None)
    start = pd.to_datetime(raw.iloc[0,0], unit='ms', utc=True).floor('h') - pd.Timedelta('1h')
    end = pd.to_datetime(raw.iloc[-1,0], unit='ms', utc=True).floor('h')
    funding = pd.DataFrame({'timestamp': pd.date_range(start, end, freq='h'), 'rate': .001, 'direction': 'long'})
    missing = funding.iloc[[5]].copy()
    (source/'lighter').mkdir()
    funding.drop(index=5).to_csv(source/'lighter'/'funding.csv', index=False)
    original = candles.read_bytes()
    config = _v2_config(source, tmp_path/'runs')
    config['local_batch'] = {'source_candles':'raw/ETHUSDT-5m-lighter-fixture.csv', 'source_funding':'lighter/funding.csv',
                            'rows':160,'warmup_bars':20,'repair_funding_gaps':True}
    config['costs']['dynamic_funding'] = {'paths':[]}
    template = tmp_path/'template.yml'
    template.write_text(yaml.safe_dump(config))
    requests = []
    def fetch(**kwargs):
        requests.append(kwargs)
        return missing.copy()
    saved = prepare_batch(template, source, tmp_path/'frozen', funding_fetcher=fetch)
    manifest = verify_prepared(saved)
    assert len(requests) == 1 and len(manifest['funding_repairs']) == 1
    assert manifest['funding_repairs'][0]['response_sha256']
    assert candles.read_bytes() == original
    with pytest.raises(FileExistsError):
        prepare_batch(template, source, tmp_path/'frozen', funding_fetcher=fetch)
    Path(manifest['sources'][0]['frozen_path']).write_text('mutated')
    with pytest.raises(ValueError, match='changed'):
        verify_prepared(saved)
