import hashlib
from pathlib import Path

import pytest

from grader.data import DataContractError, utc_ms
from grader import models


def test_model_snapshot_checks_bytes_and_removes_external_cache_symlinks(tmp_path, monkeypatch):
    blob=tmp_path/'cache-blob';blob.write_bytes(b'synthetic weights')
    root=tmp_path/'model';root.mkdir();(root/'weights.bin').symlink_to(blob)
    record={'fixture':{'files':{'weights.bin':hashlib.sha256(blob.read_bytes()).hexdigest()}}}
    monkeypatch.setattr(models,'registry',lambda:record)
    copied=models.stage_models(['fixture'],{'fixture':root},tmp_path/'registered')['fixture']
    assert not (copied/'weights.bin').is_symlink()
    assert (copied/'weights.bin').read_bytes()==blob.read_bytes()
    blob.write_bytes(b'changed')
    with pytest.raises(DataContractError,match='SHA-256'):models.verified_model_paths(['fixture'],{'fixture':root})
    with pytest.raises(DataContractError,match='registered'):models.verified_model_paths(['unknown'],{})


def test_registered_checkpoint_has_license_and_conservative_release_labels():
    item=models.registry()['timesfm-2.5']
    assert item['license']=='Apache-2.0' and len(item['revision'])==40
    before=models.contamination(['timesfm-2.5'],start_ms=utc_ms('2025-06-04'),
        end_ms=utc_ms('2026-05-01'),holdout_start_ms=utc_ms('2026-08-01'))[0]
    assert before['period_label']=='possibly in pretraining' and before['holdout_release_eligible']
    assert before['release_safe_after_ms']==utc_ms('2025-09-16')
    after=models.contamination(['timesfm-2.5'],start_ms=utc_ms('2025-09-16'),
        end_ms=utc_ms('2026-05-01'),holdout_start_ms=utc_ms('2025-09-15'))[0]
    assert after['possibly_pretraining_window'] is None and not after['holdout_release_eligible']


def test_model_labels_separate_pre_release_training_from_each_execution_block():
    from types import SimpleNamespace
    import numpy as np
    import pandas as pd
    clock=pd.date_range('2025-09-14',periods=96,freq='h',tz='UTC').as_unit('ms').asi8
    blocks=[SimpleNamespace(block=0,train=np.arange(10),evaluation=np.arange(12,30)),
            SimpleNamespace(block=1,train=np.arange(24),evaluation=np.arange(50,70))]
    report=models.block_contamination(['timesfm-2.5'],blocks,clock,resolution='1h',max_horizon_bars=1,
                                      holdout_start_ms=utc_ms('2026-08-01'))
    assert report[0]['evaluation'][0]['period_label']=='possibly in pretraining'
    assert report[1]['training'][0]['period_label']=='possibly in pretraining'
    assert report[1]['evaluation'][0]['period_label']=='after documented public release'
    assert report[1]['evaluation_window']=={'start_ms':int(clock[50])+3600000,'end_ms':int(clock[70])+3600000}
    assert report[1]['training_window']['end_ms']==int(clock[24])+3600000
    assert models.block_contamination([],blocks,clock,resolution='1h',max_horizon_bars=1,
                                      holdout_start_ms=utc_ms('2026-08-01'))==[]
