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
