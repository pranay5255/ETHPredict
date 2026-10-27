import hashlib
import json
import os
from pathlib import Path
import stat
from types import SimpleNamespace

import pandas as pd
import pytest

from scripts.seal_grader_tiers import (DATASETS, MARKETS, apply_plan, encoded,
                                      make_plan, verify_private)
from scripts.verify_sealed_tiers import verify_denial


@pytest.fixture
def seal_fixture(tmp_path):
    source, private = tmp_path/'public', tmp_path/'private'
    for market in MARKETS:
        manifest={'symbol':market,'market_id':MARKETS.index(market),'source':'fixture','datasets':{}}
        for dataset in DATASETS:
            interval=60_000 if dataset.endswith('1m') else 300_000 if dataset.endswith('5m') else 3_600_000
            directory=source/market/dataset;directory.mkdir(parents=True)
            hashes={}
            for month in ['2026-04','2026-05','2026-08']:
                frame=pd.DataFrame({'timestamp_ms':[int(pd.Timestamp(month+'-01',tz='UTC').value//1_000_000)]})
                path=directory/(month+'.parquet');frame.to_parquet(path,index=False)
                hashes[path.name]=hashlib.sha256(path.read_bytes()).hexdigest()
            manifest['datasets'][dataset]={'partitions':hashes,'expected_interval_ms':interval}
        (source/market/'_manifest.json').write_text(json.dumps(manifest))
    recovery=source/'_funding_recovery/run/ETH/before';recovery.mkdir(parents=True)
    (recovery/'backup.parquet').write_bytes(b'old private source bytes')
    (recovery.parent/'receipt.json').write_text('{"private_funding": 123}')
    return source,private


def test_plan_does_not_mutate_source_and_enrolls_recovery_copies(seal_fixture):
    source,private=seal_fixture
    before={str(p.relative_to(source)):p.read_bytes() for p in source.rglob('*') if p.is_file()}
    plan=make_plan(source,private)
    assert not private.exists()
    assert before=={str(p.relative_to(source)):p.read_bytes() for p in source.rglob('*') if p.is_file()}
    assert len(plan['moves'])==52
    assert sum(m['destination'].startswith('recovery/') for m in plan['moves'])==2
    for item in plan['manifests']:
        for data in item['public']['datasets'].values():
            assert set(data['partitions'])=={'2026-04.parquet'}
        for data in item['private']['datasets'].values():
            assert set(data['partitions'])=={'2026-05.parquet','2026-08.parquet'}


def test_apply_moves_bytes_rewrites_views_and_is_idempotent(seal_fixture):
    source,private=seal_fixture;plan=make_plan(source,private)
    original={item['market']: (source/item['market']/'_manifest.json').read_bytes() for item in plan['manifests']}
    result=apply_plan(plan,os.getuid(),os.getgid())
    assert result['lake_sealed'] and result['private_files_moved']==52
    for move in plan['moves']:
        assert not (source/move['source']).exists()
        assert hashlib.sha256((private/move['destination']).read_bytes()).hexdigest()==move['sha256']
    for item in plan['manifests']:
        assert (private/'original_manifests'/item['market']/'_manifest.json').read_bytes()==original[item['market']]
        assert (source/item['market']/'_manifest.json').read_bytes()==encoded(item['public'])
    assert not (source/'_funding_recovery').exists()
    assert verify_private(plan,os.getuid())['lake_sealed']
    assert apply_plan(plan,os.getuid(),os.getgid())['lake_sealed']
    for path in [private,*private.rglob('*')]:
        assert stat.S_IMODE(path.stat().st_mode)==(0o700 if path.is_dir() else 0o600)


def test_copy_failure_retains_source_and_can_resume(seal_fixture,monkeypatch):
    import scripts.seal_grader_tiers as seal
    source,private=seal_fixture;plan=make_plan(source,private)
    copy=seal.shutil.copyfileobj
    def fail(reader,writer):
        writer.write(reader.read(7))
        raise OSError('simulated interrupted transfer')
    monkeypatch.setattr(seal.shutil,'copyfileobj',fail)
    with pytest.raises(OSError,match='interrupted'):
        apply_plan(plan,os.getuid(),os.getgid())
    assert all((source/move['source']).exists() for move in plan['moves'])
    monkeypatch.setattr(seal.shutil,'copyfileobj',copy)
    stored=json.loads((private/'seal-plan.json').read_text())
    assert apply_plan(stored,os.getuid(),os.getgid())['lake_sealed']


def test_failure_after_some_moves_resumes_from_the_same_plan(seal_fixture,monkeypatch):
    import scripts.seal_grader_tiers as seal
    source,private=seal_fixture;plan=make_plan(source,private)
    copy=seal.shutil.copyfileobj;calls=0
    def fail_second(reader,writer):
        nonlocal calls
        calls+=1
        if calls==2: raise OSError('second transfer failed')
        copy(reader,writer)
    monkeypatch.setattr(seal.shutil,'copyfileobj',fail_second)
    with pytest.raises(OSError): apply_plan(plan,os.getuid(),os.getgid())
    assert not (source/plan['moves'][0]['source']).exists()
    assert (source/plan['moves'][1]['source']).exists()
    monkeypatch.setattr(seal.shutil,'copyfileobj',copy)
    assert apply_plan(plan,os.getuid(),os.getgid())['lake_sealed']


def test_hash_failure_aborts_before_any_source_move(seal_fixture):
    source,private=seal_fixture;plan=make_plan(source,private)
    (source/plan['moves'][-1]['source']).write_bytes(b'tampered')
    with pytest.raises(ValueError,match='Changed source'):
        apply_plan(plan,os.getuid(),os.getgid())
    assert all((source/move['source']).exists() for move in plan['moves'])


def test_unknown_files_and_symlinks_fail_planning(seal_fixture):
    source,private=seal_fixture
    extra=source/'unknown.csv';extra.write_text('unknown')
    with pytest.raises(ValueError,match='Unenrolled'): make_plan(source,private)
    extra.unlink();extra.symlink_to(source/'ETH/_manifest.json')
    with pytest.raises(ValueError,match='symlink'): make_plan(source,private)


def test_missing_file_hash_tamper_and_hardlinks_fail_planning(seal_fixture):
    source,private=seal_fixture
    path=source/'ETH/candles_1m/2026-05.parquet'
    original=path.read_bytes();path.write_bytes(b'bad')
    with pytest.raises(ValueError,match='hash mismatch'): make_plan(source,private)
    path.write_bytes(original)
    alias=source.parent/'alias';os.link(path,alias)
    with pytest.raises(ValueError,match='hard-linked'): make_plan(source,private)
    alias.unlink();path.unlink()
    with pytest.raises(FileNotFoundError): make_plan(source,private)


def test_readable_private_file_cannot_pass_denial_probe(seal_fixture):
    source,private=seal_fixture;plan=make_plan(source,private)
    apply_plan(plan,os.getuid(),os.getgid())
    with pytest.raises(ValueError,match='Agent can open|not root'):
        verify_denial(plan)


def test_wrong_private_modes_fail_verification(seal_fixture):
    source,private=seal_fixture;plan=make_plan(source,private)
    apply_plan(plan,os.getuid(),os.getgid())
    (private/'sealed.json').chmod(0o644)
    with pytest.raises(ValueError,match='mode'): verify_private(plan,os.getuid())


def test_nested_public_and_private_roots_are_rejected(seal_fixture):
    source,_=seal_fixture
    with pytest.raises(ValueError,match='separate'): make_plan(source,source/'hidden')


def test_unexpected_public_copy_fails_verification(seal_fixture):
    source,private=seal_fixture;plan=make_plan(source,private)
    apply_plan(plan,os.getuid(),os.getgid())
    (source/'unexpected.csv').write_text('unknown source')
    with pytest.raises(ValueError,match='Unexpected files'): verify_private(plan,os.getuid())


def test_actual_agent_user_cannot_open_sealed_paths():
    plan_path=os.environ.get('ETHPREDICT_SEAL_PLAN')
    if not plan_path:
        pytest.skip('Human sealing required: set ETHPREDICT_SEAL_PLAN for the real PermissionError acceptance test')
    result=verify_denial(json.loads(Path(plan_path).read_text()))
    assert result['private_path_denial']=='passed' and result['permission_errors']>0


def test_cli_refuses_unprivileged_apply_before_any_mutation(seal_fixture,monkeypatch):
    import scripts.seal_grader_tiers as seal
    source,private=seal_fixture
    monkeypatch.setattr(seal.os,'geteuid',lambda:1000)
    monkeypatch.setattr('sys.argv',['seal','--apply','--source',str(source)])
    with pytest.raises(SystemExit) as error: seal.main()
    assert error.value.code==2 and not private.exists()
    assert len(list(source.glob('*/candles_1m/*.parquet')))==15


def test_cli_requires_reviewed_plan_on_first_apply(seal_fixture,monkeypatch):
    import scripts.seal_grader_tiers as seal
    source,_=seal_fixture
    owner=SimpleNamespace(pw_uid=source.stat().st_uid+1,pw_gid=os.getgid())
    monkeypatch.setattr(seal.os,'geteuid',lambda:0)
    monkeypatch.setattr(seal.pwd,'getpwnam',lambda name:owner)
    exists=Path.exists
    monkeypatch.setattr(Path,'exists',lambda path:False if str(path).startswith('/srv/ethpredict-grader') else exists(path))
    monkeypatch.setattr('sys.argv',['seal','--apply','--source',str(source)])
    with pytest.raises(SystemExit) as error: seal.main()
    assert error.value.code==2
    assert len(list(source.glob('*/candles_1m/*.parquet')))==15
