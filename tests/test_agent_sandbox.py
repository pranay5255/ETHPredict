import hashlib
import io
import json
import os
from pathlib import Path
import sys
import tarfile
from types import SimpleNamespace

import pytest

from scripts import agent_sandbox as sandbox
from scripts.seal_grader_tiers import apply_plan, make_plan
from tests.test_grader_sealing import seal_fixture


def test_empty_rootfs_is_reproducible_and_contains_no_files_or_software():
    data=sandbox.rootfs_bytes()
    assert data==sandbox.rootfs_bytes()
    with tarfile.open(fileobj=io.BytesIO(data)) as archive:
        members=archive.getmembers()
        assert all(item.isdir() or item.issym() for item in members)
        assert all(not item.name.startswith('/') and '..' not in Path(item.name).parts for item in members)
        assert {item.name:item.linkname for item in members if item.issym()}=={
            'bin':'usr/bin','sbin':'usr/sbin','lib':'usr/lib','lib64':'usr/lib64'}


def test_unreviewed_images_and_mutable_tags_are_rejected(monkeypatch):
    with pytest.raises(ValueError,match='immutable'): sandbox.verify_image('redis:7')
    monkeypatch.setattr(sandbox,'run',lambda *args,**kwargs:SimpleNamespace(stdout=json.dumps([
        {'RootFS':{'Layers':['sha256:unreviewed']},'Config':{}}])))
    with pytest.raises(ValueError,match='empty reviewed'): sandbox.verify_image('sha256:'+'0'*64)


def test_empty_layer_image_with_executable_hooks_is_rejected(monkeypatch):
    monkeypatch.setattr(sandbox,'run',lambda *args,**kwargs:SimpleNamespace(stdout=json.dumps([
        {'RootFS':{'Layers':['sha256:'+hashlib.sha256(sandbox.rootfs_bytes()).hexdigest()]},
         'Config':{'Healthcheck':{'Test':['CMD','unreviewed']}}}])))
    with pytest.raises(ValueError,match='hooks'): sandbox.verify_image('sha256:'+'0'*64)


@pytest.fixture
def candidate_repo(tmp_path):
    root=tmp_path/'source-code';root.mkdir()
    files={'src/module.py':'VALUE = 1\n','grader/__init__.py':'# trusted code\n',
           'pyproject.toml':'[project]\nname="fixture"\nversion="0.1.0"\n',
           'data/raw/ETHUSDT-5m-lighter-20260328-20260628.csv':'synthetic outside data\n',
           'context/private.json':'{"synthetic":true}', '.env':'SYNTHETIC_CANARY=1\n',
           'artifacts/positions.parquet':'synthetic outside artifact'}
    for name,value in files.items():
        path=root/name;path.parent.mkdir(parents=True,exist_ok=True);path.write_text(value)
    environment=sandbox.git_environment()
    sandbox.run(['git','init','--initial-branch=main',str(root)],env=environment)
    sandbox.run(['git','-C',str(root),'add','.'],env=environment)
    sandbox.run(['git','-C',str(root),'-c','commit.gpgsign=false','-c','core.hooksPath=/dev/null',
                 'commit','-m','Synthetic historical fixture'],env=environment)
    return root


@pytest.fixture
def prepared(tmp_path,candidate_repo,seal_fixture,monkeypatch):
    source,private=seal_fixture
    plan=make_plan(source,private)
    apply_plan(plan,os.getuid(),os.getgid())
    plan_path=tmp_path/'reviewed-plan.json';plan_path.write_bytes(sandbox.encoded(plan))
    monkeypatch.setattr(sandbox,'verify_image',lambda image:None)
    workspace=tmp_path/'isolated-code'
    metadata=sandbox.prepare_workspace(candidate_repo,workspace,source,plan_path,
                                       Path(sys.prefix),'sha256:'+'0'*64)
    return workspace,metadata


def test_prepared_workspace_has_only_code_and_fresh_git_history(prepared):
    workspace,metadata=prepared
    assert set(metadata['code_hashes'])=={'src/module.py','grader/__init__.py','pyproject.toml'}
    assert (workspace/'src/module.py').read_text()=='VALUE = 1\n'
    assert not (workspace/'data/raw').exists()
    assert not (workspace/'context').exists()
    assert not (workspace/'artifacts').exists()
    assert not (workspace/'.env').exists()
    assert not (workspace/'.git/objects/info/alternates').exists()
    assert sandbox.run(['git','-C',str(workspace),'rev-list','--count','HEAD']).stdout.strip()=='1'
    assert not sandbox.run(['git','-C',str(workspace),'remote']).stdout.strip()
    import subprocess
    for identity in [metadata['source_commit'],metadata['historical_blob']]:
        assert subprocess.run(['git','-C',str(workspace),'cat-file','-e',identity],capture_output=True).returncode!=0


def test_only_src_and_strategy_changes_are_accepted_by_preflight(prepared):
    workspace,metadata=prepared
    (workspace/'src/module.py').write_text('VALUE = 2\n')
    assert sandbox.load_workspace(workspace)['source_commit']==metadata['source_commit']
    (workspace/'grader/__init__.py').write_text('# changed protected code\n')
    with pytest.raises(ValueError,match='Protected sandbox code'): sandbox.load_workspace(workspace)


def test_extra_public_lake_file_stops_sandbox_launch(prepared):
    workspace,metadata=prepared
    (Path(metadata['dev_lake'])/'unexpected.csv').write_text('synthetic unknown source')
    with pytest.raises(ValueError,match='Extra files'): sandbox.load_workspace(workspace)


def test_dev_partition_tampering_stops_sandbox_launch(prepared):
    workspace,metadata=prepared
    path=next(Path(metadata['dev_lake']).glob('ETH/candles_1m/*.parquet'))
    path.write_bytes(path.read_bytes()+b'tampered')
    with pytest.raises(ValueError,match='hash mismatch'): sandbox.load_workspace(workspace)


def test_code_symlinks_are_rejected_before_copy(tmp_path,candidate_repo,seal_fixture,monkeypatch):
    source,private=seal_fixture;plan=make_plan(source,private)
    apply_plan(plan,os.getuid(),os.getgid())
    plan_path=tmp_path/'plan.json';plan_path.write_bytes(sandbox.encoded(plan))
    (candidate_repo/'src/escape.py').symlink_to(candidate_repo/'context/private.json')
    sandbox.run(['git','-C',str(candidate_repo),'add','src/escape.py'])
    sandbox.run(['git','-C',str(candidate_repo),'-c','commit.gpgsign=false','-c','core.hooksPath=/dev/null',
                 'commit','-m','Synthetic symlink fixture'],env=sandbox.git_environment())
    monkeypatch.setattr(sandbox,'verify_image',lambda image:None)
    workspace=tmp_path/'not-created'
    with pytest.raises(ValueError,match='Symlink'):
        sandbox.prepare_workspace(candidate_repo,workspace,source,plan_path,Path(sys.prefix),'sha256:'+'0'*64)
    assert not workspace.exists()


def test_docker_command_cannot_add_host_privileges_or_broad_mounts(prepared):
    workspace,metadata=prepared
    command=sandbox.docker_command(metadata,['/bin/bash','-c','true'])
    assert command[command.index('--network')+1]=='none'
    assert command[command.index('--pull')+1]=='never'
    assert command[command.index('--cap-drop')+1]=='ALL'
    assert '--read-only' in command
    assert 'no-new-privileges=true' in command
    assert '--privileged' not in command and '--pid' not in command and '--use-api-socket' not in command
    mounts=[command[i+1] for i,value in enumerate(command) if value=='--mount']
    writable=[value for value in mounts if not value.endswith(',readonly')]
    assert {value.split('dst=')[1] for value in writable}=={
        '/workspace/src','/workspace/strategy','/workspace/.git'}
    assert all('src=/srv/' not in value and 'src=/home,' not in value and 'src=/,' not in value for value in mounts)


def test_actual_docker_filesystem_network_and_history_isolation(tmp_path):
    path=os.environ.get('ETHPREDICT_AGENT_WORKSPACE')
    if not path:
        pytest.skip('Set ETHPREDICT_AGENT_WORKSPACE for the real installed-Docker acceptance probe')
    canary=tmp_path/'synthetic-outside-archive'
    canary.write_text('outside mount boundary')
    report=sandbox.probe(Path(path),extra_forbidden=[str(canary)])
    assert report['host_paths_denied']>=9
    assert report['historical_objects_absent']==2
    assert report['capabilities']==0 and report['no_new_privileges'] and report['seccomp']
    assert report['network']=='loopback only' and report['protected_code']=='read_only'
