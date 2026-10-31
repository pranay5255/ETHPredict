import argparse
import json
import os
from pathlib import Path
import subprocess

import pytest

from autoresearch.provision import (digest, frozen_repository, install, tree_manifest,
                                    units, verify_bundle)
from test_autoresearch_protocol import repository, git, commit


def test_tagged_snapshot_preserves_exact_code_without_historical_data_objects(repository, tmp_path):
    source, _ = repository
    (source/'data').mkdir()
    (source/'data/legacy.csv').write_text('synthetic sealed-period canary; no observations\n')
    commit(source)
    git(source, 'tag', '-a', 'grader-v1', '-m', 'synthetic tag fixture')
    data_blob = git(source, 'rev-parse', 'HEAD:data/legacy.csv')
    destination = tmp_path/'frozen'
    frozen_repository(source, destination)
    assert git(source, 'rev-parse', 'grader-v1^{commit}') == git(destination, 'rev-parse', 'HEAD')
    assert git(source, 'rev-parse', 'grader-v1') == git(destination, 'rev-parse', 'grader-v1')
    assert (destination/'src/model.py').read_bytes() == (source/'src/model.py').read_bytes()
    assert (destination/'grader/score.py').read_bytes() == (source/'grader/score.py').read_bytes()
    assert not (destination/'data').exists()
    result = subprocess.run(['git', '-C', str(destination), 'cat-file', '-e', data_blob], capture_output=True)
    assert result.returncode != 0
    assert git(destination, 'archive', 'grader-v1', '--format=tar', '--', 'src/')


def test_reviewed_bundle_rejects_tampering_extra_files_and_escaping_links(tmp_path):
    bundle = tmp_path/'bundle'
    bundle.mkdir()
    (bundle/'runtime').mkdir()
    (bundle/'runtime/code.py').write_text('# fixture\n')
    manifest = {'version': 1, 'source_commit': 'a'*40, 'entries': tree_manifest(bundle)}
    (bundle/'manifest.json').write_text(json.dumps(manifest))
    digest_before = digest(bundle/'manifest.json')
    assert verify_bundle(bundle, digest_before)['source_commit'] == 'a'*40
    (bundle/'runtime/extra.py').write_text('# not reviewed\n')
    with pytest.raises(ValueError, match='changed'):
        verify_bundle(bundle, digest_before)
    (bundle/'runtime/extra.py').unlink()
    (bundle/'runtime/escape').symlink_to('/etc/passwd')
    with pytest.raises(ValueError, match='escapes'):
        tree_manifest(bundle)


def test_install_refuses_unprivileged_execution_before_touching_any_path(monkeypatch, tmp_path):
    monkeypatch.setattr(os, 'geteuid', lambda: 1000)
    path = tmp_path/'not-created'
    with pytest.raises(PermissionError, match='human'):
        install(argparse.Namespace(bundle=path, manifest_sha256='a'*64))
    assert not path.exists()


def test_systemd_keeps_private_output_out_of_journal_and_kills_worker_group():
    service, socket = units(Path('/srv/ethpredict-grader/runtime-fixture'), 'fixture-agent')
    assert 'User=ethgrader' in service and 'ProtectHome=yes' in service
    assert 'KillMode=control-group' in service and 'ProtectSystem=strict' in service
    assert 'StandardOutput=append:/srv/ethpredict-grader/autoresearch/service.log' in service
    assert 'StandardError=append:/srv/ethpredict-grader/autoresearch/service.log' in service
    assert 'SocketMode=0660' in socket and 'SocketGroup=fixture-agent' in socket
