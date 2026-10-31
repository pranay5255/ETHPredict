import argparse
import os
from pathlib import Path
import shutil

import pytest

from autoresearch.provision import tree_manifest
from scripts.complete_autoresearch_install import complete, reconcile_runtime


@pytest.fixture
def payload(tmp_path):
    source, target = tmp_path/'source', tmp_path/'installed'
    for name, body in {'python/bin/python': b'reviewed-interpreter-fixture',
                       'control/service.py': b'# reviewed controller fixture\n'}.items():
        path = source/name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(body)
    (source/'venv/bin').mkdir(parents=True)
    (source/'venv/bin/python').symlink_to('../../python/bin/python')
    target.mkdir()
    return source, target, tree_manifest(source)


def test_completion_copies_missing_entries_and_verified_partial_prefixes(payload):
    source, target, expected = payload
    (target/'python/bin').mkdir(parents=True)
    (target/'python/bin/python').write_bytes(b'reviewed-')
    result = reconcile_runtime(source, target, expected)
    assert result == {'missing_entries_copied': 2, 'partial_or_metadata_entries_completed': 1}
    assert tree_manifest(target) == expected
    assert reconcile_runtime(source, target, expected) == {
        'missing_entries_copied': 0, 'partial_or_metadata_entries_completed': 0}


def test_differing_complete_file_is_rejected_before_other_entries_are_copied(payload):
    source, target, expected = payload
    (target/'python/bin').mkdir(parents=True)
    path = target/'python/bin/python'
    path.write_bytes(b'changed-interpreter-fixture')
    before = path.read_bytes()
    with pytest.raises(ValueError, match='differs'):
        reconcile_runtime(source, target, expected)
    assert path.read_bytes() == before and not (target/'control').exists()


def test_wrong_truncated_prefix_is_rejected(payload):
    source, target, expected = payload
    (target/'control').mkdir()
    (target/'control/service.py').write_bytes(b'wrong')
    with pytest.raises(ValueError, match='differs'):
        reconcile_runtime(source, target, expected)
    assert not (target/'python').exists()


@pytest.mark.parametrize('extra', ['file', 'directory', 'link'])
def test_unexpected_entries_and_escaping_links_are_rejected(payload, extra):
    source, target, expected = payload
    path = target/'unexpected'
    if extra == 'file':
        path.write_text('unreviewed')
    elif extra == 'directory':
        path.mkdir()
    else:
        path.symlink_to('/etc/passwd')
    with pytest.raises(ValueError, match='Unexpected'):
        reconcile_runtime(source, target, expected)
    assert not (target/'control').exists()


def test_identical_bytes_with_interrupted_copy_metadata_are_completed(payload):
    source, target, expected = payload
    shutil.copytree(source, target, dirs_exist_ok=True, symlinks=True)
    (target/'control/service.py').chmod(0o600)
    result = reconcile_runtime(source, target, expected)
    assert result['partial_or_metadata_entries_completed'] == 1
    assert tree_manifest(target) == expected


def test_reviewed_empty_git_directories_are_allowed_and_restored(payload):
    source, target, expected = payload
    (source/'.git/objects/pack').mkdir(parents=True)
    (target/'.git/objects/pack').mkdir(parents=True)
    reconcile_runtime(source, target, expected)
    assert (target/'.git/objects/pack').is_dir()


def test_hard_link_cannot_be_completed_or_overwritten(payload, tmp_path):
    source, target, expected = payload
    (target/'control').mkdir()
    other = tmp_path/'outside'
    other.write_bytes((source/'control/service.py').read_bytes())
    os.link(other, target/'control/service.py')
    with pytest.raises(ValueError, match='hard link'):
        reconcile_runtime(source, target, expected)
    assert other.read_bytes() == (source/'control/service.py').read_bytes()


def test_human_only_guard_precedes_any_private_filesystem_access(tmp_path, monkeypatch):
    monkeypatch.setattr(os, 'geteuid', lambda: 1000)
    with pytest.raises(PermissionError, match='human'):
        complete(argparse.Namespace(bundle=tmp_path/'absent', manifest_sha256='a'*64))
