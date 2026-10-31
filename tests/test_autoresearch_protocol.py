import base64
from copy import deepcopy
import socket
import struct
import subprocess

import pytest

from autoresearch.protocol import (MAX_MESSAGE, object_id, parse_tree, protected_entries,
    receive, send, source_proof, verify_source)


def git(repo, *args):
    return subprocess.check_output(['git', '-C', str(repo), *args], text=True).strip()


@pytest.fixture
def repository(tmp_path):
    repo = tmp_path/'candidate'
    repo.mkdir()
    git(repo, 'init', '-q')
    git(repo, 'config', 'user.name', 'fixture')
    git(repo, 'config', 'user.email', 'fixture@example.invalid')
    for name, content in {'src/model.py': 'value=1\n', 'strategy/strategy.py': 'class Strategy: pass\n',
            'strategy/README.md': '# Hypothesis\nFlat control.\n', 'grader/score.py': 'score=0\n',
            'autoresearch/run_experiment.py': '# protected\n', 'pyproject.toml': '# locked\n'}.items():
        path = repo/name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content)
    git(repo, 'add', '.')
    git(repo, 'commit', '-qm', 'initialize')
    initial = source_proof(repo)
    return repo, protected_entries(initial)


def commit(repo, message='change'):
    git(repo, 'add', '.')
    git(repo, 'commit', '-qm', message)


def test_raw_git_proof_roundtrip_requires_no_service_side_git_or_import(repository, monkeypatch):
    repo, protected = repository
    (repo/'src/model.py').write_text('raise RuntimeError("never import me")\n')
    commit(repo)
    proof = source_proof(repo)
    monkeypatch.setattr(subprocess, 'check_output', lambda *a, **kw: pytest.fail('service must not run Git'))
    files = verify_source(proof, protected)
    assert set(files) == {'src/model.py', 'strategy/strategy.py', 'strategy/README.md'}
    assert files['src/model.py'][1].startswith(b'raise RuntimeError')
    assert proof['commit'] == object_id('commit', base64.b64decode(proof['commits'][proof['commit']]))


@pytest.mark.parametrize('path', ['grader/score.py', 'autoresearch/run_experiment.py', 'pyproject.toml', 'new-file'])
def test_forbidden_path_is_rejected(repository, path):
    repo, protected = repository
    (repo/path).write_text('forbidden\n')
    commit(repo)
    with pytest.raises(ValueError, match='forbidden'):
        verify_source(source_proof(repo), protected)


def test_reverting_forbidden_parent_does_not_hide_the_forbidden_commit(repository):
    repo, protected = repository
    original = (repo/'grader/score.py').read_text()
    (repo/'grader/score.py').write_text('score=1\n')
    commit(repo)
    (repo/'grader/score.py').write_text(original)
    commit(repo, 'restore protected bytes')
    with pytest.raises(ValueError, match='forbidden'):
        verify_source(source_proof(repo), protected)


@pytest.mark.parametrize('kind', ['commits', 'trees', 'blobs'])
def test_tampered_object_bytes_fail_hash_verification(repository, kind):
    repo, protected = repository
    proof = source_proof(repo)
    oid = next(iter(proof[kind]))
    proof[kind][oid] = base64.b64encode(b'tampered').decode()
    with pytest.raises(ValueError, match='hash mismatch'):
        verify_source(proof, protected)


def test_symlink_cannot_enter_editable_source(repository):
    repo, _ = repository
    (repo/'src/escape').symlink_to('/srv/ethpredict-grader/seal-plan.json')
    commit(repo)
    with pytest.raises(ValueError, match='regular'):
        source_proof(repo)


def test_unknown_fields_and_unreferenced_objects_are_rejected(repository):
    repo, protected = repository
    proof = source_proof(repo)
    with pytest.raises(ValueError, match='Unknown'):
        verify_source({**proof, 'private_lake': '/srv'}, protected)
    extra = deepcopy(proof)
    body = b'unused'
    extra['blobs'][object_id('blob', body)] = base64.b64encode(body).decode()
    with pytest.raises(ValueError, match='Extraneous'):
        verify_source(extra, protected)


def test_malformed_tree_names_modes_and_truncated_oid_fail_closed():
    for body in [b'120000 link\0'+b'x'*20, b'100644 ../escape\0'+b'x'*20,
                 b'100644 .git\0'+b'x'*20, b'100644 file\0'+b'x'*19]:
        with pytest.raises(ValueError):
            parse_tree(body)


def test_wire_roundtrip_and_duplicate_key_or_oversized_frame_rejection():
    left, right = socket.socketpair()
    with left, right:
        send(left, {'version': 1, 'operation': 'status', 'request_id': 'abc'})
        assert receive(right)['operation'] == 'status'
        body = b'{"operation":"status","operation":"submit"}'
        left.sendall(struct.pack('!I', len(body)) + body)
        with pytest.raises(ValueError, match='Repeated'):
            receive(right)
        left.sendall(struct.pack('!I', MAX_MESSAGE + 1))
        with pytest.raises(ValueError, match='size'):
            receive(right)
