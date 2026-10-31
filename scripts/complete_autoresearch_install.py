"""Human-only completion of an interrupted, reviewed P4 runtime install.

Existing bytes must match the reviewed bundle, or be an exact truncated prefix
of its expected file. Unexpected files, changed full files and escaping links
are rejected before mutation. No sealed lake or trial directory is traversed.
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path
import pwd
import shutil
import stat
import subprocess
import sys
import tempfile

if __package__ in {None, ''}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from autoresearch.provision import (GPU_LOCK, PUBLIC, SEALED, STATE, digest,
                                   tree_manifest, unique, verify_bundle)
import json


def _prefix(partial, complete):
    with partial.open('rb') as left, complete.open('rb') as right:
        while chunk := left.read(8*1024**2):
            if chunk != right.read(len(chunk)):
                return False
    return True


def reconcile_runtime(source, runtime, expected):
    """Validate every existing entry first, then finish only known payloads."""
    source, runtime = Path(source).resolve(strict=True), Path(runtime)
    if runtime.is_symlink() or not runtime.is_dir():
        raise ValueError('Expected an existing real runtime directory')
    allowed_directories = {path.relative_to(source).as_posix() for path in source.rglob('*')
                           if path.is_dir() and not path.is_symlink()}
    present, repair = set(), set()
    for current, directories, files in os.walk(runtime, followlinks=False):
        for name in [*directories, *files]:
            path = Path(current)/name
            relative = path.relative_to(runtime).as_posix()
            metadata = path.lstat()
            if stat.S_ISDIR(metadata.st_mode):
                if relative not in allowed_directories:
                    raise ValueError(f'Unexpected runtime directory: {relative}')
                continue
            if relative not in expected:
                raise ValueError(f'Unexpected runtime entry: {relative}')
            record = expected[relative]
            present.add(relative)
            if stat.S_ISLNK(metadata.st_mode):
                if (record['kind'] != 'symlink' or os.readlink(path) != record['target']
                        or not path.resolve(strict=False).is_relative_to(runtime.resolve())):
                    raise ValueError(f'Unexpected runtime link: {relative}')
            elif stat.S_ISREG(metadata.st_mode) and metadata.st_nlink == 1 and record['kind'] == 'file':
                if metadata.st_size == record['bytes'] and digest(path) == record['sha256']:
                    if stat.S_IMODE(metadata.st_mode) != record['mode']:
                        repair.add(relative)
                elif metadata.st_size < record['bytes'] and _prefix(path, source/relative):
                    repair.add(relative)
                else:
                    raise ValueError(f'Existing runtime file differs from reviewed bytes: {relative}')
            else:
                raise ValueError(f'Unexpected runtime file type or hard link: {relative}')
    missing = set(expected)-present
    for relative in sorted(missing | repair, key=lambda name: (expected[name]['kind'] == 'symlink', name)):
        target, record = runtime/relative, expected[relative]
        target.parent.mkdir(parents=True, exist_ok=True)
        if record['kind'] == 'symlink':
            target.symlink_to(record['target'])
        else:
            descriptor, temporary = tempfile.mkstemp(prefix='.resume-', dir=target.parent)
            os.close(descriptor)
            try:
                shutil.copy2(source/relative, temporary)
                if digest(temporary) != record['sha256']:
                    raise ValueError('Resumed copy hash mismatch')
                os.replace(temporary, target)
            finally:
                if os.path.exists(temporary):
                    os.unlink(temporary)
    # Git/Python payloads legitimately contain empty directories. Restore their
    # reviewed directory metadata too; file manifests intentionally omit dirs.
    for relative in sorted(allowed_directories, key=lambda name: len(Path(name).parts)):
        target = runtime/relative
        target.mkdir(parents=True, exist_ok=True)
        shutil.copystat(source/relative, target)
    if tree_manifest(runtime) != expected:
        raise ValueError('Completed runtime does not match the reviewed bundle')
    return {'missing_entries_copied': len(missing), 'partial_or_metadata_entries_completed': len(repair)}


def complete(args):
    if os.geteuid() != 0:
        raise PermissionError('Section 9 reserves this completion step for the human running sudo')
    bundle = args.bundle.resolve(strict=True)
    manifest = verify_bundle(bundle, args.manifest_sha256)
    config = json.loads((bundle/'config.json').read_text(), object_pairs_hook=unique)
    identity = pwd.getpwnam('ethgrader')
    runtime = Path(config['grader_root']).parent
    if (runtime != SEALED/('runtime-p4-'+manifest['source_commit'][:12])
            or SEALED.is_symlink() or SEALED.stat().st_uid != identity.pw_uid
            or stat.S_IMODE(SEALED.stat().st_mode) != 0o700):
        raise ValueError('Private root or runtime identity differs from the reviewed deployment')
    if (config['private_root'] != str(STATE) or config['public_root'] != str(PUBLIC)
            or config['gpu_lock'] != str(GPU_LOCK) or type(config['agent_uid']) is not int
            or config['agent_uid'] <= 0):
        raise ValueError('Unexpected service configuration destinations')
    for name in ['ethpredict-autoresearch.service', 'ethpredict-autoresearch.socket']:
        active = subprocess.run(['systemctl', 'is-active', '--quiet', name]).returncode == 0
        if active:
            raise ValueError('A deployment is already active; inspect it before completing this interrupted install')
    target_config = Path('/etc/ethpredict-autoresearch.json')
    if target_config.is_symlink() or (target_config.exists() and target_config.read_bytes() != (bundle/'config.json').read_bytes()):
        raise ValueError('Existing service configuration requires separate review')
    for name in ['ethpredict-autoresearch.service', 'ethpredict-autoresearch.socket']:
        target = Path('/etc/systemd/system')/name
        if target.is_symlink() or (target.exists() and target.read_bytes() != (bundle/'units'/name).read_bytes()):
            raise ValueError('An existing service unit requires separate review')
    for path in [STATE, PUBLIC.parent, PUBLIC]:
        if path.is_symlink():
            raise ValueError('Service directories cannot be symbolic links')
    if PUBLIC.parent.exists() and (PUBLIC.parent.stat().st_uid != 0 or PUBLIC.parent.stat().st_mode & 0o022):
        raise ValueError('Public service parent permissions changed')
    expected = {name[len('runtime/'):]: value for name, value in manifest['entries'].items()
                if name.startswith('runtime/')}
    summary = reconcile_runtime(bundle/'runtime', runtime, expected)
    for path in [runtime, *runtime.rglob('*')]:
        if not path.is_symlink():
            os.chown(path, 0, 0)
    for directory in [STATE, PUBLIC]:
        directory.mkdir(parents=True, exist_ok=True)
        os.chown(directory, identity.pw_uid, identity.pw_gid)
        directory.chmod(0o700 if directory == STATE else 0o755)
    descriptor = os.open(GPU_LOCK, os.O_CREAT | os.O_RDWR | os.O_NOFOLLOW, 0o660)
    try:
        metadata = os.fstat(descriptor)
        if not stat.S_ISREG(metadata.st_mode) or metadata.st_nlink != 1:
            raise ValueError('Unsafe shared GPU lock')
        os.fchown(descriptor, identity.pw_uid, pwd.getpwuid(config['agent_uid']).pw_gid)
        os.fchmod(descriptor, 0o660)
    finally:
        os.close(descriptor)
    shutil.copyfile(bundle/'config.json', target_config)
    os.chown(target_config, 0, identity.pw_gid)
    target_config.chmod(0o640)
    for name in ['ethpredict-autoresearch.service', 'ethpredict-autoresearch.socket']:
        target = Path('/etc/systemd/system')/name
        if target.is_symlink():
            raise ValueError('An existing unit symlink requires separate review')
        shutil.copyfile(bundle/'units'/name, target)
    subprocess.run(['systemctl', 'daemon-reload'], check=True)
    subprocess.run(['systemctl', 'enable', '--now', 'ethpredict-autoresearch.socket'], check=True)
    print(json.dumps({'private_runtime_installed': True, 'socket_enabled': True, **summary}), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--bundle', required=True, type=Path)
    parser.add_argument('--manifest-sha256', required=True)
    complete(parser.parse_args())


if __name__ == '__main__':
    main()
