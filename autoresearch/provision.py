"""Stage a reviewed standalone runtime; installation is a human-only root step.

Staging reads only public dev data, the pinned public model and existing code/
packages. It never opens sealed observations. Installation moves no lake data,
creates no account, widens no home permissions and grants no Docker access.
"""
from __future__ import annotations

import argparse
import grp
import hashlib
import json
import os
from pathlib import Path
import pwd
import re
import shutil
import stat
import subprocess
import sys
import zlib

from .protocol import object_id, parse_tree, protected_entries, unique, verify_source


SEALED = Path('/srv/ethpredict-grader')
STATE = SEALED/'autoresearch'
PUBLIC = Path('/var/lib/ethpredict-autoresearch/public')
GPU_LOCK = Path('/var/lib/ethpredict-autoresearch/gpu.lock')
SOCKET = '/run/ethpredict-grader/control.sock'


def digest(path):
    result = hashlib.sha256()
    with Path(path).open('rb') as source:
        while chunk := source.read(8*1024**2):
            result.update(chunk)
    return result.hexdigest()


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False)+'\n')


def tree_manifest(root):
    root = Path(root).resolve(strict=True)
    entries = {}
    for path in sorted(root.rglob('*')):
        relative = path.relative_to(root).as_posix()
        if relative == 'manifest.json':
            continue
        metadata = path.lstat()
        if path.is_symlink():
            target = os.readlink(path)
            if Path(target).is_absolute() or not path.resolve(strict=True).is_relative_to(root):
                raise ValueError('Provisioning link escapes its bundle')
            entries[relative] = {'kind': 'symlink', 'target': target}
        elif path.is_file():
            if metadata.st_nlink != 1:
                raise ValueError('Provisioning files cannot be hard links')
            entries[relative] = {'kind': 'file', 'bytes': metadata.st_size,
                'sha256': digest(path), 'mode': metadata.st_mode & 0o777}
        elif not path.is_dir():
            raise ValueError('Special file in provisioning bundle')
    return entries


def verify_bundle(root, manifest_sha256):
    root = Path(root).resolve(strict=True)
    if not re.fullmatch('[a-f0-9]{64}', manifest_sha256) or digest(root/'manifest.json') != manifest_sha256:
        raise ValueError('Reviewed bundle manifest hash mismatch')
    value = json.loads((root/'manifest.json').read_text(), object_pairs_hook=unique)
    if (set(value) != {'version', 'source_commit', 'entries'} or type(value['version']) is not int
            or value['version'] != 1 or not isinstance(value['source_commit'], str)
            or not re.fullmatch('[a-f0-9]{40}', value['source_commit'])):
        raise ValueError('Unknown provisioning manifest')
    if tree_manifest(root) != value['entries']:
        raise ValueError('Provisioning files changed after review')
    return value


def frozen_repository(source, destination):
    """Copy tag/tree metadata and grader/src blobs, never historical data blobs.

    The exact published commit/tag/root tree identities are retained. Unneeded
    historical/data objects are absent; only the required paths are traversed.
    """
    source, destination = Path(source), Path(destination)
    def git(*args):
        return subprocess.check_output(['git', '-C', str(source), '-c', 'core.fsmonitor=false',
            '-c', 'core.hooksPath=/dev/null', *args])
    commit = git('rev-parse', 'grader-v1^{commit}').decode().strip()
    tag = git('rev-parse', 'grader-v1').decode().strip()
    subprocess.run(['git', 'init', '-q', str(destination)], check=True)
    def store(kind, oid, body):
        if object_id(kind, body) != oid:
            raise ValueError('Frozen Git object hash mismatch')
        target = destination/'.git/objects'/oid[:2]/oid[2:]
        if not target.exists():
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(zlib.compress(kind.encode()+b' '+str(len(body)).encode()+b'\0'+body))
    body = git('cat-file', 'commit', commit)
    store('commit', commit, body)
    root = body.splitlines()[0].split()[1].decode()
    root_body = git('cat-file', 'tree', root)
    def copy_tree_metadata(oid):
        # Git archive walks directory metadata even with a src-only pathspec.
        # Retain those tree objects, including gitlink identities, while never
        # requesting unrelated file blobs or historical parent commits.
        store('tree', oid, git('cat-file', 'tree', oid))
        for entry in git('ls-tree', '-z', oid).split(b'\0'):
            if not entry:
                continue
            header, _ = entry.split(b'\t', 1)
            _, kind, child = header.split()
            if kind == b'tree':
                copy_tree_metadata(child.decode())
    copy_tree_metadata(root)
    def copy_tree(oid, prefix):
        raw = git('cat-file', 'tree', oid)
        store('tree', oid, raw)
        for name, (mode, child) in parse_tree(raw).items():
            target = destination/prefix/name
            if mode == '40000':
                copy_tree(child, f'{prefix}/{name}')
            else:
                blob = git('cat-file', 'blob', child)
                store('blob', child, blob)
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(blob)
                target.chmod(int(mode, 8) & 0o777)
    for name, (mode, oid) in parse_tree(root_body).items():
        if name in {'grader', 'src'}:
            if mode != '40000':
                raise ValueError('Frozen source roots must be directories')
            copy_tree(oid, name)
        elif name == '.gitattributes':
            blob = git('cat-file', 'blob', oid)
            store('blob', oid, blob)
            (destination/name).write_bytes(blob)
    if tag != commit:
        store('tag', tag, git('cat-file', 'tag', tag))
    (destination/'.git/refs/tags/grader-v1').write_text(tag+'\n')
    (destination/'.git/HEAD').write_text(commit+'\n')
    # Archive only source dependencies as the frozen baseline materializer does.
    subprocess.run(['git', '-C', str(destination), 'archive', 'grader-v1', '--', 'src/'],
        stdout=subprocess.DEVNULL, check=True)
    return commit


def copy_runtime(venv, target):
    venv, target = Path(venv).resolve(strict=True), Path(target)
    base = Path(subprocess.check_output([str(venv/'bin/python'), '-I', '-c',
        'import sys;print(sys.base_prefix)'], text=True).strip())
    allowed = {'_virtualenv.pth', 'distutils-precedence.pth', '_cuda_bindings_redirector.pth', 'a1_coverage.pth'}
    for path in venv.glob('lib/python*/site-packages/*.pth'):
        if path.name not in allowed:
            raise ValueError(f'Review the additional Python startup hook before copying: {path.name}')
        for line in path.read_text().splitlines():
            if line and not line.startswith(('#', 'import ')) and not (path.parent/line).resolve().is_relative_to(venv):
                raise ValueError('Python startup hook points outside the runtime')
    shutil.copytree(base, target/'python', symlinks=False)
    shutil.copytree(venv, target/'venv', symlinks=False, ignore=shutil.ignore_patterns('__pycache__', '*.pyc'))
    executable = (target/'python/bin/python3').resolve(strict=True).name
    for name in ['python', 'python3', 'python3.13']:
        path = target/'venv/bin'/name
        if path.exists():
            path.unlink()
        path.symlink_to('../../python/bin/'+executable)
    # Verify a standalone staging prefix before recording its installed home.
    (target/'venv/pyvenv.cfg').write_text(f'home = {target.resolve()}/python/bin\ninclude-system-site-packages = false\n')
    python = target/'venv/bin/python'
    subprocess.run([str(python), '-I', '-c',
        'import sys,numpy,pandas,torch,timesfm;assert sys.prefix!=sys.base_prefix;print("standalone_runtime_imports: passed",flush=True)'], check=True)


def units(runtime, agent_group):
    service = f'''[Unit]
Description=ETHPredict private autoresearch grader
Requires=ethpredict-autoresearch.socket
After=ethpredict-autoresearch.socket
[Service]
User=ethgrader
Group=ethgrader
ExecStart={runtime}/venv/bin/python -I {runtime}/control/launch.py /etc/ethpredict-autoresearch.json
Environment=GIT_CONFIG_GLOBAL={runtime}/gitconfig
Environment=GIT_CONFIG_NOSYSTEM=1
Environment=PYTHONDONTWRITEBYTECODE=1
Environment=OMP_NUM_THREADS=1
Environment=OPENBLAS_NUM_THREADS=1
Environment=MKL_NUM_THREADS=1
WorkingDirectory={STATE}
UMask=0022
NoNewPrivileges=yes
ProtectHome=yes
ProtectSystem=strict
ReadWritePaths={STATE} {PUBLIC} {GPU_LOCK}
PrivateTmp=yes
KillMode=control-group
TimeoutStopSec=45
StandardOutput=append:{STATE}/service.log
StandardError=append:{STATE}/service.log
Restart=on-failure
RestartSec=2
'''
    sock = f'''[Unit]
Description=ETHPredict bounded dev-result broker
[Socket]
ListenStream={SOCKET}
SocketUser=ethgrader
SocketGroup={agent_group}
SocketMode=0660
DirectoryMode=0755
RemoveOnStop=yes
[Install]
WantedBy=sockets.target
'''
    return service, sock


def stage(args):
    if os.geteuid() == 0:
        raise PermissionError('Stage as the ordinary construction user')
    from scripts.agent_sandbox import load_workspace, docker_command, run
    from grader.checks import verify_frozen_grader
    from grader.models import registry, verified_model_paths
    root = Path(__file__).resolve().parents[1]
    if subprocess.check_output(['git', '-C', str(root), 'status', '--porcelain']).strip():
        raise ValueError('Commit the reviewed P4 source before staging')
    source_commit = subprocess.check_output(['git', '-C', str(root), 'rev-parse', 'HEAD'], text=True).strip()
    metadata = load_workspace(args.workspace)
    if metadata['source_commit'] != source_commit:
        raise ValueError('Prepare a fresh workspace from this reviewed source commit')
    proof_command = [str(Path(metadata['venv'])/'bin/python'), '-c',
        'import json;from autoresearch.protocol import source_proof;print(json.dumps(source_proof("/workspace")))']
    proof = json.loads(run(docker_command(metadata, proof_command), timeout=60).stdout)
    protected = protected_entries(proof)
    verify_source(proof, protected)
    destination = args.bundle.resolve()
    if destination.exists():
        raise ValueError('Use a new staging bundle directory')
    destination.mkdir(parents=True, mode=0o700)
    runtime = destination/'runtime'
    runtime.mkdir()
    installed = SEALED/('runtime-p4-'+source_commit[:12])
    print('stage: frozen code-only Git snapshot', flush=True)
    frozen_repository(root, runtime/'grader')
    assert verify_frozen_grader(runtime/'grader').passed
    control = runtime/'control'
    (control/'autoresearch').mkdir(parents=True)
    for source in (root/'autoresearch').glob('*.py'):
        shutil.copyfile(source, control/'autoresearch'/source.name)
    (control/'launch.py').write_text('import json,sys\nfrom pathlib import Path\n'
        'config=json.loads(Path(sys.argv[1]).read_text())\n'
        'sys.path[:0]=[str(Path(__file__).parent),config["grader_root"]]\n'
        'from autoresearch.service import main\nmain(["--config",sys.argv[1]])\n')
    print('stage: existing Python and installed packages (no installation or upgrade)', flush=True)
    copy_runtime(args.venv, runtime)
    # Replace the relative staging home after the relocation smoke check.
    (runtime/'venv/pyvenv.cfg').write_text(f'home = {installed}/python/bin\ninclude-system-site-packages = false\n')
    (runtime/'gitconfig').write_text(f'[safe]\n\tdirectory = {installed}/grader\n')
    print('stage: verified public dev lake and pinned public checkpoint', flush=True)
    shutil.copytree(metadata['dev_lake'], runtime/'dev-lake', symlinks=False)
    models = verified_model_paths(['timesfm-2.5'], {'timesfm-2.5': args.model_path})
    model_target = runtime/'models/timesfm-2.5'
    model_target.mkdir(parents=True)
    for name in registry()['timesfm-2.5']['files']:
        shutil.copyfile(models['timesfm-2.5']/name, model_target/name)
    config = {'version': 1, 'grader_root': str(installed/'grader'), 'private_root': str(STATE),
        'public_root': str(PUBLIC), 'dev_lake': str(installed/'dev-lake'),
        'private_lake': str(SEALED/'lake/lighter'),
        'model_paths': {'timesfm-2.5': str(installed/'models/timesfm-2.5')},
        'protected_entries': protected, 'agent_uid': metadata['uid'], 'gpu_lock': str(GPU_LOCK)}
    write_json(destination/'config.json', config)
    service, sock = units(installed, grp.getgrgid(metadata['gid']).gr_name)
    (destination/'units').mkdir()
    (destination/'units/ethpredict-autoresearch.service').write_text(service)
    (destination/'units/ethpredict-autoresearch.socket').write_text(sock)
    print('stage: hashing the complete reviewable bundle', flush=True)
    write_json(destination/'manifest.json', {'version': 1, 'source_commit': source_commit,
                                          'entries': tree_manifest(destination)})
    summary = {'bundle': str(destination), 'manifest_sha256': digest(destination/'manifest.json'),
               'source_commit': source_commit, 'runtime': str(installed)}
    print(json.dumps(summary, indent=2), flush=True)


def install(args):
    if os.geteuid() != 0:
        raise PermissionError('Section 9 reserves this installation for the human running sudo')
    bundle = args.bundle.resolve(strict=True)
    manifest = verify_bundle(bundle, args.manifest_sha256)
    identity = pwd.getpwnam('ethgrader')
    if SEALED.stat().st_uid != identity.pw_uid or stat.S_IMODE(SEALED.stat().st_mode) != 0o700:
        raise PermissionError('Preserve the verified P2 private root ownership and 0700 mode')
    for path in [SEALED, STATE, PUBLIC.parent, PUBLIC]:
        if path.is_symlink():
            raise ValueError('Provisioning directories cannot be symbolic links')
    if PUBLIC.parent.exists() and (PUBLIC.parent.stat().st_uid != 0 or PUBLIC.parent.stat().st_mode & 0o022):
        raise ValueError('The public export needs a protected parent directory')
    config = json.loads((bundle/'config.json').read_text(), object_pairs_hook=unique)
    if type(config.get('agent_uid')) is not int or config['agent_uid'] <= 0:
        raise ValueError('Invalid provisioned loop identity')
    runtime = Path(config['grader_root']).parent
    if runtime != SEALED/('runtime-p4-'+manifest['source_commit'][:12]):
        raise ValueError('Unexpected installed runtime destination')
    if (config['private_root'] != str(STATE) or config['public_root'] != str(PUBLIC)
            or config['gpu_lock'] != str(GPU_LOCK) or config['private_lake'] != str(SEALED/'lake/lighter')):
        raise ValueError('Unexpected private/public service destinations')
    if runtime.exists() or runtime.is_symlink():
        raise ValueError('Runtime already exists; review its deployment before replacing anything')
    # All copies finish before activating the service. Existing sealed data,
    # private journals and previous runtime versions are never removed.
    shutil.copytree(bundle/'runtime', runtime, symlinks=True)
    expected_runtime = {name[len('runtime/'):]: value for name, value in manifest['entries'].items()
                        if name.startswith('runtime/')}
    if tree_manifest(runtime) != expected_runtime:
        raise ValueError('Installed runtime copy does not match the reviewed bundle')
    for path in [runtime, *runtime.rglob('*')]:
        if not path.is_symlink():
            os.chown(path, 0, 0)
    for directory in [STATE, PUBLIC]:
        directory.mkdir(parents=True, exist_ok=True)
        os.chown(directory, identity.pw_uid, identity.pw_gid)
        directory.chmod(0o700 if directory == STATE else 0o755)
    GPU_LOCK.parent.mkdir(parents=True, exist_ok=True)
    parent = GPU_LOCK.parent.stat()
    if GPU_LOCK.parent.is_symlink() or parent.st_uid != 0 or parent.st_mode & 0o022:
        raise ValueError('The shared lock needs a protected persistent parent')
    descriptor = os.open(GPU_LOCK, os.O_CREAT | os.O_RDWR | os.O_NOFOLLOW, 0o660)
    try:
        if not stat.S_ISREG(os.fstat(descriptor).st_mode) or os.fstat(descriptor).st_nlink != 1:
            raise ValueError('Unsafe shared GPU lock file')
        os.fchown(descriptor, identity.pw_uid, pwd.getpwuid(config['agent_uid']).pw_gid)
        os.fchmod(descriptor, 0o660)
    finally:
        os.close(descriptor)
    target = Path('/etc/ethpredict-autoresearch.json')
    shutil.copyfile(bundle/'config.json', target)
    os.chown(target, 0, identity.pw_gid)
    target.chmod(0o640)
    for name in ['ethpredict-autoresearch.service', 'ethpredict-autoresearch.socket']:
        shutil.copyfile(bundle/'units'/name, Path('/etc/systemd/system')/name)
    subprocess.run(['systemctl', 'daemon-reload'], check=True)
    subprocess.run(['systemctl', 'enable', '--now', 'ethpredict-autoresearch.socket'], check=True)
    print(json.dumps({'private_runtime_installed': True, 'socket_enabled': True,
        'runtime': str(runtime), 'source_commit': manifest['source_commit']}), flush=True)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='operation', required=True)
    prepare = sub.add_parser('stage')
    prepare.add_argument('--bundle', required=True, type=Path)
    prepare.add_argument('--workspace', required=True, type=Path)
    prepare.add_argument('--venv', required=True, type=Path)
    prepare.add_argument('--model-path', required=True, type=Path)
    apply = sub.add_parser('install')
    apply.add_argument('--bundle', required=True, type=Path)
    apply.add_argument('--manifest-sha256', required=True)
    args = parser.parse_args(argv)
    (stage if args.operation == 'stage' else install)(args)


if __name__ == '__main__':
    main()
