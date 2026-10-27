"""Dev-only tool execution for the future loop agent, using installed Docker.

No base image or package is downloaded. An empty local rootfs contains only
mount-point directories/symlinks; existing OS/Python libraries are mounted
read-only. The candidate gets fresh code-only Git history, writable src/strategy,
and a verified dev-only lake. Host homes, credentials, Docker sockets, historical
Git objects, old CSVs and experiment outputs are never mounted.
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import tarfile

if __package__:
    from .seal_grader_tiers import DATASETS, MARKETS, encoded, read_plan, require_regular
else:
    from seal_grader_tiers import DATASETS, MARKETS, encoded, read_plan, require_regular


CODE_PREFIXES = ('src/', 'grader/', 'strategy/')
CODE_ROOTS = {'.python-version', '.gitignore', 'pyproject.toml', 'uv.lock',
              'autoresearch/program.md', 'autoresearch/run_experiment.py'}
METADATA = '.agent-sandbox.json'


def run(arguments, **kwargs):
    return subprocess.run(arguments, check=True, text=True, stdout=subprocess.PIPE,
                          stderr=subprocess.PIPE, **kwargs)


def rootfs_bytes() -> bytes:
    """A deterministic, empty filesystem; there are no executables or data in it."""
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode='w') as archive:
        for name in ['usr', 'tmp', 'workspace', 'home', 'srv', 'etc', 'run', 'var', 'proc', 'dev']:
            entry = tarfile.TarInfo(name)
            entry.type = tarfile.DIRTYPE
            entry.mode = 0o1777 if name == 'tmp' else 0o755
            entry.mtime = 0
            archive.addfile(entry)
        for name, target in [('bin', 'usr/bin'), ('sbin', 'usr/sbin'),
                             ('lib', 'usr/lib'), ('lib64', 'usr/lib64')]:
            entry = tarfile.TarInfo(name)
            entry.type = tarfile.SYMTYPE
            entry.linkname = target
            entry.mode = 0o755
            entry.mtime = 0
            archive.addfile(entry)
    return buffer.getvalue()


def verify_image(image: str) -> None:
    if not re.fullmatch(r'sha256:[0-9a-f]{64}', image):
        raise ValueError('Use an immutable image ID from build-image')
    info = json.loads(run(['docker', 'image', 'inspect', image]).stdout)[0]
    if info['RootFS']['Layers'] != ['sha256:' + hashlib.sha256(rootfs_bytes()).hexdigest()]:
        raise ValueError('Sandbox image is not the empty reviewed filesystem')
    config = info['Config']
    if config.get('Env') or config.get('Volumes') or config.get('Healthcheck') or config.get('Entrypoint'):
        raise ValueError('Unexpected image environment, volumes or executable hooks')


def build_image() -> str:
    result = subprocess.run(['docker', 'import', '--change', 'USER 1000:1000',
        '--change', 'CMD ["/bin/false"]', '-', 'ethpredict-agent-empty:p2'],
        input=rootfs_bytes(), stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)
    image = result.stdout.decode().strip()
    verify_image(image)
    return image


def verify_dev_lake(lake: Path, plan: dict) -> dict:
    if lake.resolve(strict=True) != Path(plan['source_root']):
        raise ValueError('Dev lake differs from the reviewed sealed source')
    expected, hashes = set(), {}
    for item in plan['manifests']:
        manifest = lake / item['market'] / '_manifest.json'
        require_regular(manifest)
        if manifest.read_bytes() != encoded(item['public']):
            raise ValueError('Only the sealed dev-only manifest view may be mounted')
        expected.add(manifest)
        hashes[str(manifest.relative_to(lake))] = hashlib.sha256(manifest.read_bytes()).hexdigest()
        for dataset in DATASETS:
            for name, digest in item['public']['datasets'][dataset]['partitions'].items():
                path = lake / item['market'] / dataset / name
                require_regular(path)
                with path.open('rb') as stream:
                    if hashlib.file_digest(stream, 'sha256').hexdigest() != digest:
                        raise ValueError(f'Dev partition hash mismatch: {path}')
                expected.add(path)
                hashes[str(path.relative_to(lake))] = digest
    for path in lake.rglob('*'):
        if path.is_symlink():
            raise ValueError('Symlink in dev lake')
    if {path for path in lake.rglob('*') if path.is_file()} != expected:
        raise ValueError('Extra files in dev lake require review before mounting')
    return hashes


def git_environment() -> dict:
    # No caller Git configuration, hooks, credentials or alternate object stores.
    return {'PATH': '/usr/bin:/bin', 'GIT_CONFIG_NOSYSTEM': '1',
            'GIT_CONFIG_GLOBAL': '/dev/null', 'GIT_AUTHOR_NAME': 'ETHPredict research',
            'GIT_AUTHOR_EMAIL': 'research@invalid', 'GIT_COMMITTER_NAME': 'ETHPredict research',
            'GIT_COMMITTER_EMAIL': 'research@invalid'}


def prepare_workspace(repo: Path, workspace: Path, lake: Path, plan_path: Path,
                      venv: Path, image: str) -> dict:
    if os.getuid() == 0:
        raise ValueError('Prepare/run the agent sandbox as an unprivileged user')
    repo, workspace, lake, venv = repo.resolve(strict=True), workspace.resolve(), lake.resolve(strict=True), venv.resolve(strict=True)
    plan = read_plan(plan_path)
    verify_image(image)
    lake_hashes = verify_dev_lake(lake, plan)
    if workspace.exists() or any(workspace.is_relative_to(root) or root.is_relative_to(workspace)
                                 for root in [repo, lake, venv, Path(plan['private_root'])]):
        raise ValueError('Use a new workspace outside source, runtime and data roots')
    environment = git_environment()
    run(['git', '-C', str(repo), 'diff', '--exit-code', 'HEAD', '--', *CODE_PREFIXES, *sorted(CODE_ROOTS)], env=environment)
    names = run(['git', '-C', str(repo), 'ls-files', '-z'], env=environment).stdout.split('\0')
    names = sorted(name for name in names if name and (name in CODE_ROOTS or name.startswith(CODE_PREFIXES)))
    if not names:
        raise ValueError('No permitted tracked code to prepare')
    for name in names:
        path = repo / name
        if not path.resolve(strict=True).is_relative_to(repo) or any(parent.is_symlink() for parent in [path, *path.parents] if parent != repo.parent):
            raise ValueError('Symlink or escaped source code path')
        require_regular(path)
    python = (venv / 'bin/python').resolve(strict=True)
    runtime = python.parent.parent
    workspace.mkdir(mode=0o700, parents=True)
    file_hashes = {}
    for name in names:
        target = workspace / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(repo / name, target)
        file_hashes[name] = hashlib.sha256(target.read_bytes()).hexdigest()
    for name in ['src', 'strategy', 'grader', 'data/lake/lighter']:
        (workspace / name).mkdir(parents=True, exist_ok=True)
    (workspace / '.venv').symlink_to(venv, target_is_directory=True)
    empty_template = workspace / '.empty-git-template'
    empty_template.mkdir()
    run(['git', 'init', '--initial-branch=autoresearch/isolated', f'--template={empty_template}', str(workspace)], env=environment)
    empty_template.rmdir()
    run(['git', '-C', str(workspace), 'add', '--', *names], env=environment)
    run(['git', '-C', str(workspace), '-c', 'commit.gpgsign=false', '-c', 'core.hooksPath=/dev/null',
         'commit', '-m', 'Initialize code-only isolated research workspace'], env=environment)
    original_commit = run(['git', '-C', str(repo), 'rev-parse', 'HEAD'], env=environment).stdout.strip()
    # Object identity only: never deserialize the historical CSV's contents.
    historical_blob = run(['git', '-C', str(repo), 'rev-parse',
        'HEAD:data/raw/ETHUSDT-5m-lighter-20260328-20260628.csv'], env=environment).stdout.strip()
    tools = {name: str(Path(path).resolve(strict=True)) for name in ['uv', 'rg']
             if (path := shutil.which(name)) is not None}
    metadata = {'version': 1, 'workspace': str(workspace), 'source_repo': str(repo),
                'source_commit': original_commit, 'historical_blob': historical_blob,
                'image': image, 'venv': str(venv), 'python_runtime': str(runtime),
                'tools': tools, 'dev_lake': str(lake), 'plan_path': str(plan_path.resolve(strict=True)),
                'code_hashes': file_hashes, 'dev_hashes': lake_hashes,
                'uid': os.getuid(), 'gid': os.getgid()}
    (workspace / METADATA).write_bytes(encoded(metadata))
    return metadata


def load_workspace(workspace: Path) -> dict:
    workspace = workspace.resolve(strict=True)
    require_regular(workspace / METADATA)
    metadata = json.loads((workspace / METADATA).read_text())
    expected = {'version', 'workspace', 'source_repo', 'source_commit', 'historical_blob',
                'image', 'venv', 'python_runtime', 'tools', 'dev_lake', 'plan_path',
                'code_hashes', 'dev_hashes', 'uid', 'gid'}
    if set(metadata) != expected or metadata['version'] != 1 or metadata['workspace'] != str(workspace):
        raise ValueError('Unknown sandbox metadata')
    if metadata['uid'] != os.getuid() or metadata['gid'] != os.getgid() or os.getuid() == 0:
        raise ValueError('Run as the unprivileged workspace owner')
    verify_image(metadata['image'])
    if verify_dev_lake(Path(metadata['dev_lake']), read_plan(Path(metadata['plan_path']))) != metadata['dev_hashes']:
        raise ValueError('Dev snapshot changed')
    for name, digest in metadata['code_hashes'].items():
        if name.startswith('src/') or name.startswith('strategy/'):
            continue  # These are intentionally editable.
        require_regular(workspace / name)
        if hashlib.sha256((workspace / name).read_bytes()).hexdigest() != digest:
            raise ValueError('Protected sandbox code changed')
    return metadata


def docker_command(metadata: dict, command: list[str]) -> list[str]:
    if not command:
        raise ValueError('Provide a command to execute inside the sandbox')
    workspace, venv = Path(metadata['workspace']), Path(metadata['venv'])
    arguments = ['docker', 'run', '--rm', '--pull', 'never', '--network', 'none',
        '--read-only', '--cap-drop', 'ALL', '--security-opt', 'no-new-privileges=true',
        '--user', f"{metadata['uid']}:{metadata['gid']}", '--pids-limit', '512',
        '--memory', '48g', '--memory-swap', '48g',
        '--tmpfs', '/tmp:rw,nosuid,nodev,size=2g,mode=1777', '--workdir', '/workspace']
    bindings = [(Path('/usr'), '/usr', True), (venv, str(venv), True),
                (Path(metadata['python_runtime']), metadata['python_runtime'], True),
                (workspace, '/workspace', True),
                (Path(metadata['dev_lake']), '/workspace/data/lake/lighter', True)]
    bindings += [(workspace / name, f'/workspace/{name}', False) for name in ['src', 'strategy', '.git']]
    bindings += [(Path(path), '/tools/' + name, True) for name, path in metadata['tools'].items()]
    for source, target, readonly in bindings:
        if ',' in str(source) or ',' in target:
            raise ValueError('Unsupported comma in mount path')
        arguments += ['--mount', f'type=bind,src={source},dst={target}' + (',readonly' if readonly else '')]
    environment = {'PATH': f'{venv}/bin:/tools:/usr/bin:/bin',
        'PYTHONDONTWRITEBYTECODE': '1', 'PYTHONNOUSERSITE': '1',
        'XDG_CACHE_HOME': '/tmp/cache', 'UV_CACHE_DIR': '/tmp/uv-cache',
        'UV_NO_SYNC': '1', 'UV_OFFLINE': '1', 'UV_PYTHON_DOWNLOADS': 'never',
        'GIT_CONFIG_NOSYSTEM': '1', 'GIT_CONFIG_GLOBAL': '/dev/null',
        'GIT_AUTHOR_NAME': 'ETHPredict research', 'GIT_AUTHOR_EMAIL': 'research@invalid',
        'GIT_COMMITTER_NAME': 'ETHPredict research', 'GIT_COMMITTER_EMAIL': 'research@invalid'}
    for name, value in environment.items():
        arguments += ['--env', f'{name}={value}']
    return arguments + ['--entrypoint', '/usr/bin/env', metadata['image'], *command]


PROBE = r'''
import errno,hashlib,json,os,socket,subprocess,sys
from pathlib import Path
spec=json.loads(sys.argv[1])
assert os.getuid()!=0
status=dict(line.split(':',1) for line in Path('/proc/self/status').read_text().splitlines() if ':' in line)
assert all(int(status[name].strip(),16)==0 for name in ['CapEff','CapPrm','CapBnd','CapAmb'])
assert status['NoNewPrivs'].strip()=='1'
assert status['Seccomp'].strip()=='2'
assert {name for _,name in socket.if_nameindex()}=={'lo'}
denied=[]
for value in spec['forbidden']:
 try:
  with open(value,'rb'): pass
 except (PermissionError,FileNotFoundError): denied.append(value)
 else: raise AssertionError('Forbidden host path is readable: '+value)
for value in ['/workspace/.agent-sandbox.json','/workspace/grader/__init__.py']:
 try:
  with open(value,'ab') as stream: stream.write(b'forbidden write')
 except OSError as error: assert error.errno in {errno.EROFS,errno.EACCES,errno.EPERM}
 else: raise AssertionError('Protected path is writable: '+value)
for folder in ['src','strategy']:
 path=Path('/workspace')/folder/'.sandbox-write-probe'
 path.write_bytes(b'editable');path.unlink()
for value in [spec['source_commit'],spec['historical_blob']]:
 result=subprocess.run(['git','-C','/workspace','cat-file','-e',value],capture_output=True)
 assert result.returncode!=0, 'Original historical object is available'
assert not subprocess.check_output(['git','-C','/workspace','remote']).strip()
assert not Path('/workspace/.git/objects/info/alternates').exists()
import pandas,torch
sample=next(Path('/workspace/data/lake/lighter/ETH/candles_5m').glob('*.parquet'))
frame=pandas.read_parquet(sample,columns=['timestamp_ms'])
assert len(frame)>0 and frame.timestamp_ms.max()<int(pandas.Timestamp('2026-05-01',tz='UTC').value//1_000_000)
print(json.dumps({'uid':os.getuid(),'host_paths_denied':len(denied),'historical_objects_absent':2,
 'network':'loopback only','capabilities':0,'no_new_privileges':True,'seccomp':True,
 'protected_code':'read_only','editable_paths':['src','strategy'],
 'dev_lake':'readable; pre-May partition verified','pandas':pandas.__version__,'torch':torch.__version__}))
'''


def probe(workspace: Path, extra_forbidden: list[str] | None = None) -> dict:
    metadata = load_workspace(workspace)
    source = Path(metadata['source_repo'])
    primary = Path(metadata['dev_lake']).parents[2]
    forbidden = ['/srv/ethpredict-grader/sealed.json', '/var/run/docker.sock',
        str(source / '.git/config'), str(source / 'data/raw/ETHUSDT-5m-lighter-20260328-20260628.csv'),
        str(source / 'context/funding_gap_recovery_20261027.json'),
        str(primary / 'data/raw/ETHUSDT-5m-lighter-20260328-20260628.csv'),
        str(primary / '.git/config'), str(primary / '.env'), str(primary / '.env.testnet'),
        str(primary / 'artifacts/first_bounded_16_17_20/_final_test_reuse_ledger.json')]
    forbidden += extra_forbidden or []
    spec = {'forbidden': sorted(set(forbidden)), 'source_commit': metadata['source_commit'],
            'historical_blob': metadata['historical_blob']}
    result = run(docker_command(metadata, [str(Path(metadata['venv'])/'bin/python'), '-c', PROBE, json.dumps(spec)]), timeout=90)
    return json.loads(result.stdout)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='mode', required=True)
    sub.add_parser('build-image')
    prepare = sub.add_parser('prepare')
    prepare.add_argument('--repo', type=Path, required=True)
    prepare.add_argument('--workspace', type=Path, required=True)
    prepare.add_argument('--dev-lake', type=Path, required=True)
    prepare.add_argument('--plan', type=Path, required=True)
    prepare.add_argument('--venv', type=Path, required=True)
    prepare.add_argument('--image', required=True)
    check = sub.add_parser('probe')
    check.add_argument('--workspace', type=Path, required=True)
    check.add_argument('--output', type=Path)
    execute = sub.add_parser('run')
    execute.add_argument('--workspace', type=Path, required=True)
    execute.add_argument('command', nargs=argparse.REMAINDER)
    args = parser.parse_args()
    if args.mode == 'build-image':
        print(build_image())
    elif args.mode == 'prepare':
        result = prepare_workspace(args.repo, args.workspace, args.dev_lake, args.plan, args.venv, args.image)
        print(json.dumps({'workspace': result['workspace'], 'code_files': len(result['code_hashes']),
                          'dev_files': len(result['dev_hashes']), 'source_commit': result['source_commit']}, indent=2))
    elif args.mode == 'probe':
        result = probe(args.workspace)
        if args.output:
            args.output.write_bytes(encoded(result))
        print(json.dumps(result, indent=2))
    else:
        command = args.command[1:] if args.command[:1] == ['--'] else args.command
        metadata = load_workspace(args.workspace)
        completed = subprocess.run(docker_command(metadata, command))
        raise SystemExit(completed.returncode)


if __name__ == '__main__':
    main()
