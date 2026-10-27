"""Human-operated, journaled physical partition sealing. Default: read-only plan.

--apply requires root and an existing dedicated ethgrader account; this program
never invokes sudo or creates users. No tier is evaluated. Recovery backups are
quarantined too. Existing tracked CSVs/Git history require a restricted agent
workspace: this script neither rewrites history nor claims to seal those copies.
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
from pathlib import Path
import pwd
import re
import shutil
import stat
import tempfile

import pandas as pd


CUTOFF = '2026-05'
MARKETS = ['ETH', 'BTC', 'SOL', 'HYPE', 'NEAR']
DATASETS = ['candles_1m', 'candles_5m', 'candles_1h', 'fundings_1h', 'mark_price_5m']


def digest(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def encoded(value):
    return (json.dumps(value, indent=2, sort_keys=True) + '\n').encode()


def require_regular(path):
    info = path.lstat()
    if not stat.S_ISREG(info.st_mode) or info.st_nlink != 1:
        raise ValueError(f'Refusing symlink, special file or hard-linked source: {path}')


def summary(frames, hashes, interval):
    stamps = pd.concat([frame.timestamp_ms for frame in frames], ignore_index=True).sort_values()
    differences = stamps.diff().dropna()
    if not pd.api.types.is_integer_dtype(stamps) or stamps.isna().any() or (differences % interval).any():
        raise ValueError('Invalid source clock during seal planning')
    return {'rows': len(stamps), 'empty_partitions': [name for name, frame in zip(hashes, frames) if frame.empty],
            'duplicates': int(stamps.duplicated().sum()),
            'first': str(pd.Timestamp(int(stamps.min()), unit='ms')) if len(stamps) else None,
            'last': str(pd.Timestamp(int(stamps.max()), unit='ms')) if len(stamps) else None,
            'expected_interval_ms': interval,
            'missing_intervals': int(((differences // interval) - 1).clip(lower=0).sum()),
            'largest_gap_intervals': int(differences.max() // interval) if len(differences) else 0,
            'partitions': hashes}


def make_plan(source: Path, private: Path) -> dict:
    source, private = source.resolve(strict=True), private.resolve()
    if private.is_relative_to(source) or source.is_relative_to(private):
        raise ValueError('Public and private roots must be separate')
    for path in source.rglob('*'):
        if path.is_symlink():
            raise ValueError(f'Refusing symlink in source lake: {path}')
    moves, manifests, known = [], [], set()
    for market in MARKETS:
        manifest_path = source / market / '_manifest.json'
        require_regular(manifest_path)
        raw = manifest_path.read_bytes()
        original = json.loads(raw)
        if original['symbol'] != market or set(original['datasets']) != set(DATASETS):
            raise ValueError(f'Unexpected market/dataset manifest: {market}')
        known.add(manifest_path)
        identity = {key: original[key] for key in ['symbol', 'market_id', 'market_type', 'source', 'network', 'listed'] if key in original}
        public = identity | {'tier_view': 'dev', 'datasets': {}}
        hidden = identity | {'tier_view': 'shadow_and_holdout', 'datasets': {}}
        for dataset in DATASETS:
            contract = original['datasets'][dataset]
            groups = {'dev': ([], {}), 'private': ([], {})}
            for name, expected in sorted(contract['partitions'].items()):
                if not re.fullmatch(r'\d{4}-(0[1-9]|1[0-2])\.parquet', name):
                    raise ValueError('Invalid manifest partition name')
                path = source / market / dataset / name
                require_regular(path)
                content = path.read_bytes()
                if hashlib.sha256(content).hexdigest() != expected:
                    raise ValueError(f'Source hash mismatch: {path}')
                known.add(path)
                group = 'dev' if name[:7] < CUTOFF else 'private'
                # Planning reads structural timestamps only, without repairing source values.
                frame = pd.read_parquet(io.BytesIO(content), columns=['timestamp_ms'])
                groups[group][0].append(frame)
                groups[group][1][name] = expected
                if group == 'private':
                    moves.append({'source': str(path.relative_to(source)),
                                  'destination': f'lake/lighter/{market}/{dataset}/{name}', 'sha256': expected})
            for target, group in [(public, 'dev'), (hidden, 'private')]:
                frames, hashes = groups[group]
                if not frames:
                    raise ValueError(f'Missing {group} partitions: {market}/{dataset}')
                target['datasets'][dataset] = summary(frames, hashes, contract['expected_interval_ms'])
        manifests.append({'market': market, 'original_text': raw.decode(),
                          'original_sha256': hashlib.sha256(raw).hexdigest(),
                          'public': public, 'private': hidden,
                          'uid': manifest_path.stat().st_uid, 'gid': manifest_path.stat().st_gid})
    recovery = source / '_funding_recovery'
    if recovery.exists():
        for path in sorted(recovery.rglob('*')):
            if path.is_file():
                require_regular(path)
                known.add(path)
                moves.append({'source': str(path.relative_to(source)),
                              'destination': 'recovery/' + str(path.relative_to(recovery)), 'sha256': digest(path)})
    actual = {path for path in source.rglob('*') if path.is_file()}
    if actual != known:
        raise ValueError(f'Unenrolled source files require review: {sorted(str(p) for p in actual-known)}')
    return {'version': 1, 'source_root': str(source), 'private_root': str(private),
            'private_owner': 'ethgrader', 'cutoff_month': CUTOFF, 'moves': moves,
            'manifests': manifests,
            'legacy_warning': 'Tracked CSVs, historical Git objects and derived experiment copies are not sealed by lake moves. Provision a restricted agent workspace before claiming full tier isolation.'}


def secure_directory(path: Path, root: Path, uid: int, gid: int):
    if not path.is_relative_to(root):
        raise ValueError('Private destination escapes root')
    chain = [path, *list(path.parents)]
    for directory in reversed(chain[:chain.index(root)+1]):
        if directory.is_symlink():
            raise ValueError('Symlink in private destination')
        directory.mkdir(exist_ok=True)
        os.chmod(directory, 0o700)
        os.chown(directory, uid, gid)


def atomic_write(path: Path, content: bytes, uid: int, gid: int, mode=0o600):
    descriptor, temporary = tempfile.mkstemp(prefix='.seal-', dir=path.parent)
    try:
        with os.fdopen(descriptor, 'wb') as stream:
            os.fchmod(stream.fileno(), mode)
            os.fchown(stream.fileno(), uid, gid)
            stream.write(content)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        directory_fd = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def apply_plan(plan: dict, uid: int, gid: int) -> dict:
    """Apply/resume an already verified plan. CLI performs the root/account gate."""
    source, private = Path(plan['source_root']), Path(plan['private_root'])
    secure_directory(private, private, uid, gid)
    plan_path = private / 'seal-plan.json'
    if plan_path.exists():
        if plan_path.read_bytes() != encoded(plan):
            raise ValueError('Stored seal plan differs; do not replace it')
    else:
        atomic_write(plan_path, encoded(plan), uid, gid)
    # Recheck all source/destination bytes before the first move on every attempt.
    for move in plan['moves']:
        src, dst = source / move['source'], private / move['destination']
        for path in [src, dst]:
            if path.exists():
                require_regular(path)
                if digest(path) != move['sha256']:
                    raise ValueError(f'Changed source or destination: {path}')
        if not src.exists() and not dst.exists():
            raise ValueError(f'Missing both source and destination: {src}')
    journal = private / 'seal-journal.jsonl'
    if not journal.exists():
        atomic_write(journal, b'', uid, gid)
    if journal.is_symlink():
        raise ValueError('Refusing journal symlink')
    for item in plan['manifests']:
        directory = private / 'original_manifests' / item['market']
        secure_directory(directory, private, uid, gid)
        path = directory / '_manifest.json'
        content = item['original_text'].encode()
        if path.exists() and path.read_bytes() != content:
            raise ValueError('Original manifest archive changed')
        atomic_write(path, content, uid, gid)
    for move in plan['moves']:
        src, dst = source / move['source'], private / move['destination']
        secure_directory(dst.parent, private, uid, gid)
        if src.exists():
            # Copy/verify/rename before removing source works across filesystems and
            # preserves restartability if interrupted before source removal.
            if not dst.exists():
                descriptor, name = tempfile.mkstemp(prefix='.seal-copy-', dir=dst.parent)
                temporary = Path(name)
                with src.open('rb') as reader, os.fdopen(descriptor, 'wb') as writer:
                    os.fchmod(writer.fileno(), 0o600)
                    os.fchown(writer.fileno(), uid, gid)
                    shutil.copyfileobj(reader, writer)
                    writer.flush()
                    os.fsync(writer.fileno())
                if digest(temporary) != move['sha256']:
                    raise ValueError('Private copy hash mismatch; source retained')
                os.replace(temporary, dst)
                directory_fd = os.open(dst.parent, os.O_RDONLY | os.O_DIRECTORY)
                try:
                    os.fsync(directory_fd)
                finally:
                    os.close(directory_fd)
            if digest(dst) != move['sha256']:
                raise ValueError('Destination changed; source retained')
            # Recheck source immediately before removal; do not discard a changed file.
            if digest(src) != move['sha256']:
                raise ValueError('Source changed during copy; source retained')
            src.unlink()
            directory_fd = os.open(src.parent, os.O_RDONLY | os.O_DIRECTORY)
            try:
                os.fsync(directory_fd)
            finally:
                os.close(directory_fd)
        os.chmod(dst, 0o600)
        os.chown(dst, uid, gid)
        with journal.open('a') as stream:
            stream.write(json.dumps({'moved': move['source'], 'sha256': move['sha256']}) + '\n')
            stream.flush()
            os.fsync(stream.fileno())
    for item in plan['manifests']:
        path = source / item['market'] / '_manifest.json'
        allowed = [item['original_sha256'], hashlib.sha256(encoded(item['public'])).hexdigest()]
        if digest(path) not in allowed:
            raise ValueError('Public manifest changed during sealing')
        directory = private / 'lake/lighter' / item['market']
        secure_directory(directory, private, uid, gid)
        atomic_write(directory / '_manifest.json', encoded(item['private']), uid, gid)
        atomic_write(path, encoded(item['public']), item['uid'], item['gid'], mode=0o644)
    recovery = source / '_funding_recovery'
    if recovery.exists():
        for directory in sorted((p for p in recovery.rglob('*') if p.is_dir()), key=lambda p: len(p.parts), reverse=True):
            directory.rmdir()  # Fails on unexpected new files; never recursive deletion.
        recovery.rmdir()
    result = verify_private(plan, uid)
    atomic_write(private / 'sealed.json', encoded(result), uid, gid)
    return result


def verify_private(plan: dict, uid: int) -> dict:
    source, private = Path(plan['source_root']), Path(plan['private_root'])
    for path in [private, *private.rglob('*')]:
        info = path.lstat()
        if path.is_symlink() or info.st_uid != uid or stat.S_IMODE(info.st_mode) != (0o700 if path.is_dir() else 0o600):
            raise ValueError(f'Wrong ownership/type/mode in private storage: {path}')
    for move in plan['moves']:
        if (source / move['source']).exists() or digest(private / move['destination']) != move['sha256']:
            raise ValueError('Incomplete physical move')
    for item in plan['manifests']:
        if (source / item['market'] / '_manifest.json').read_bytes() != encoded(item['public']):
            raise ValueError('Public manifest differs from dev view')
        if (private / 'lake/lighter' / item['market'] / '_manifest.json').read_bytes() != encoded(item['private']):
            raise ValueError('Private manifest differs')
        if digest(private / 'original_manifests' / item['market'] / '_manifest.json') != item['original_sha256']:
            raise ValueError('Original manifest hash differs')
    expected_public = {source / item['market'] / '_manifest.json' for item in plan['manifests']}
    expected_public.update(source / item['market'] / dataset / name
                           for item in plan['manifests'] for dataset, contract in item['public']['datasets'].items()
                           for name in contract['partitions'])
    if {path for path in source.rglob('*') if path.is_file()} != expected_public:
        raise ValueError('Unexpected files remain in public lake')
    return {'lake_sealed': True, 'private_files_moved': len(plan['moves']),
            'private_uid': uid, 'overall_agent_isolation': 'requires separate agent denial and legacy-copy checks',
            'legacy_warning': plan['legacy_warning']}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument('--plan', action='store_true')
    modes.add_argument('--apply', action='store_true')
    modes.add_argument('--verify-private', action='store_true')
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--private', type=Path, default=Path('/srv/ethpredict-grader'))
    parser.add_argument('--output', type=Path)
    parser.add_argument('--approved-plan', type=Path,
                        help='Previously reviewed --plan output; required on first --apply')
    args = parser.parse_args()
    if args.apply or args.verify_private:
        if args.apply and os.geteuid() != 0:
            parser.error('--apply is human-only: invoke as root after reviewing the plan')
        owner = pwd.getpwnam('ethgrader')
        if args.private.resolve() != Path('/srv/ethpredict-grader'):
            parser.error('Mutating/private verification target must be /srv/ethpredict-grader')
        if owner.pw_uid == 0 or owner.pw_uid == args.source.stat().st_uid:
            parser.error('ethgrader must be a distinct unprivileged account')
        if args.private.is_symlink():
            parser.error('Private root must not be a symlink')
        stored = args.private / 'seal-plan.json'
        if args.private.exists():
            if args.private.stat().st_uid not in {0, owner.pw_uid}:
                parser.error('Private root is owned by an unexpected account')
            if any(args.private.iterdir()) and not stored.exists():
                parser.error('Nonempty private root has no seal journal; stop for review')
        if stored.exists():
            require_regular(stored)
        if stored.exists():
            plan = json.loads(stored.read_text())
            if args.approved_plan and json.loads(args.approved_plan.read_text()) != plan:
                parser.error('Approved plan differs from stored resume plan')
        else:
            if args.verify_private:
                parser.error('No stored seal plan: sealing has not completed')
            if not args.approved_plan:
                parser.error('First --apply requires --approved-plan from the reviewed read-only plan')
            require_regular(args.approved_plan)
            plan = make_plan(args.source, args.private)
            if json.loads(args.approved_plan.read_text()) != plan:
                parser.error('Source inventory changed since review: regenerate and review the plan')
        if plan['source_root'] != str(args.source.resolve()) or plan['private_root'] != str(args.private.resolve()):
            parser.error('Stored plan root mismatch')
        result = apply_plan(plan, owner.pw_uid, owner.pw_gid) if args.apply else verify_private(plan, owner.pw_uid)
        print(json.dumps(result, indent=2))
    else:
        plan = make_plan(args.source, args.private)
        if args.output:
            args.output.write_bytes(encoded(plan))
        print(json.dumps({'mode': 'read_only_plan', 'partitions_to_move': sum(m['destination'].startswith('lake/') for m in plan['moves']),
                          'recovery_files_to_quarantine': sum(m['destination'].startswith('recovery/') for m in plan['moves']),
                          'source': plan['source_root'], 'private': plan['private_root'],
                          'legacy_warning': plan['legacy_warning']}, indent=2))


if __name__ == '__main__':
    main()
