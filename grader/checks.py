"""Integrity checks executed by the frozen parent, never candidate assertions."""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
from pathlib import Path
import secrets
import subprocess

import numpy as np
import pandas as pd

from .data import DataContractError, LoadedDataset, MarketData, RESOLUTIONS
from .runner import WorkerFailure
from .feature_packets import same_packet


@dataclass(frozen=True)
class CheckResult:
    name: str
    passed: bool
    details: dict


def causal_slice(data: MarketData, *, end_ms: int, start_ms: int | None = None) -> MarketData:
    """Availability, rather than source-open time, governs every worker input."""
    result = {}
    for market, item in data.datasets.items():
        frame = item.frame
        if 'available_at_ms' not in frame or not pd.api.types.is_integer_dtype(frame.available_at_ms):
            raise DataContractError('History requires integer availability timestamps')
        mask = frame.available_at_ms.le(end_ms)
        if start_ms is not None:
            mask &= frame.timestamp_ms.ge(start_ms)
        selected = frame.loc[mask].copy().reset_index(drop=True)
        if selected.empty:
            raise DataContractError(f'No permitted history for {market}')
        provenance = {'source': item.provenance, 'slice': {
            'rows': len(selected), 'start_ms': int(selected.timestamp_ms.iloc[0]),
            'last_available_at_ms': int(selected.available_at_ms.iloc[-1]),
            'permitted_end_ms': int(end_ms)}}
        result[market] = LoadedDataset(selected, provenance)
    return MarketData(result)


def _same_positions(left: pd.DataFrame, right: pd.DataFrame, tolerance):
    index = ['timestamp', 'market']
    a, b = left.set_index(index).sort_index(), right.set_index(index).sort_index()
    return a.index.equals(b.index) and np.allclose(a.target_position, b.target_position,
                                                  rtol=0, atol=tolerance)


def causality_check(runner, state, history, decision_times, declaration, reference,
                    *, block, seed, cutoffs=20, tolerance=1e-9, reference_features=None):
    reference_features = reference_features if reference_features is not None else runner.last_features
    times = pd.DatetimeIndex(decision_times)
    if type(cutoffs) is not int or cutoffs < 20 or len(times) <= cutoffs:
        raise DataContractError('Causality requires at least 20 distinct proper-prefix cutoffs per block')
    # Draw only after the full reference prediction is committed. Policy RNG
    # seeds are deliberately visible to the strategy; audit locations must not
    # be derivable from them. This nonce remains solely in the scoring parent.
    rng = np.random.default_rng(secrets.randbits(128))
    indices = sorted(rng.choice(len(times)-1, size=cutoffs, replace=False).tolist())
    checked = []
    for index in indices:
        clock = times[:index+1]
        end = int(clock[-1].value // 1_000_000)
        prefix = causal_slice(history, end_ms=end)
        positions = runner.predict(state, prefix, clock, declaration)
        expected = reference[reference.timestamp.le(clock[-1])]
        checked.append(end)
        features_agree = same_packet(reference_features,runner.last_features,cutoff=clock[-1],tolerance=tolerance)
        if not _same_positions(positions, expected, tolerance) or not features_agree:
            return CheckResult('causality', False, {'block': block, 'policy_seed':seed,
                'cutoff_randomization':'parent-only entropy after reference prediction','cutoffs_ms': checked,
                'requested_cutoffs': cutoffs, 'mismatch_at_ms': end, 'tolerance': tolerance,
                'feature_inputs_agree':bool(features_agree)})
    return CheckResult('causality', True, {'block': block, 'policy_seed':seed,
        'cutoff_randomization':'parent-only entropy after reference prediction',
        'cutoffs_ms': checked, 'tolerance': tolerance})


def determinism_check(runner, train, history, decision_times, declaration, reference,
                      *, replay=None, dev_lake=None, train_anchor_end_ms=None, tolerance=1e-9):
    original_features = runner.last_features
    state = runner.fit(train, replay=replay, dev_lake=dev_lake,
                       train_anchor_end_ms=train_anchor_end_ms)
    repeated = runner.predict(state, history, decision_times, declaration)
    features_agree = same_packet(original_features,runner.last_features,tolerance=tolerance)
    return CheckResult('determinism', bool(_same_positions(repeated, reference, tolerance) and features_agree),
                       {'seed': runner.seed, 'repeated_fit_and_predict': True, 'tolerance': tolerance,
                        'feature_inputs_agree':bool(features_agree)})


def verify_frozen_grader(root: Path, *, tag='grader-v1') -> CheckResult:
    root = root.resolve(strict=True)
    try:
        names = subprocess.check_output(['git', '-C', str(root), 'ls-tree', '-rz', '--name-only', tag,
                                         '--', 'grader/'], stderr=subprocess.PIPE).split(b'\0')
        files = [name.decode() for name in names if name]
        if not files:
            raise ValueError('Tag has no grader files')
        expected_set = set(files)
        actual_set = {str(path.relative_to(root)) for path in (root / 'grader').rglob('*')
                      if path.is_file() and '__pycache__' not in path.parts}
        failures = sorted(expected_set ^ actual_set)
        hashes = {}
        for name in files:
            source = subprocess.check_output(['git', '-C', str(root), 'show', f'{tag}:{name}'], stderr=subprocess.PIPE)
            path = root / name
            digest = hashlib.sha256(source).hexdigest()
            hashes[name] = digest
            if path.is_symlink() or not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != digest:
                failures.append(name)
        return CheckResult('grader_hash', not failures, {'tag': tag, 'sha256': hashes,
                                                         'mismatches': sorted(set(failures))})
    except (OSError, subprocess.CalledProcessError, ValueError) as exc:
        return CheckResult('grader_hash', False, {'tag': tag, 'error': str(exc)})


def check_candidate_commit(repository: Path, base: str, head: str) -> CheckResult:
    """P4 reuses this gate; frozen grading ignores the candidate's grader copy."""
    try:
        changed = subprocess.check_output(['git', '-C', str(repository), 'diff', '--name-only',
                                           '--no-renames', '-z', base, head, '--']).split(b'\0')
        paths = [item.decode() for item in changed if item]
        forbidden = [p for p in paths if not (p.startswith('src/') or p.startswith('strategy/'))]
        entries = subprocess.check_output(['git', '-C', str(repository), 'ls-tree', '-rz', head,
                                           '--', 'src/', 'strategy/']).split(b'\0')
        forbidden += [entry.decode().split('\t')[-1] for entry in entries
                      if entry and not entry.startswith((b'100644 blob ', b'100755 blob '))]
        return CheckResult('allowed_commit_paths', not forbidden,
                           {'changed': paths, 'forbidden': sorted(set(forbidden))})
    except (OSError, UnicodeError, subprocess.CalledProcessError) as exc:
        return CheckResult('allowed_commit_paths', False, {'error': str(exc)})
