"""Pinned model identity and explicit release/contamination reporting."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import shutil

import pandas as pd

from .data import DataContractError, RESOLUTIONS, _json_unique


def registry():
    value = json.loads(Path(__file__).with_suffix('.json').read_text(), object_pairs_hook=_json_unique)
    if set(value) != {'version', 'models'} or value['version'] != 1 or not isinstance(value['models'], dict):
        raise DataContractError('Unknown model registry')
    expected = {'repository', 'revision', 'license', 'release_date', 'release_evidence', 'license_evidence', 'files'}
    for key, item in value['models'].items():
        if set(item) != expected or not item['license'] or not isinstance(item['files'], dict):
            raise DataContractError(f'Invalid registered model: {key}')
    return value['models']


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as source:
        for chunk in iter(lambda: source.read(8 * 1024**2), b''):
            digest.update(chunk)
    return digest.hexdigest()


def cached_model_paths():
    """Locate only the pinned public snapshots; do not inspect authentication."""
    result = {}
    for identifier,item in registry().items():
        path = Path.home()/'.cache/huggingface/hub'/('models--'+item['repository'].replace('/','--'))/'snapshots'/item['revision']
        if path.is_dir() and all((path/name).is_file() for name in item['files']):
            result[identifier] = path
    return result


def verified_model_paths(identifiers, paths):
    """No network, implicit token, unregistered weight or unverified revision."""
    known, result = registry(), {}
    for identifier in identifiers:
        if identifier not in known or identifier not in paths:
            raise DataContractError(f'Model is not registered/provisioned: {identifier}')
        root = Path(paths[identifier]).resolve(strict=True)
        for filename, digest in known[identifier]['files'].items():
            path = root / filename
            if not path.is_file() or sha256_file(path) != digest:
                raise DataContractError(f'Model SHA-256 mismatch: {identifier}/{filename}')
        result[identifier] = root
    return result


def stage_models(identifiers, paths, destination):
    """Make regular-file snapshots so HF's external blob symlinks cannot escape."""
    verified = verified_model_paths(identifiers, paths)
    result = {}
    known = registry()
    for identifier, root in verified.items():
        target = Path(destination) / identifier
        target.mkdir(parents=True)
        for filename, digest in known[identifier]['files'].items():
            copied = target / filename
            shutil.copyfile(root / filename, copied, follow_symlinks=True)
            if sha256_file(copied) != digest:
                raise DataContractError('Model changed while snapshotting')
        result[identifier] = target
    return result


def contamination(identifiers, *, start_ms, end_ms, holdout_start_ms):
    result = []
    for identifier in identifiers:
        item = registry()[identifier]
        # A dated announcement does not establish an intra-day publication time.
        boundary = pd.Timestamp(item['release_date'], tz='UTC') + pd.Timedelta(days=1) if item['release_date'] else None
        release_ms = int(boundary.value // 1_000_000) if boundary is not None else None
        possible = release_ms is None or start_ms < release_ms
        result.append({'model': identifier, **item,
            'release_safe_after_ms': release_ms,
            'period_label': 'possibly in pretraining' if possible else 'after documented public release',
            'possibly_pretraining_window': {'start_ms': start_ms, 'end_ms': min(end_ms, release_ms)}
                if release_ms is not None and possible else {'start_ms': start_ms, 'end_ms': end_ms} if possible else None,
            'holdout_release_eligible': release_ms is not None and holdout_start_ms >= release_ms,
            'note': 'Release timing does not establish the complete pretraining corpus or statistical independence.'})
    return result


def block_contamination(identifiers, blocks, clock, *, resolution, max_horizon_bars, holdout_start_ms):
    """Separate visible fit history from each block's execution evidence.

    Raw fitting includes the declared label horizon. Evaluation begins at the
    first next-open execution and includes the reserved final liquidation bar.
    All windows are half-open, matching the lake/tier convention.
    """
    if not identifiers:
        return []
    interval = RESOLUTIONS[resolution]
    result = []
    for block in blocks:
        train_end = int(block.train[-1])+max_horizon_bars
        last_execution = int(block.evaluation[-1])+1
        if max(train_end,last_execution) >= len(clock):
            raise DataContractError('Model provenance window escapes source data')
        training = {'start_ms':int(clock[0]),'end_ms':int(clock[train_end])+interval}
        evaluation = {'start_ms':int(clock[int(block.evaluation[0])])+interval,
                      'end_ms':int(clock[last_execution])+interval}
        result.append({'block':block.block+1,'training_window':training,'evaluation_window':evaluation,
            'training':contamination(identifiers,**training,holdout_start_ms=holdout_start_ms),
            'evaluation':contamination(identifiers,**evaluation,holdout_start_ms=holdout_start_ms)})
    return result
