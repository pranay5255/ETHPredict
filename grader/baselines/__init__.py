"""Baseline source materialization; no candidate module is imported here."""
from __future__ import annotations

import hashlib
import io
import json
from pathlib import Path
import subprocess
import tarfile

from ..data import DataContractError, RESOLUTIONS


NAMES = ('flat', 'buy_and_hold', 'momentum_1h', 'momentum_24h', 'timesfm_zero_shot', 'lstm_recipe')


def materialize(name, directory, *, markets, resolution, source_ref=None):
    if name not in NAMES or resolution not in RESOLUTIONS or not markets:
        raise DataContractError('Unknown baseline or decision clock')
    root = Path(__file__).resolve().parents[2]
    if source_ref is None:
        tagged = subprocess.run(['git','-C',str(root),'rev-parse','--verify','grader-v1'],capture_output=True).returncode == 0
        source_ref = 'grader-v1' if tagged else 'HEAD'
    source_commit = subprocess.check_output(['git','-C',str(root),'rev-parse',source_ref],text=True).strip()
    destination = Path(directory)
    destination.mkdir(parents=True)
    strategy = destination / 'strategy'
    strategy.mkdir()
    # .py.in files are worker payloads, not importable scoring-package modules.
    path = Path(__file__).with_name('strategies') / f'{name}.py.in'
    code = path.read_bytes()
    overrides = f'\nStrategy.markets = {list(markets)!r}\nStrategy.resolution = {resolution!r}\n'
    (strategy / 'strategy.py').write_bytes(code + overrides.encode())
    # Source dependencies come from Git objects, not a candidate's mutable src.
    if name in {'timesfm_zero_shot', 'lstm_recipe'}:
        archive = subprocess.check_output(['git', '-C', str(root), 'archive', '--format=tar', source_ref, '--', 'src/'])
        with tarfile.open(fileobj=io.BytesIO(archive), mode='r:') as members:
            for member in members:
                target = destination / member.name
                if member.isdir():
                    continue
                if not member.isfile() or not target.resolve().is_relative_to(destination.resolve()):
                    raise DataContractError('Unsafe baseline source archive')
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(members.extractfile(member).read())
    if name == 'lstm_recipe':
        recipe = Path(__file__).with_name('lstm_recipe.yaml')
        (strategy / 'recipe.yaml').write_bytes(recipe.read_bytes())
    identity = {'name': name, 'markets': list(markets), 'resolution': resolution,
                'source_ref': source_ref, 'source_commit':source_commit,
                'entry_sha256': hashlib.sha256(code).hexdigest()}
    (strategy / 'README.md').write_text('# Frozen baseline\n\n' + json.dumps(identity, indent=2) + '\n')
    return destination
