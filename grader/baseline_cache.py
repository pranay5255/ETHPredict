"""Trusted baseline initialization for each declared market/clock population."""
from __future__ import annotations

import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import uuid

from .baseline_bank import build_bank, grading_fingerprint, read_bank
from .contracts import regular_bytes
from .data import DataContractError, _json_unique
from .grade import data_identity
from .models import sha256_file


def comparable_bank(bank, declaration, data):
    if (bank['data_sha256'] != data_identity(data)['sha256'] or bank['tier'] != data.tier
            or bank['resolution'] != declaration.resolution or set(bank['markets']) != set(declaration.markets)):
        raise DataContractError('Baseline bank does not match every declared market, clock and source block')
    return bank


def get_bank(lake, cache_root, declaration, data, *, model_paths=None, private_root=None,
             device='cuda', progress=None):
    """Build only missing populations; corruption is an error, never a reset.

    Each attempt has a stable artifact directory. An atomic index is published
    only after a complete bank passes validation. Failed/interrupted attempts
    are retained. The key lock serializes initialization and is released by the
    OS on owner exit; it is never inferred from a stale marker's existence.
    The cache belongs to the trusted coordinator, outside the loop worker.
    """
    contract = {'markets':sorted(declaration.markets),'resolution':declaration.resolution,'tier':data.tier,
                'data_sha256':data_identity(data)['sha256'],'grading_fingerprint':grading_fingerprint()}
    key = hashlib.sha256(json.dumps(contract,sort_keys=True).encode()).hexdigest()
    root = Path(cache_root).resolve()
    root.mkdir(parents=True,exist_ok=True,mode=0o700)
    lock = os.open(root/f'{key}.lock',os.O_RDWR|os.O_CREAT|os.O_NOFOLLOW,0o600)
    try:
        fcntl.flock(lock,fcntl.LOCK_EX)
        index = root/f'{key}.json'
        if index.exists() or index.is_symlink():
            record = json.loads(regular_bytes(index,maximum=4096),object_pairs_hook=_json_unique)
            if (not isinstance(record,dict) or set(record) != {'version','bank','sha256'}
                    or type(record['version']) is not int or record['version'] != 1
                    or not isinstance(record['bank'],str)
                    or not re.fullmatch(key+r'-[a-f0-9]{32}/baseline_bank.json',record['bank'])):
                raise DataContractError('Unknown or escaping baseline cache index')
            bank_path = root/record['bank']
            if not bank_path.resolve(strict=True).is_relative_to(root) or sha256_file(bank_path) != record['sha256']:
                raise DataContractError('Baseline cache index hash/path mismatch')
            return comparable_bank(read_bank(bank_path),declaration,data)
        attempt = root/f'{key}-{uuid.uuid4().hex}'
        bank = build_bank(lake,attempt,markets=sorted(declaration.markets),resolution=declaration.resolution,
            tier=data.tier,private_root=private_root,model_paths=model_paths or {},device=device,progress=progress)
        if not bank['complete']:
            raise DataContractError(f'Baseline initialization failed; retained evidence: {attempt}')
        bank_path = attempt/'baseline_bank.json'
        bank = comparable_bank(read_bank(bank_path),declaration,data)
        record = {'version':1,'bank':bank_path.relative_to(root).as_posix(),'sha256':sha256_file(bank_path)}
        temporary = root/f'.{key}-{uuid.uuid4().hex}.tmp'
        descriptor = os.open(temporary,os.O_WRONLY|os.O_CREAT|os.O_EXCL|os.O_NOFOLLOW,0o600)
        with os.fdopen(descriptor,'w') as target:
            json.dump(record,target,sort_keys=True,allow_nan=False)
            target.write('\n');target.flush();os.fsync(target.fileno())
        os.replace(temporary,index)
        return bank
    finally:
        os.close(lock)
