"""Hash trial artifacts without interpreting candidate checkpoints or links."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import stat

from .contracts import regular_bytes
from .data import DataContractError, _json_unique


MANIFEST = 'artifact_manifest.json'
EXCLUDED = {MANIFEST, 'grade_report.json'}


def _file_record(path, relative):
    descriptor = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode) or before.st_nlink != 1:
            raise DataContractError('Artifact must be a regular file without hard links')
        digest = hashlib.sha256()
        count = 0
        while chunk := os.read(descriptor, 8*1024**2):
            count += len(chunk)
            digest.update(chunk)
        after = os.fstat(descriptor)
        if (before.st_ino,before.st_size,before.st_mtime_ns,before.st_ctime_ns) != (
                after.st_ino,after.st_size,after.st_mtime_ns,after.st_ctime_ns) or count != before.st_size:
            raise DataContractError('Artifact changed while hashing')
        return {'path':relative, 'bytes':count, 'sha256':digest.hexdigest()}
    finally:
        os.close(descriptor)


def _records(directory):
    result = []
    def fail(error):
        raise error
    for current, directories, files in os.walk(directory, followlinks=False, onerror=fail):
        for name in directories:
            if (Path(current)/name).is_symlink():
                raise DataContractError('Artifact directories cannot be symbolic links')
        for name in sorted(files):
            path = Path(current)/name
            relative = path.relative_to(directory).as_posix()
            if relative not in EXCLUDED:
                result.append(_file_record(path,relative))
    return sorted(result,key=lambda row:row['path'])


def capture_artifacts(directory):
    """Run only after the worker and all its descendants have exited.

    The report references this manifest's hash. Excluding that report and the
    manifest itself avoids a hash cycle and permits later statistical rescoring.
    All other files, including inputs, positions, traces and opaque checkpoints,
    are covered. No candidate pickle is deserialized by this module.
    """
    directory = Path(directory).resolve(strict=True)
    files = _records(directory)
    content = json.dumps({'version':1,'files':files},sort_keys=True,indent=2,allow_nan=False).encode()+b'\n'
    path = directory/MANIFEST
    # A candidate cannot choose a link target for a trusted manifest write.
    descriptor = os.open(path,os.O_WRONLY|os.O_CREAT|os.O_EXCL|os.O_NOFOLLOW,0o600)
    with os.fdopen(descriptor,'wb') as target:
        target.write(content)
    return {'path':MANIFEST,'sha256':hashlib.sha256(content).hexdigest(),
            'files':len(files),'bytes':sum(row['bytes'] for row in files)}


def verify_artifacts(directory, reference):
    directory = Path(directory).resolve(strict=True)
    if (not isinstance(reference,dict) or set(reference) != {'path','sha256','files','bytes'}
            or reference['path'] != MANIFEST or not isinstance(reference['sha256'],str)
            or not re.fullmatch('[a-f0-9]{64}',reference['sha256'])
            or any(type(reference[name]) is not int or reference[name]<0 for name in ['files','bytes'])):
        raise DataContractError('Unknown artifact manifest reference')
    content = regular_bytes(directory/MANIFEST,maximum=64*1024**2)
    if hashlib.sha256(content).hexdigest() != reference['sha256']:
        raise DataContractError('Artifact manifest hash mismatch')
    value = json.loads(content,object_pairs_hook=_json_unique)
    if (not isinstance(value,dict) or set(value) != {'version','files'} or type(value['version']) is not int
            or value['version'] != 1 or not isinstance(value['files'],list)):
        raise DataContractError('Unknown artifact manifest fields/version')
    for record in value['files']:
        if (not isinstance(record,dict) or set(record) != {'path','bytes','sha256'} or not isinstance(record['path'],str)
                or type(record['bytes']) is not int or record['bytes']<0 or not isinstance(record['sha256'],str)
                or not re.fullmatch('[a-f0-9]{64}',record['sha256'])):
            raise DataContractError('Unknown artifact record fields')
        path = PurePosixPath(record['path'])
        if path.is_absolute() or not path.parts or any(part in {'.','..'} for part in path.parts) or path.as_posix()!=record['path']:
            raise DataContractError('Artifact path escapes its trial directory')
    actual = _records(directory)
    if actual != value['files'] or len(actual) != reference['files'] or sum(row['bytes'] for row in actual) != reference['bytes']:
        raise DataContractError('Trial artifacts are missing, changed or unmanifested')
    return value
