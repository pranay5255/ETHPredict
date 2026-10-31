"""Bounded source proofs and a file-free Unix-socket request protocol.

Only the confined loop client runs Git. The private service verifies raw Git
objects itself; it never opens an agent checkout or interprets its Git config.
"""
from __future__ import annotations

import base64
import hashlib
import json
import re
import socket
import struct
import subprocess


MAX_SOURCE = 32 * 1024**2
MAX_MESSAGE = 48 * 1024**2
MAX_FILES = 8192
OID = re.compile('[a-f0-9]{40}')
EDITABLE = {'src', 'strategy'}


def unique(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError('Repeated JSON key')
        result[key] = value
    return result


def encode(value):
    result = json.dumps(value, separators=(',', ':'), allow_nan=False).encode()
    if len(result) > MAX_MESSAGE:
        raise ValueError('Request exceeds the protocol size limit')
    return result


def _receive(stream, length):
    result = bytearray()
    while len(result) < length:
        part = stream.recv(min(length - len(result), 1024**2))
        if not part:
            raise EOFError('Incomplete broker message')
        result.extend(part)
    return bytes(result)


def receive(stream):
    length = struct.unpack('!I', _receive(stream, 4))[0]
    if not 0 < length <= MAX_MESSAGE:
        raise ValueError('Invalid broker message size')
    return json.loads(_receive(stream, length), object_pairs_hook=unique,
        parse_constant=lambda token: (_ for _ in ()).throw(ValueError('Nonfinite JSON')))


def send(stream, value):
    body = encode(value)
    stream.sendall(struct.pack('!I', len(body)) + body)


def request(path, value, *, timeout=30):
    with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as stream:
        stream.settimeout(timeout)
        stream.connect(str(path))
        send(stream, value)
        return receive(stream)


def object_id(kind, body):
    return hashlib.sha1(kind.encode() + b' ' + str(len(body)).encode() + b'\0' + body).hexdigest()


def _bytes(value, maximum):
    if not isinstance(value, str) or len(value) > (maximum + 2) // 3 * 4:
        raise ValueError('Oversized source object')
    body = base64.b64decode(value, validate=True)
    if len(body) > maximum:
        raise ValueError('Oversized source object')
    return body


def parse_tree(body):
    result, offset = {}, 0
    while offset < len(body):
        end = body.index(b'\0', offset)
        mode, name = body[offset:end].split(b' ', 1)
        name = name.decode('utf-8')
        if (not name or name in {'.', '..', '.git'} or '/' in name or '\\' in name
                or any(ord(character) < 32 or ord(character) == 127 for character in name)
                or name in result or len(body) - end - 1 < 20):
            raise ValueError('Unsafe Git tree entry')
        if mode not in {b'40000', b'100644', b'100755'}:
            raise ValueError('Source trees must contain only regular files and directories')
        result[name] = (mode.decode(), body[end+1:end+21].hex())
        offset = end + 21
        if len(result) > MAX_FILES:
            raise ValueError('Too many tree entries')
    return result


def parse_commit(body):
    header = body.split(b'\n\n', 1)[0].splitlines()
    trees = [line[5:].decode('ascii') for line in header if line.startswith(b'tree ')]
    parents = [line[7:].decode('ascii') for line in header if line.startswith(b'parent ')]
    if len(trees) != 1 or len(parents) > 1 or any(not OID.fullmatch(oid) for oid in trees + parents):
        raise ValueError('Use one linear candidate commit with a valid tree')
    return trees[0], parents


def verify_source(proof, protected):
    """Return verified candidate bytes, without Git, imports or extraction tools.

    Both the candidate and its immediate parent's protected root entries must
    match the provisioned code-only workspace. A forbidden edit followed by a
    revert is therefore rejected too. Every editable blob is hash-verified.
    """
    if not isinstance(proof, dict) or set(proof) != {'commit', 'commits', 'trees', 'blobs'}:
        raise ValueError('Unknown source proof fields')
    commit = proof['commit']
    if not isinstance(commit, str) or not OID.fullmatch(commit):
        raise ValueError('Invalid commit identity')
    decoded = {}
    for kind, maximum in [('commits', 16384), ('trees', 1024**2), ('blobs', MAX_SOURCE)]:
        values = proof[kind]
        if not isinstance(values, dict) or len(values) > MAX_FILES:
            raise ValueError('Invalid source object collection')
        decoded[kind] = {}
        total = 0
        for oid, value in values.items():
            if not isinstance(oid, str) or not OID.fullmatch(oid):
                raise ValueError('Invalid object identity')
            body = _bytes(value, maximum)
            total += len(body)
            if total > (MAX_SOURCE if kind == 'blobs' else 4 * 1024**2):
                raise ValueError('Source object collection exceeds its limit')
            if object_id({'commits': 'commit', 'trees': 'tree', 'blobs': 'blob'}[kind], body) != oid:
                raise ValueError('Git object hash mismatch')
            decoded[kind][oid] = body
    root, parents = parse_commit(decoded['commits'][commit])
    if set(decoded['commits']) != {commit, *parents}:
        raise ValueError('Incomplete or extraneous commit proof')
    used_trees, used_blobs = set(), set()

    def tree(oid):
        used_trees.add(oid)
        return parse_tree(decoded['trees'][oid])

    def check_root(oid):
        entries = tree(oid)
        actual = {name: list(item) for name, item in entries.items() if name not in EDITABLE}
        if actual != protected or not EDITABLE.issubset(entries):
            raise ValueError('Candidate commit touches forbidden paths')
        if any(entries[name][0] != '40000' for name in EDITABLE):
            raise ValueError('Editable roots must be directories')
        return entries

    entries = check_root(root)
    for parent in parents:
        parent_root, _ = parse_commit(decoded['commits'][parent])
        check_root(parent_root)
    files = {}

    def walk(oid, prefix, depth=0):
        if depth > 32:
            raise ValueError('Source tree is too deeply nested')
        for name, (mode, child) in tree(oid).items():
            path = f'{prefix}/{name}'
            if mode == '40000':
                walk(child, path, depth + 1)
            else:
                used_blobs.add(child)
                files[path] = (int(mode, 8), decoded['blobs'][child])
                if len(files) > MAX_FILES:
                    raise ValueError('Too many source files')

    for name in sorted(EDITABLE):
        walk(entries[name][1], name)
    if used_trees != set(decoded['trees']) or used_blobs != set(decoded['blobs']):
        raise ValueError('Extraneous objects in source proof')
    if sum(len(body) for _, body in files.values()) > MAX_SOURCE:
        raise ValueError('Decoded candidate source exceeds its limit')
    if 'strategy/strategy.py' not in files or 'strategy/README.md' not in files:
        raise ValueError('Strategy code and pre-run hypothesis are required')
    return files


def source_proof(repository):
    """CLIENT ONLY: call inside the isolated tool worker, never on its host."""
    def git(*arguments):
        return subprocess.check_output(['git', '-C', str(repository), '-c', 'core.fsmonitor=false',
            '-c', 'core.hooksPath=/dev/null', *arguments], timeout=30)
    commit = git('rev-parse', 'HEAD').decode().strip()
    commits, trees, blobs = {}, {}, {}
    raw = git('cat-file', 'commit', commit)
    commits[commit] = raw
    root, parents = parse_commit(raw)
    for parent in parents:
        raw_parent = git('cat-file', 'commit', parent)
        commits[parent] = raw_parent
        parent_root, _ = parse_commit(raw_parent)
        trees[parent_root] = git('cat-file', 'tree', parent_root)
    trees[root] = git('cat-file', 'tree', root)

    def walk(oid):
        trees[oid] = git('cat-file', 'tree', oid)
        for mode, child in parse_tree(trees[oid]).values():
            if mode == '40000':
                walk(child)
            elif child not in blobs:
                blobs[child] = git('cat-file', 'blob', child)
    for name, (mode, oid) in parse_tree(trees[root]).items():
        if name in EDITABLE:
            if mode != '40000':
                raise ValueError('Editable roots must be directories')
            walk(oid)
    result = {'commit': commit, **{kind: {oid: base64.b64encode(body).decode()
        for oid, body in values.items()} for kind, values in [('commits', commits), ('trees', trees), ('blobs', blobs)]}}
    encode(result)
    return result


def protected_entries(proof):
    root, _ = parse_commit(_bytes(proof['commits'][proof['commit']], 16384))
    return {name: list(item) for name, item in parse_tree(_bytes(proof['trees'][root], 1024**2)).items()
            if name not in EDITABLE}
