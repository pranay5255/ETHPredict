"""Fail-closed interpretation of the worker's private Linux syscall trace."""
from __future__ import annotations

import ast
from dataclasses import dataclass
from pathlib import Path
import re


@dataclass(frozen=True)
class AuditResult:
    opened: int
    violations: tuple[str, ...]
    complete: bool

    @property
    def passed(self):
        return self.complete and self.opened > 0 and not self.violations


def _under(path: Path, roots):
    return any(path == root or root.is_dir() and path.is_relative_to(root) for root in roots)


def audit_trace(content: str, *, server_pid: int, worker_pid: int, cwd: Path,
                read_only, writable) -> AuditResult:
    read = [Path(p).resolve() for p in read_only]
    write = [Path(p).resolve() for p in writable]
    opened, violations, exited, sealed = 0, [], False, False
    pending = {}
    for line in content.splitlines():
        match = re.match(r'^\s*(\d+)\s+(.*)$', line)
        if not match:
            violations.append('unparseable trace entry')
            continue
        pid, entry = int(match[1]), match[2]
        if pid == server_pid:
            continue
        if entry.endswith('<unfinished ...>'):
            pending[pid] = entry.removesuffix('<unfinished ...>')
            continue
        if entry.startswith('<... '):
            resumed = re.match(r'<\.\.\. (\w+) resumed>(.*)', entry)
            prefix = pending.pop(pid, None)
            if not resumed or prefix is None:
                violations.append('unmatched resumed syscall')
                continue
            entry = prefix + resumed[2]
        if pid == worker_pid and entry.startswith('landlock_restrict_self(') and entry.endswith(' = 0'):
            sealed = True
            continue
        if pid == worker_pid and (entry.startswith('+++ exited with ') or entry.startswith('+++ killed by ')):
            exited = True
        if not sealed:
            continue  # trusted bootstrap only; no candidate import/input read yet
        call = re.match(r'(\w+)\(', entry)
        if not call:
            continue  # exit/signal annotations
        name = call[1]
        if 'EACCES' in entry or 'EPERM' in entry:
            violations.append(f'permission-denied operation: {name}')
        if name in {'chdir', 'fchdir'}:
            # Trusted bootstrap changes cwd once, before installing seccomp.
            if entry == f'chdir("{cwd}") = 0':
                continue
            violations.append('worker attempted to change its fixed working directory')
            continue
        if name not in {'open', 'openat', 'openat2', 'creat', 'execve', 'execveat'}:
            # Metadata probes during Python imports do not disclose file contents.
            continue
        opened += 1
        quoted = re.search(r'"(?:[^"\\]|\\.)*"', entry)
        if not quoted or '...' in entry[:quoted.end()]:
            violations.append('unparseable or truncated file operation')
            continue
        try:
            raw = ast.literal_eval(quoted[0])
            path = Path(raw)
            if not path.is_absolute():
                if name in {'openat', 'openat2', 'execveat'}:
                    anchor = entry[entry.index('(') + 1:quoted.start()].rstrip(', ')
                    if re.fullmatch(r'AT_FDCWD(?:<[^>]+>)?', anchor):
                        annotation = re.fullmatch(r'AT_FDCWD<([^>]+)>', anchor)
                        if annotation and Path(annotation[1]) != cwd:
                            raise ValueError('Unexpected working directory')
                        base = cwd
                    else:
                        descriptor = re.fullmatch(r'\d+<(/[^>]+)>', anchor)
                        if descriptor is None:
                            raise ValueError('Unknown dirfd')
                        base = Path(descriptor[1])
                else:
                    base = cwd
                path = base / path
            path = path.resolve()
        except (ValueError, SyntaxError, OSError, RuntimeError):
            violations.append('unresolvable file operation')
            continue
        writing = name == 'creat' or bool(re.search(r'O_(?:WRONLY|RDWR|CREAT|TRUNC|APPEND)\b', entry))
        if not _under(path, write if writing else read + write):
            violations.append(f'outside allowed {"write" if writing else "read"}: {raw}')
        elif 'EACCES' in entry or 'EPERM' in entry:
            # Includes temporary symlink escapes, even if the link is removed.
            violations.append(f'permission-denied file operation: {raw}')
    return AuditResult(opened, tuple(dict.fromkeys(violations)), sealed and exited and not pending)
