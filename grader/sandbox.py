"""Linux child-only restrictions for strategy execution, without sudo.

Landlock confines reads/writes to explicitly supplied paths. Seccomp blocks
network access, process-memory/FD theft, namespace/mount changes and metadata
operations that could weaken the private Unix permissions. Set these before
importing candidate code, after closing inherited control/file descriptors.
"""
from __future__ import annotations

import ctypes
import errno
import os
from pathlib import Path
import platform


_LIBC = ctypes.CDLL(None, use_errno=True)
_LIBC.syscall.restype = ctypes.c_long
READ_FILE, READ_DIR, EXECUTE = 1 << 2, 1 << 3, 1
WRITE_FILE, TRUNCATE = 1 << 1, 1 << 14
ALL_FS_V3 = (1 << 15) - 1


class Ruleset(ctypes.Structure):
    _fields_ = [('handled_access_fs', ctypes.c_uint64)]


class PathRule(ctypes.Structure):
    _pack_ = 1
    _fields_ = [('allowed_access', ctypes.c_uint64), ('parent_fd', ctypes.c_int32)]


class ArgCompare(ctypes.Structure):
    _fields_ = [('arg',ctypes.c_uint),('op',ctypes.c_uint),
                ('datum_a',ctypes.c_uint64),('datum_b',ctypes.c_uint64)]


class FilterInstruction(ctypes.Structure):
    _fields_=[('code',ctypes.c_uint16),('jt',ctypes.c_uint8),('jf',ctypes.c_uint8),('k',ctypes.c_uint32)]


class FilterProgram(ctypes.Structure):
    _fields_=[('length',ctypes.c_uint16),('instructions',ctypes.POINTER(FilterInstruction))]


def _deny_new_syscall_numbers():
    # This reviewed profile ends at fchmodat2 (452). Future metadata/namespace
    # APIs, unknown to the installed libseccomp, must not bypass its deny rules.
    architecture={'x86_64':0xc000003e,'aarch64':0xc00000b7}[platform.machine()]
    entries=[(0x20,0,0,4),(0x15,1,0,architecture),(0x06,0,0,0x80000000),
             (0x20,0,0,0),(0x35,0,1,453),
             (0x06,0,0,0x00050000|errno.EACCES),(0x06,0,0,0x7fff0000)]
    instructions=(FilterInstruction*len(entries))(*[FilterInstruction(*entry) for entry in entries])
    program=FilterProgram(len(entries),instructions)
    _check(_LIBC.prctl(22,2,ctypes.byref(program),0,0),'seccomp syscall ceiling')


def _check(result, operation):
    if result < 0:
        code = ctypes.get_errno()
        raise OSError(code, f'{operation}: {os.strerror(code)}')
    return result


def protect_parent_memory():
    """The trusted grader/server must call this BEFORE loading private inputs."""
    _check(_LIBC.prctl(4, 0, 0, 0, 0), 'PR_SET_DUMPABLE')


def landlock_abi():
    if platform.machine() not in {'x86_64','aarch64'}:
        raise RuntimeError('Unsupported Linux syscall architecture')
    return int(_check(_LIBC.syscall(444,ctypes.c_void_p(),0,1),'Landlock ABI'))


def restrict_filesystem(*, read_only, writable):
    """Every path is a file or directory; unavailable confinement fails closed."""
    if landlock_abi() < 3:
        raise RuntimeError('Landlock ABI >=3 is required for truncate confinement')
    specification = Ruleset(ALL_FS_V3)
    descriptor = _check(_LIBC.syscall(444,ctypes.byref(specification),ctypes.sizeof(specification),0),'Landlock ruleset')
    try:
        for paths, write in [(read_only,False),(writable,True)]:
            for raw in paths:
                path = Path(raw).resolve(strict=True)
                handle = os.open(path,os.O_PATH|os.O_CLOEXEC)
                try:
                    directory = path.is_dir()
                    permissions = (ALL_FS_V3 if write else READ_FILE|READ_DIR|EXECUTE) if directory else (
                        READ_FILE|EXECUTE|(WRITE_FILE|TRUNCATE if write else 0))
                    rule = PathRule(permissions,handle)
                    _check(_LIBC.syscall(445,descriptor,1,ctypes.byref(rule),0),'Landlock path rule')
                finally:
                    os.close(handle)
        _check(_LIBC.prctl(38,1,0,0,0),'PR_SET_NO_NEW_PRIVS')
        _check(_LIBC.syscall(446,descriptor,0),'Landlock restrict_self')
    finally:
        os.close(descriptor)


def restrict_syscalls():
    library = ctypes.CDLL('libseccomp.so.2',use_errno=True)
    library.seccomp_init.argtypes=[ctypes.c_uint32];library.seccomp_init.restype=ctypes.c_void_p
    library.seccomp_syscall_resolve_name.argtypes=[ctypes.c_char_p];library.seccomp_syscall_resolve_name.restype=ctypes.c_int
    library.seccomp_rule_add_array.argtypes=[ctypes.c_void_p,ctypes.c_uint32,ctypes.c_int,ctypes.c_uint,ctypes.POINTER(ArgCompare)]
    library.seccomp_load.argtypes=[ctypes.c_void_p];library.seccomp_release.argtypes=[ctypes.c_void_p]
    context=library.seccomp_init(0x7fff0000)  # SCMP_ACT_ALLOW
    if not context:
        raise RuntimeError('Cannot initialize seccomp')
    def deny(name,code=errno.EACCES,comparison=None):
        number=library.seccomp_syscall_resolve_name(name.encode())
        if number<0:
            return  # This syscall does not exist on this architecture.
        action=0x00050000|code  # SCMP_ACT_ERRNO
        count=0 if comparison is None else 1
        pointer=None if comparison is None else ctypes.pointer(comparison)
        result=library.seccomp_rule_add_array(context,action,number,count,pointer)
        if result<0:
            raise RuntimeError(f'Cannot restrict syscall {name}: {result}')
    try:
        for name in ['socket','socketpair','connect','bind','listen','accept','accept4',
            'sendto','sendmsg','sendmmsg','recvfrom','recvmsg','recvmmsg',
            'ptrace','process_vm_readv','process_vm_writev','pidfd_getfd',
            'shmget','shmat','shmctl','shmdt','msgget','msgsnd','msgrcv','msgctl',
            'semget','semop','semtimedop','semctl',
            'bpf','perf_event_open','keyctl','add_key','request_key',
            'mount','umount2','pivot_root','chroot','unshare','setns',
            'open_tree','move_mount','fsopen','fsconfig','fsmount','fspick','mount_setattr',
            'name_to_handle_at','open_by_handle_at','io_uring_setup',
            'chmod','fchmod','fchmodat','fchmodat2','chown','lchown','fchown','fchownat',
            'setxattr','lsetxattr','fsetxattr','removexattr','lremovexattr','fremovexattr',
            'mknod','mknodat','setuid','setgid','setreuid','setregid','setresuid','setresgid','setgroups']:
            deny(name)
        deny('clone3',errno.ENOSYS)  # libc may fall back to ordinary clone for threads.
        for flag in [0x00800000,0x00020000,0x02000000,0x04000000,0x08000000,
                     0x10000000,0x20000000,0x40000000,0x00000080]:
            deny('clone',comparison=ArgCompare(0,7,flag,flag))
        _check(_LIBC.prctl(38,1,0,0,0),'PR_SET_NO_NEW_PRIVS')
        result=library.seccomp_load(context)
        if result<0:
            raise RuntimeError(f'Cannot load seccomp: {result}')
        _deny_new_syscall_numbers()
    finally:
        library.seccomp_release(context)


def close_inherited_descriptors(keep=(0,1,2)):
    for name in os.listdir('/proc/self/fd'):
        if name.isdigit() and int(name) not in keep:
            try:
                os.close(int(name))
            except OSError:
                pass
