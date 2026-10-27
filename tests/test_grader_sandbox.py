import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

from grader.sandbox import landlock_abi


def test_real_child_cannot_read_outside_paths_weaken_modes_or_open_network(tmp_path):
    assert landlock_abi()>=3
    allowed=tmp_path/'allowed';allowed.mkdir()
    (allowed/'read.txt').write_text('permitted')
    secret=tmp_path/'outside.txt';secret.write_text('synthetic private canary')
    scratch=tmp_path/'scratch';scratch.mkdir()
    script=r'''
import ctypes,errno,json,os,socket,sys
from pathlib import Path
from grader.sandbox import restrict_filesystem,restrict_syscalls,close_inherited_descriptors
spec=json.loads(sys.argv[1])
close_inherited_descriptors()
restrict_syscalls()
restrict_filesystem(read_only=spec['read_only'],writable=[spec['scratch']])
assert Path(spec['allowed']).read_text()=='permitted'
denied=[]
for name,operation in [
 ('private_read',lambda:Path(spec['secret']).read_bytes()),
 ('outside_write',lambda:Path(spec['secret']).write_text('bad')),
 ('readonly_write',lambda:Path(spec['allowed']).write_text('bad')),
 ('chmod',lambda:os.chmod(spec['secret'],0o777)),
 ('acl',lambda:os.setxattr(spec['secret'],'user.synthetic',b'forbidden')),
 ('socket',lambda:socket.socket()),
]:
 try:operation()
 except PermissionError:denied.append(name)
 else:raise AssertionError(name+' unexpectedly allowed')
Path(spec['scratch'],'output.txt').write_text('permitted output')
libc=ctypes.CDLL(None,use_errno=True)
assert libc.syscall(452,-100,ctypes.c_char_p(spec['secret'].encode()),0o777,0)==-1
assert ctypes.get_errno()==errno.EACCES
assert libc.syscall(463)==-1 and ctypes.get_errno()==errno.EACCES
print(json.dumps({'denied':denied,'new_privileges':Path('/proc/self/status').read_text().split('NoNewPrivs:')[1].split()[0]}))
'''
    runtime=Path(sys.executable).resolve().parent.parent
    spec={'read_only':['/usr',str(runtime),str(Path(sys.prefix)),str(allowed),'/proc/self/status'],
          'scratch':str(scratch),'allowed':str(allowed/'read.txt'),'secret':str(secret)}
    completed=subprocess.run([sys.executable,'-c',script,json.dumps(spec)],text=True,capture_output=True,check=True)
    report=json.loads(completed.stdout)
    assert set(report['denied'])=={'private_read','outside_write','readonly_write','chmod','acl','socket'}
    assert report['new_privileges']=='1'
    assert secret.read_text()=='synthetic private canary'
    assert (scratch/'output.txt').read_text()=='permitted output'


def test_auditor_observes_denied_open_even_if_policy_catches_error(tmp_path):
    secret=tmp_path/'outside';secret.write_text('synthetic hidden')
    allowed=tmp_path/'allowed';allowed.mkdir()
    trace=tmp_path/'opens.trace'
    code=r'''
import json,sys
from pathlib import Path
from grader.sandbox import restrict_filesystem,restrict_syscalls
allowed,secret=sys.argv[1:]
restrict_syscalls()
restrict_filesystem(read_only=[allowed],writable=[])
try:open(secret,'rb')
except PermissionError:pass
print('finished')
'''
    completed=subprocess.run(['strace','-f','-qq','-s','4096','-e','trace=%file',
        '-o',str(trace),sys.executable,'-c',code,str(allowed),str(secret)],text=True,capture_output=True,check=True)
    assert 'finished' in completed.stdout
    entries=[line for line in trace.read_text().splitlines() if str(secret) in line]
    assert entries and any('EACCES' in line for line in entries)


def test_real_torch_cuda_and_parquet_work_inside_restrictions(tmp_path):
    script=r'''
import glob,json,os,sys
from pathlib import Path
import pandas as pd
import torch
from grader.sandbox import restrict_filesystem,restrict_syscalls
scratch=Path(sys.argv[1])
runtime=Path(sys.executable).resolve().parent.parent
read=[Path('/usr'),runtime,Path(sys.prefix),Path('/sys'),Path('/proc/driver/nvidia')]
read += [Path(p) for p in ['/proc/cpuinfo','/proc/meminfo','/proc/self/maps','/proc/self/status',
 '/proc/self/stat','/proc/self/statm','/etc/ld.so.cache','/etc/localtime','/etc/nsswitch.conf','/etc/passwd','/etc/group',
 '/dev/random','/dev/urandom'] if Path(p).exists()]
write=[scratch,Path('/dev/null')]+[Path(p) for p in ['/dev/nvidia0','/dev/nvidiactl','/dev/nvidia-uvm','/dev/nvidia-uvm-tools'] if Path(p).exists()]
restrict_syscalls()
restrict_filesystem(read_only=[path for path in read if path.exists()],writable=write)
device='cuda' if torch.cuda.is_available() else 'cpu'
x=torch.ones((8,8),dtype=torch.float64,device=device,requires_grad=True)
(x.square().sum()).backward()
assert torch.equal(x.grad,torch.full_like(x,2))
frame=pd.DataFrame({'value':[1.,2.]})
frame.to_parquet(scratch/'data.parquet',index=False)
pd.testing.assert_frame_equal(pd.read_parquet(scratch/'data.parquet'),frame)
print(json.dumps({'device':device,'gradient':True,'parquet':True}))
'''
    environment=dict(os.environ,TMPDIR=str(tmp_path),XDG_CACHE_HOME=str(tmp_path),
                     OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',
                     PYTHONDONTWRITEBYTECODE='1')
    trace=tmp_path/'cuda.trace'
    completed=subprocess.run(['strace','-f','-qq','-s','4096','-e','trace=%file',
        '-o',str(trace),sys.executable,'-c',script,str(tmp_path)],env=environment,
        text=True,capture_output=True)
    assert completed.returncode==0,completed.stderr
    report=json.loads(completed.stdout)
    assert report['gradient'] and report['parquet']
