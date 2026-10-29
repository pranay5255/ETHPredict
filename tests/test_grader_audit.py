from grader.audit import audit_trace


def audit(content,tmp_path):
    return audit_trace(content,server_pid=1,worker_pid=2,cwd=tmp_path,
                       read_only=['/usr'],writable=[str(tmp_path)])


def test_pointer_only_blinded_trace_or_missing_confinement_fails_closed(tmp_path):
    trace='2 landlock_restrict_self(4, 0) = 0\n2 openat(AT_FDCWD, 0x123456, O_RDONLY) = 4\n2 +++ exited with 0 +++\n'
    result=audit(trace,tmp_path)
    assert not result.passed and result.violations
    assert not audit('2 openat(AT_FDCWD, "/usr/file", O_RDONLY) = 4\n2 +++ exited with 0 +++\n',tmp_path).passed


def test_descriptor_relative_escapes_and_denied_temporary_links_are_audited(tmp_path):
    trace=f'''2 landlock_restrict_self(4, 0) = 0
2 openat(3</private>, "secret", O_RDONLY) = -1 EACCES (Permission denied)
2 openat(AT_FDCWD, "{tmp_path}/removed-symlink", O_RDONLY) = -1 EACCES (Permission denied)
2 +++ exited with 0 +++
'''
    result=audit(trace,tmp_path)
    assert not result.passed
    assert any('secret' in item for item in result.violations)
    assert any('removed-symlink' in item for item in result.violations)


def test_interleaved_resumed_operations_preserve_the_path_and_worker_exit(tmp_path):
    trace=f'''2 landlock_restrict_self(4, 0) = 0
2 openat(AT_FDCWD<{tmp_path}>, "output", O_WRONLY|O_CREAT, 0600 <unfinished ...>
3 openat(AT_FDCWD<{tmp_path}>, "/usr/lib/runtime", O_RDONLY) = 5
2 <... openat resumed>) = 4
2 +++ exited with 0 +++
'''
    result=audit(trace,tmp_path)
    assert result.passed and result.opened==2
