import os

import pytest

from grader.artifacts import capture_artifacts, verify_artifacts
from grader.data import DataContractError


def test_artifacts_cover_checkpoints_inputs_and_reports_can_be_rescored(tmp_path):
    (tmp_path/'worker').mkdir()
    # Arbitrary executable pickle-looking bytes are hashed, never interpreted.
    (tmp_path/'worker/state.pkl').write_bytes(b'opaque candidate checkpoint')
    (tmp_path/'positions.parquet').write_bytes(b'synthetic positions')
    reference=capture_artifacts(tmp_path)
    assert reference['files']==2
    (tmp_path/'grade_report.json').write_text('{"rescored":true}')
    value=verify_artifacts(tmp_path,reference)
    assert {row['path'] for row in value['files']}=={'worker/state.pkl','positions.parquet'}


@pytest.mark.parametrize('change',['modify','delete','extra','manifest'])
def test_artifact_verification_rejects_tampering_and_unmanifested_files(tmp_path,change):
    (tmp_path/'input').write_bytes(b'original')
    reference=capture_artifacts(tmp_path)
    if change=='modify': (tmp_path/'input').write_bytes(b'changed')
    if change=='delete': (tmp_path/'input').unlink()
    if change=='extra': (tmp_path/'extra').write_bytes(b'not recorded')
    if change=='manifest': (tmp_path/'artifact_manifest.json').write_text('{}')
    with pytest.raises(DataContractError): verify_artifacts(tmp_path,reference)


@pytest.mark.parametrize('kind',['symlink','directory_symlink','hardlink','fifo'])
def test_artifact_capture_never_follows_links_or_blocks_on_special_files(tmp_path,kind):
    outside=tmp_path.parent/f'{tmp_path.name}-outside'
    outside.mkdir();(outside/'canary').write_bytes(b'outside bytes must not be read')
    if kind=='symlink': (tmp_path/'link').symlink_to(outside/'canary')
    if kind=='directory_symlink': (tmp_path/'link').symlink_to(outside,target_is_directory=True)
    if kind=='hardlink': os.link(outside/'canary',tmp_path/'link')
    if kind=='fifo': os.mkfifo(tmp_path/'fifo')
    with pytest.raises((DataContractError,OSError)): capture_artifacts(tmp_path)
    assert not (tmp_path/'artifact_manifest.json').exists()


def test_artifact_capture_cannot_silently_omit_unreadable_directories(tmp_path):
    directory=tmp_path/'unreadable'; directory.mkdir(mode=0)
    try:
        with pytest.raises(PermissionError): capture_artifacts(tmp_path)
    finally:
        directory.chmod(0o700)
