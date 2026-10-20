"""Publication, rollback, retained recovery data and retry contracts."""
from contextvars import Context
import json
from pathlib import Path
import sys
import subprocess

import pytest

from riima import exports
from test_export_integrity import file_bytes


def write_generation(root):
    with exports.export_batch():
        staging = exports.staged_directory(root)
        (staging / 'nested').mkdir(exist_ok=True)
        for name, content in [('a.npy', b'new a'), ('b.csv', b'new b'), ('nested/c.json', b'new c')]:
            (staging / name).write_bytes(content)


def old_generation(root, existing=True):
    root.mkdir(parents=True)
    if existing:
        (root / 'nested').mkdir()
        for name in ('a.npy', 'b.csv', 'nested/c.json'):
            (root / name).write_bytes(('previous ' + name).encode())
        (root / 'unrelated.txt').write_bytes(b'preserve the user file')
    return file_bytes(root)


@pytest.mark.parametrize('existing', [False, True])
@pytest.mark.parametrize('failure_at', [1, 2, 3])
def test_publication_failure_restores_all_previous_bytes(tmp_path, monkeypatch, existing, failure_at):
    root = tmp_path / 'Saída científica ç'
    before = old_generation(root, existing)
    original = exports.os.replace
    calls = []
    def fail(source, target):
        if Path(source).is_relative_to(root / '.riima-export/new'):
            calls.append(Path(target))
            if len(calls) == failure_at:
                raise PermissionError('simulated locked destination')
        return original(source, target)
    with monkeypatch.context() as patch:
        patch.setattr(exports.os, 'replace', fail)
        with pytest.raises(PermissionError, match='simulated'):
            write_generation(root)
    assert len(calls) == failure_at
    assert file_bytes(root) == before
    assert not (root / '.riima-export').exists()
    write_generation(root)
    assert (root / 'a.npy').read_bytes() == b'new a'
    assert (root / 'nested/c.json').read_bytes() == b'new c'
    if existing:
        assert (root / 'unrelated.txt').read_bytes() == b'preserve the user file'


def test_failed_rollback_retains_backups_and_blocks_writes_until_recovery(tmp_path, monkeypatch):
    root = tmp_path / 'recovery'
    before = old_generation(root)
    original = exports.os.replace
    def fail(source, target):
        source, target = Path(source), Path(target)
        if source.is_relative_to(root / '.riima-export/new') and target.name == 'b.csv':
            raise OSError('publication failed')
        if source.is_relative_to(root / '.riima-export/restore'):
            raise PermissionError('rollback blocked')
        return original(source, target)
    with monkeypatch.context() as patch:
        patch.setattr(exports.os, 'replace', fail)
        with pytest.raises(exports.ExportRecoveryError) as caught:
            write_generation(root)
    error = caught.value
    assert 'rollback' in str(error)
    assert str(error.original_error) == 'publication failed'
    assert str(error.recovery_error) == 'rollback blocked'
    assert error.journal.is_file()
    assert json.loads(error.journal.read_text())['phase'] == 'publishing'
    assert (root / '.riima-export/backup/a.npy').read_bytes() == before['a.npy']
    assert (root / 'a.npy').read_bytes() == b'new a'
    with pytest.raises(exports.ExportRecoveryError, match='recupera'):
        write_generation(root)
    exports.recover_export(root)
    assert file_bytes(root) == before
    exports.recover_export(root)
    write_generation(root)
    assert not (root / '.riima-export').exists()


def test_interrupted_publication_is_recovered_from_journal(tmp_path, monkeypatch):
    root = tmp_path / 'interrupted'
    before = old_generation(root)
    original = exports.os.replace
    def fail(source, target):
        source, target = Path(source), Path(target)
        if source.is_relative_to(root / '.riima-export/new') and target.name == 'b.csv':
            raise KeyboardInterrupt('interrupted publication')
        if source.is_relative_to(root / '.riima-export/restore'):
            raise OSError('recovery temporarily unavailable')
        return original(source, target)
    with monkeypatch.context() as patch:
        patch.setattr(exports.os, 'replace', fail)
        with pytest.raises(exports.ExportRecoveryError):
            write_generation(root)
    exports.recover_export(root)
    assert file_bytes(root) == before


def test_backup_write_failure_does_not_touch_outputs(tmp_path, monkeypatch):
    root = tmp_path / 'backup'
    before = old_generation(root)
    def fail(source, target):
        Path(target).write_bytes(b'partial backup')
        raise OSError('backup disk full')
    with monkeypatch.context() as patch:
        patch.setattr(exports.shutil, 'copy2', fail)
        with pytest.raises(OSError, match='disk full'):
            write_generation(root)
    assert file_bytes(root) == before
    write_generation(root)


def test_failed_commit_record_rolls_back_even_after_all_files_were_replaced(tmp_path, monkeypatch):
    root = tmp_path / 'journal'
    before = old_generation(root)
    original = exports._write_journal
    def fail(path, data):
        if data.get('phase') == 'committed':
            raise OSError('commit record failed')
        return original(path, data)
    with monkeypatch.context() as patch:
        patch.setattr(exports, '_write_journal', fail)
        with pytest.raises(OSError, match='commit record'):
            write_generation(root)
    assert file_bytes(root) == before


def test_cleanup_failure_preserves_committed_data_and_recoverable_journal(tmp_path, monkeypatch):
    root = tmp_path / 'cleanup'
    old_generation(root)
    original = exports.shutil.rmtree
    def fail(path, *args, **kwargs):
        if Path(path).name == 'backup':
            raise PermissionError('cleanup blocked')
        return original(path, *args, **kwargs)
    with monkeypatch.context() as patch:
        patch.setattr(exports.shutil, 'rmtree', fail)
        with pytest.raises(exports.ExportRecoveryError) as caught:
            write_generation(root)
    assert json.loads(caught.value.journal.read_text())['phase'] == 'committed'
    assert (root / 'a.npy').read_bytes() == b'new a'
    exports.recover_export(root)
    assert (root / 'a.npy').read_bytes() == b'new a'
    assert not (root / '.riima-export').exists()


def test_nested_failure_caught_by_caller_cannot_publish_partial_results(tmp_path):
    root = tmp_path / 'nested'
    before = old_generation(root)
    with pytest.raises(OSError, match='etapa'):
        with exports.export_batch():
            staging = exports.staged_directory(root)
            (staging / 'a.npy').write_bytes(b'partial generation')
            try:
                with exports.export_batch():
                    raise OSError('nested failed')
            except OSError:
                pass
    assert file_bytes(root) == before


def test_overlapping_writer_is_rejected_without_removing_active_staging(tmp_path):
    root = tmp_path / 'overlap'
    old_generation(root)
    with exports.export_batch():
        staging = exports.staged_directory(root)
        (staging / 'a.npy').write_bytes(b'first writer')
        with pytest.raises(exports.ExportRecoveryError):
            Context().run(write_generation, root)
        assert (staging / 'a.npy').read_bytes() == b'first writer'
    assert (root / 'a.npy').read_bytes() == b'first writer'


def test_multiple_destinations_stage_on_their_own_filesystems_and_rollback_together(tmp_path, monkeypatch):
    first, second = tmp_path / 'data', tmp_path / 'report'
    old_generation(first)
    old_generation(second)
    before = file_bytes(first), file_bytes(second)
    original = exports.os.replace
    def fail(source, target):
        source, target = Path(source), Path(target)
        assert source.stat().st_dev == target.parent.stat().st_dev
        if target == second / 'b.csv':
            raise OSError('second destination blocked')
        return original(source, target)
    with monkeypatch.context() as patch:
        patch.setattr(exports.os, 'replace', fail)
        with pytest.raises(OSError, match='second destination'):
            with exports.export_batch():
                (exports.staged_directory(first) / 'a.npy').write_bytes(b'new first')
                (exports.staged_directory(second) / 'b.csv').write_bytes(b'new second')
    assert (file_bytes(first), file_bytes(second)) == before


@pytest.mark.skipif(sys.platform != 'win32', reason='Native Windows sharing contract')
def test_windows_file_without_delete_sharing_is_not_truncated(tmp_path):
    import ctypes
    from ctypes import wintypes
    root = tmp_path / 'native-lock'
    before = old_generation(root)
    kernel = ctypes.WinDLL('kernel32', use_last_error=True)
    kernel.CreateFileW.argtypes = [wintypes.LPCWSTR, wintypes.DWORD, wintypes.DWORD, ctypes.c_void_p,
                                   wintypes.DWORD, wintypes.DWORD, wintypes.HANDLE]
    kernel.CreateFileW.restype = wintypes.HANDLE
    kernel.CloseHandle.argtypes = [wintypes.HANDLE]
    handle = kernel.CreateFileW(str(root / 'b.csv'), 0x80000000, 1, None, 3, 0x80, None)
    assert handle != ctypes.c_void_p(-1).value
    try:
        with pytest.raises(PermissionError):
            write_generation(root)
        assert file_bytes(root) == before
    finally:
        kernel.CloseHandle(handle)
    write_generation(root)


def test_parent_destination_can_be_registered_after_child(tmp_path):
    with exports.export_batch():
        (exports.staged_directory(tmp_path / 'out') / 'data.csv').write_bytes(b'data')
        exports.staged_file(tmp_path / 'report.html').write_bytes(b'report')
    assert (tmp_path / 'out/data.csv').read_bytes() == b'data'
    assert (tmp_path / 'report.html').read_bytes() == b'report'


@pytest.mark.parametrize('failure', ['journal.json', 'directory', 'unknown'])
def test_metadata_cleanup_can_be_retried(tmp_path, monkeypatch, failure):
    root = tmp_path / 'cleanup'
    old_generation(root)
    unlink, rmdir = Path.unlink, Path.rmdir
    def blocked_unlink(path, *args, **kwargs):
        if failure == 'journal.json' and path.name == failure:
            raise PermissionError('metadata locked')
        return unlink(path, *args, **kwargs)
    def blocked_rmdir(path, *args, **kwargs):
        if failure == 'directory' and path.name.startswith('.riima-export'):
            raise PermissionError('directory locked')
        return rmdir(path, *args, **kwargs)
    with monkeypatch.context() as patch:
        patch.setattr(Path, 'unlink', blocked_unlink)
        patch.setattr(Path, 'rmdir', blocked_rmdir)
        if failure == 'unknown':
            with pytest.raises(exports.ExportRecoveryError):
                with exports.export_batch():
                    stage = exports.staged_directory(root)
                    (stage / 'a.npy').write_bytes(b'new a')
                    (stage.parent / 'unknown').write_bytes(b'user data')
            (root / '.riima-export/unknown').unlink()
        else:
            with pytest.raises(exports.ExportRecoveryError):
                write_generation(root)
    exports.recover_export(root)
    assert (root / 'a.npy').read_bytes() == b'new a'
    assert not list(root.glob('.riima-export*'))
    write_generation(root)


@pytest.mark.parametrize('owner', [False, True])
def test_preparation_without_leader_journal_can_be_recovered(tmp_path, owner):
    root = tmp_path.resolve()
    work = root / '.riima-export'
    (work / 'new').mkdir(parents=True)
    if owner:
        (work / 'owner.json').write_text(json.dumps({'journal': str(work / 'journal.json')}))
    else:
        (work / 'owner.tmp').write_bytes(b'{')
    exports.recover_export(root)
    assert not work.exists()


def test_standalone_project_save_respects_pending_export(tmp_path):
    from riima import review
    from test_riima_review import prepared_dataset
    # Reserve the destination in another context, as a pending writer would.
    with exports.export_batch():
        exports.staged_directory(tmp_path)
        with pytest.raises(exports.ExportRecoveryError):
            Context().run(review.save_project, prepared_dataset(tmp_path), tmp_path / 'project.riima')


@pytest.mark.parametrize('interruption', ['before_owner', 'after_owner', 'publication'])
def test_process_exit_preserves_recoverable_files(tmp_path, interruption):
    root = tmp_path / 'process'
    before = old_generation(root)
    code = '''
import os
from pathlib import Path
import sys
from riima import exports
root, interruption = Path(sys.argv[1]), sys.argv[2]
record, replace = exports._write_journal, exports.os.replace
def stop_record(path, data):
    if path.name == 'owner.json' and interruption == 'before_owner':
        os._exit(23)
    record(path, data)
    if path.name == 'owner.json' and interruption == 'after_owner':
        os._exit(23)
def stop_replace(source, target):
    replace(source, target)
    if Path(target) == root / 'a.npy' and interruption == 'publication':
        os._exit(23)
exports._write_journal, exports.os.replace = stop_record, stop_replace
with exports.export_batch():
    stage = exports.staged_directory(root)
    (stage / 'a.npy').write_bytes(b'new a')
    (stage / 'b.csv').write_bytes(b'new b')
'''
    result = subprocess.run([sys.executable, '-c', code, str(root), interruption],
                            cwd=Path(__file__).resolve().parents[1], capture_output=True, timeout=30)
    assert result.returncode == 23, result.stderr.decode(errors='replace')
    with pytest.raises(exports.ExportRecoveryError):
        write_generation(root)
    exports.recover_export(root)
    assert file_bytes(root) == before
    write_generation(root)


def test_failed_recovery_can_be_repeated_without_losing_backups(tmp_path, monkeypatch):
    root = tmp_path / 'repeat'
    before = old_generation(root)
    original = exports.os.replace
    def blocked(source, target):
        if Path(target) == root / 'b.csv' or 'restore' in Path(source).parts:
            raise PermissionError('blocked')
        return original(source, target)
    with monkeypatch.context() as patch:
        patch.setattr(exports.os, 'replace', blocked)
        with pytest.raises(exports.ExportRecoveryError):
            write_generation(root)
        with pytest.raises(PermissionError):
            exports.recover_export(root)
        assert (root / '.riima-export/backup/a.npy').read_bytes() == before['a.npy']
    exports.recover_export(root)
    assert file_bytes(root) == before


def test_generated_file_cannot_overwrite_transaction_metadata(tmp_path):
    before = old_generation(tmp_path / 'out')
    root = tmp_path / 'out'
    with pytest.raises(ValueError, match='reservado'):
        with exports.export_batch():
            exports.staged_directory(root)
            exports.staged_file(root / '.riima-export/owner.json').write_bytes(b'invalid')
    assert file_bytes(root) == before

