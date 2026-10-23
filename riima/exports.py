"""Staged publication with rollback for related local export files.

An operation publishes only its generated files, preserving unrelated output files.
Recoverable exceptions restore the previous generation. A failed rollback retains
backups and a journal for explicit recovery; pending destinations reject new writes.
Readers are not isolated during publication. Power-loss durability is not promised.
"""
from contextlib import contextmanager
from contextvars import ContextVar
from functools import wraps
import hashlib
import json
import os
from pathlib import Path
import shutil


_current = ContextVar('riima_export', default=None)


class ExportRecoveryError(OSError):
    """Publication or cleanup requires recovery from the retained journal."""

    def __init__(self, journal, message, *, original=None, recovery=None):
        self.journal = Path(journal)
        self.original_error, self.recovery_error = original, recovery
        super().__init__(f'{message} Recuperação pendente: {self.journal}')


def _work_directory(root):
    return root / '.riima-export'


def _retired_directory(root):
    return root / '.riima-export-cleanup'


def _validate_area(area, root):
    if area.exists() and (area.is_symlink() or area.resolve() != area or area.parent != root):
        raise OSError(f'Diretório de recuperação incompatível: {area}')


def _cleanup_retired(root):
    area = _retired_directory(root)
    _validate_area(area, root)
    if area.exists():
        allowed = {'owner.tmp', 'journal.tmp', 'owner.json', 'journal.json'}
        if any(p.name not in allowed or not p.is_file() or p.is_symlink() for p in area.iterdir()):
            raise OSError(f'Conteúdo de recuperação incompatível: {area}')
        for child in area.iterdir():
            child.unlink()
        area.rmdir()


def _digest(path):
    value = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            value.update(block)
    return value.hexdigest()


def _sync(path):
    with path.open('r+b') as stream:
        os.fsync(stream.fileno())


def _write_journal(path, data):
    temporary = path.with_suffix('.tmp')
    with temporary.open('w', encoding='utf-8') as stream:
        json.dump(data, stream, ensure_ascii=False, indent=2)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def _cleanup(roots):
    for root in roots:
        work = _work_directory(root)
        _validate_area(work, root)
        allowed = {'new', 'backup', 'restore', 'owner.tmp', 'journal.tmp', 'owner.json', 'journal.json'}
        if work.exists() and any(p.name not in allowed for p in work.iterdir()):
            raise OSError(f'Conteúdo de recuperação incompatível: {work}')
    # Retirement marks cleanup-only state, after every destination is settled.
    for root in roots:
        work = _work_directory(root)
        for name in ('new', 'backup', 'restore'):
            folder = work / name
            if folder.exists():
                if folder.is_symlink():
                    raise OSError(f'Diretório de recuperação incompatível: {folder}')
                shutil.rmtree(folder)
    for root in reversed(roots):
        work = _work_directory(root)
        if work.exists():
            os.replace(work, _retired_directory(root))
    for root in reversed(roots):
        _cleanup_retired(root)


def _entry_path(roots, entry):
    relative = Path(entry['relative'])
    if (relative.is_absolute() or relative.drive or '..' in relative.parts or not relative.parts
            or type(entry['root']) is not int or not 0 <= entry['root'] < len(roots)):
        raise ValueError('Caminho inválido no registro de exportação.')
    root = roots[entry['root']]
    target = root / relative
    if not target.parent.resolve().is_relative_to(root):
        raise ValueError('Destino externo ao diretório de exportação.')
    return target, _work_directory(root) / 'backup' / relative


def _restore(roots, entries):
    for root in roots:
        _validate_area(_work_directory(root), root)
    errors = []
    for entry in reversed(entries):
        target, backup = _entry_path(roots, entry)
        try:
            work = _work_directory(roots[entry['root']])
            if backup.is_symlink() or not backup.parent.resolve().is_relative_to(work):
                raise OSError(f'Backup externo ao diretório de recuperação: {backup}')
            if entry['existed']:
                if not backup.is_file() or _digest(backup) != entry['sha256']:
                    raise OSError(f'Backup ausente ou inválido: {backup}')
                if target.is_file() and not target.is_symlink() and _digest(target) == entry['sha256']:
                    continue
                temporary = _work_directory(roots[entry['root']]) / 'restore' / entry['relative']
                if temporary.is_symlink() or not temporary.parent.resolve().is_relative_to(work):
                    raise OSError(f'Destino de recuperação incompatível: {temporary}')
                temporary.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(backup, temporary)
                _sync(temporary)
                os.replace(temporary, target)
            else:
                target.unlink(missing_ok=True)
        except OSError as exc:
            errors.append(exc)
    if errors:
        raise errors[0]


def recover_export(destination):
    """Restore a pending operation after its writer has stopped.

    Pass any registered destination directory. Committed operations only need
    cleanup; interrupted publication restores all recorded previous files.
    Recovery must not run concurrently with an active writer.
    """
    root = Path(destination).resolve()
    work = _work_directory(root)
    _validate_area(work, root)
    _cleanup_retired(root)
    if not work.exists():
        return
    if not (work / 'owner.json').exists():
        if any((work / name).exists() for name in ('journal.json', 'backup', 'restore')):
            raise ValueError('Identificação de recuperação ausente.')
        _cleanup([root])
        return
    owner = json.loads((work / 'owner.json').read_text(encoding='utf-8'))
    journal = Path(owner['journal'])
    if journal.name != 'journal.json' or journal.parent.name != '.riima-export':
        raise ValueError('Registro de recuperação inválido.')
    primary = journal.parent.parent
    _validate_area(journal.parent, primary)
    retired_journal = _retired_directory(primary) / 'journal.json'
    if not journal.exists() and retired_journal.exists():
        journal = retired_journal
    if not journal.exists():
        # Publication is impossible before the leader journal is created.
        if primary != root or any((work / name).exists() for name in ('backup', 'restore')):
            raise ValueError('Registro de recuperação ausente.')
        _cleanup([root])
        return
    data = json.loads(journal.read_text(encoding='utf-8'))
    roots = [Path(value) for value in data['roots']]
    if root not in roots and data.get('phase') == 'preparing' and not any(
            (work / name).exists() for name in ('backup', 'restore')):
        roots.append(root)
    if (data.get('version') != 1 or root not in roots or not roots
            or any(not p.is_absolute() or p.resolve() != p for p in roots)
            or primary != roots[0]
            or data.get('phase') not in ('preparing', 'publishing', 'committed', 'rolled_back')):
        raise ValueError('Registro de recuperação inválido.')
    for area in roots:
        _validate_area(_work_directory(area), area)
        if not _work_directory(area).exists() and data['phase'] != 'publishing':
            continue
        marker = json.loads((_work_directory(area) / 'owner.json').read_text(encoding='utf-8'))
        if marker != owner:
            raise ValueError('Diretórios de recuperação incompatíveis.')
    if data['phase'] == 'publishing':
        _restore(roots, data['entries'])
        data['phase'] = 'rolled_back'
        _write_journal(journal, data)
    _cleanup(roots)


class _Export:
    def __init__(self):
        self.roots = []
        self.entries = []
        self.failure = None
        self.phase = 'preparing'

    @property
    def journal(self):
        return _work_directory(self.roots[0]) / 'journal.json'

    def record(self):
        if self.roots:
            _write_journal(self.journal, dict(version=1, roots=[str(p) for p in self.roots],
                                             entries=self.entries, phase=self.phase))

    def directory(self, destination):
        root = Path(destination).resolve()
        for registered in self.roots:
            staging = _work_directory(registered) / 'new'
            if root.is_relative_to(staging):
                root.mkdir(parents=True, exist_ok=True)
                return root
        if any(part in ('.riima-export', '.riima-export-cleanup') for part in root.parts):
            raise ValueError('Diretório reservado ao controle da exportação.')
        for ancestor in (root, *root.parents):
            if ancestor not in self.roots and any(p.exists() for p in
                    (_work_directory(ancestor), _retired_directory(ancestor))):
                raise ExportRecoveryError(_work_directory(ancestor) / 'owner.json',
                                          'Outra exportação está ativa ou aguarda recuperação.')
        for registered in self.roots:
            staging = _work_directory(registered) / 'new'
            if root.is_relative_to(registered):
                root.mkdir(parents=True, exist_ok=True)
                if root.stat().st_dev != registered.stat().st_dev:
                    continue
                result = staging / root.relative_to(registered)
                result.mkdir(parents=True, exist_ok=True)
                return result
        root.mkdir(parents=True, exist_ok=True)
        work = _work_directory(root)
        try:
            work.mkdir()
        except FileExistsError as exc:
            raise ExportRecoveryError(work / 'owner.json',
                                      'Outra exportação está ativa ou aguarda recuperação.') from exc
        self.roots.append(root)
        (work / 'new').mkdir()
        _write_journal(work / 'owner.json', dict(journal=str(self.journal)))
        self.record()
        return work / 'new'

    def abort(self):
        if self.roots:
            try:
                _cleanup(self.roots)
            except OSError as exc:
                pending = next((area for root in self.roots for area in
                                (_work_directory(root), _retired_directory(root)) if area.exists()),
                               self.journal.parent)
                journal = pending / 'journal.json'
                raise ExportRecoveryError(journal if journal.exists() else pending,
                                          f'Falha na limpeza da exportação: {exc}', recovery=exc) from exc

    def publish(self):
        if not self.roots:
            return
        try:
            targets = set()
            for index, root in enumerate(self.roots):
                work = _work_directory(root)
                for source in sorted((work / 'new').rglob('*')):
                    if source.is_symlink():
                        raise OSError(f'Link não permitido na exportação: {source}')
                    if not source.is_file():
                        continue
                    if not source.stat().st_size:
                        raise OSError(f'Arquivo exportado vazio: {source.name}')
                    _sync(source)
                    relative = source.relative_to(work / 'new')
                    target = root / relative
                    if target in targets:
                        raise OSError(f'Destino duplicado na exportação: {target}')
                    targets.add(target)
                    if target.is_symlink() or (target.exists() and not target.is_file()):
                        raise OSError(f'Destino de exportação incompatível: {target}')
                    if not target.parent.resolve().is_relative_to(root):
                        raise OSError('Destino externo ao diretório de exportação.')
                    for ancestor in (target.parent, *target.parent.parents):
                        if ancestor not in self.roots and any(p.exists() for p in
                                (_work_directory(ancestor), _retired_directory(ancestor))):
                            raise ExportRecoveryError(_work_directory(ancestor) / 'owner.json',
                                                      'Outra exportação aguarda recuperação.')
                    entry = dict(root=index, relative=relative.as_posix(), existed=target.exists(), sha256=None)
                    if entry['existed']:
                        backup = work / 'backup' / relative
                        backup.parent.mkdir(parents=True, exist_ok=True)
                        shutil.copy2(target, backup)
                        _sync(backup)
                        entry['sha256'] = _digest(backup)
                    self.entries.append(entry)
            self.phase = 'publishing'
            self.record()
        except BaseException:
            self.abort()
            raise
        try:
            for entry in self.entries:
                target, _ = _entry_path(self.roots, entry)
                target.parent.mkdir(parents=True, exist_ok=True)
                source = _work_directory(self.roots[entry['root']]) / 'new' / entry['relative']
                os.replace(source, target)
            self.phase = 'committed'
            self.record()
        except BaseException as original:
            try:
                _restore(self.roots, self.entries)
                self.phase = 'rolled_back'
                self.record()
            except OSError as recovery:
                raise ExportRecoveryError(self.journal, 'Exportação falhou e o rollback não foi concluído.',
                                          original=original, recovery=recovery) from original
            self.abort()
            raise
        self.abort()


@contextmanager
def export_batch(*, publish=None):
    """Join exports and optionally dispatch outer publication through a caller.

    ``publish`` must synchronously run the supplied zero-argument callable for
    complete publication, including rollback. Nested batches join the active operation.
    """
    active = _current.get()
    if active is not None:
        try:
            yield active
        except BaseException as exc:
            active.failure = exc
            raise
        return
    operation = _Export()
    token = _current.set(operation)
    try:
        try:
            yield operation
            if operation.failure is not None:
                raise OSError('Uma etapa da exportação falhou.') from operation.failure
        except BaseException:
            operation.abort()
            raise
        if publish is None:
            operation.publish()
        else:
            started = False

            def complete():
                nonlocal started
                started = True
                operation.publish()

            try:
                publish(complete)
                if not started:
                    raise RuntimeError('A publicação da exportação não foi executada.')
            except BaseException:
                if not started:
                    operation.abort()
                raise
    finally:
        _current.reset(token)


def export_operation(function):
    @wraps(function)
    def wrapped(*args, **kwargs):
        with export_batch():
            return function(*args, **kwargs)
    return wrapped


def staged_directory(destination):
    operation = _current.get()
    if operation is None:
        raise RuntimeError('Não há exportação em preparação.')
    return operation.directory(destination)


def staged_file(destination):
    target = Path(destination)
    return staged_directory(target.parent) / target.name


def in_export():
    return _current.get() is not None
