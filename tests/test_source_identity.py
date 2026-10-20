"""Identify distributed bytes and real Git states without rewriting history."""
import importlib.util
from pathlib import Path
import subprocess

import pytest

ROOT = Path(__file__).resolve().parents[1]


def load_script(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / 'scripts' / f'{name}.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def git(root, *args):
    result = subprocess.run(['git', '-C', str(root), *args], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    return result.stdout.strip()


@pytest.fixture
def repository(tmp_path):
    root = tmp_path / 'repository'
    root.mkdir()
    git(root, 'init')
    git(root, 'config', 'user.name', 'Local test')
    git(root, 'config', 'user.email', 'local@example.invalid')
    (root / 'source.py').write_text('value = 1\n')
    git(root, 'add', 'source.py')
    git(root, 'commit', '-m', 'Local fixture')
    return root


def test_inventory_git_status_is_clean_dirty_or_unknown(repository, tmp_path):
    inventory = load_script('source_identity')
    assert hasattr(inventory, 'git_identification'), 'Git status must be measured, not hardcoded'
    clean = inventory.git_identification(repository)
    assert clean['git_head'] == git(repository, 'rev-parse', 'HEAD')
    assert clean['working_tree_modified'] is False
    assert clean['git_status'] == 'clean'
    (repository / 'source.py').write_text('value = 2\n')
    assert inventory.git_identification(repository)['working_tree_modified'] is True
    git(repository, 'checkout', '--', 'source.py')
    (repository / 'untracked.txt').write_text('new')
    assert inventory.git_identification(repository)['git_status'] == 'dirty'
    unrelated = tmp_path / 'outside'
    unrelated.mkdir()
    assert inventory.git_identification(unrelated) == {
        'git_head': None, 'working_tree_modified': None, 'git_status': 'unavailable'}
    # An extracted project nested under a checkout must not borrow that checkout's HEAD.
    nested = repository / 'extracted'
    nested.mkdir()
    assert inventory.git_identification(nested)['git_head'] is None


def test_inventory_git_missing_is_unknown(tmp_path, monkeypatch):
    inventory = load_script('source_identity')
    assert hasattr(inventory, 'git_identification'), 'Handle systems without Git'
    monkeypatch.setenv('PATH', '')
    assert inventory.git_identification(tmp_path)['working_tree_modified'] is None


def test_source_identity_changes_only_with_distributed_bytes(tmp_path):
    import shutil
    builder = load_script('build_share')
    assert hasattr(builder, 'source_manifest'), 'Provide content identification without a Git revision'
    first = builder.source_manifest(ROOT)
    second = builder.source_manifest(ROOT)
    assert first == second
    assert len(first['source_sha256']) == 64
    names = {entry['path'] for entry in first['files']}
    assert 'LICENSE' in names and 'THIRD_PARTY.md' in names
    assert not any('reproducibility/' in name for name in names)
    assert not any(name.startswith(('.dev/', '.git/', 'imagens/')) for name in names)
    assert not any('C:' in entry['path'] for entry in first['files'])
    copied = tmp_path / 'source'
    for path in builder.distribution_files(ROOT):
        target = copied / path.relative_to(ROOT)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(path, target)
    assert builder.source_manifest(copied) == first
    (copied / '.private').write_text('Not an input')
    assert builder.source_manifest(copied) == first
    (copied / 'README.md').write_text('Changed public content')
    assert builder.source_manifest(copied)['source_sha256'] != first['source_sha256']


def test_shared_archive_declares_authorized_license_and_source_identity(tmp_path):
    import json
    import zipfile
    builder = load_script('build_share')
    output = builder.build_archive(ROOT, tmp_path / 'share.zip')
    duplicate = builder.build_archive(ROOT, tmp_path / 'duplicate.zip')
    assert output.read_bytes() == duplicate.read_bytes()
    with zipfile.ZipFile(output) as bundle:
        manifest = json.loads(bundle.read('RIIMA/SHARE_MANIFEST.json'))
        assert manifest['license'] == 'BSD-3-Clause'
        assert len(manifest['source_sha256']) == 64
        import hashlib
        canonical = json.dumps(manifest['files'], ensure_ascii=False, sort_keys=True,
                               separators=(',', ':')).encode('utf-8')
        assert manifest['source_sha256'] == hashlib.sha256(canonical).hexdigest()
        assert 'RIIMA/LICENSE' in bundle.namelist()
        assert not any('/reproducibility/' in name for name in bundle.namelist())
        for name in bundle.namelist():
            if name.startswith('RIIMA/riima/'):
                assert bundle.read(name) == (ROOT / name.removeprefix('RIIMA/')).read_bytes()


def test_distribution_allowlist_excludes_private_artifacts(tmp_path):
    import shutil
    builder = load_script('build_share')
    root = tmp_path / 'copy'
    # Copy only declared inputs; never local data or environments.
    for path in builder.distribution_files(ROOT):
        target = root / path.relative_to(ROOT)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(path, target)
    for name in ('imagens/private.jpg', 'papers/reproducibility/inventory.json',
                 'riima/private.npy', 'docs/private.json', '.env', 'tests/private.riima',
                 'riima/private.py', 'docs/private.md', 'tests/private.py',
                 'scripts/private.py', '.github/workflows/private.yml',
                 'riima/assets/branding/private.png'):
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b'PRIVATE')
    names = {path.relative_to(root).as_posix() for path in builder.distribution_files(root)}
    assert 'papers/reproducibility/inventory.json' not in names
    assert not any(name.endswith(('.jpg', '.npy', '.riima', '.env')) for name in names)
    assert not any('private' in name for name in names)


def test_source_identity_cli_prints_manifest_without_writing_files(tmp_path):
    import json
    import sys
    marker = tmp_path / 'unchanged.txt'
    marker.write_text('Preserve')
    result = subprocess.run([sys.executable, '-B', str(ROOT / 'scripts/source_identity.py')],
                            cwd=tmp_path, capture_output=True, text=True, timeout=30)
    assert result.returncode == 0, result.stderr
    manifest = json.loads(result.stdout)
    assert manifest['schema'] == 'RIIMA-source-identity-1'
    assert len(manifest['source_sha256']) == 64
    assert manifest['git_status'] in ('clean', 'dirty', 'unavailable')
    assert list(tmp_path.iterdir()) == [marker]
    assert marker.read_text() == 'Preserve'
