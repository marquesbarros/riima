"""Exercise the package and the shared archive independently of local analyses."""
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import zipfile
import pytest

ROOT = Path(__file__).resolve().parents[1]


def test_example_generator_requires_an_explicit_destination(tmp_path):
    import shutil
    script = tmp_path / 'project/scripts/generate_example.py'
    script.parent.mkdir(parents=True)
    shutil.copyfile(ROOT / 'scripts/generate_example.py', script)
    result = subprocess.run([sys.executable, '-B', str(script)], cwd=tmp_path,
                            capture_output=True, text=True, timeout=30)
    assert result.returncode == 2
    assert '--output' in result.stderr
    assert not (script.parents[1] / 'examples').exists()


def test_package_entrypoint_and_bundled_resources():
    result = subprocess.run([sys.executable, '-B', '-m', 'riima', '--help'],
                            cwd=ROOT, capture_output=True, text=True, timeout=30)
    assert result.returncode == 0, result.stderr
    assert '--image' in result.stdout
    assert '--clusters' in result.stdout
    assert (ROOT / 'riima/assets/branding/riima_logo.png').is_file()
    assert (ROOT / 'riima/assets/branding/riima_icon.ico').is_file()
    assert (ROOT / 'riima/templates/descriptive_report.html').is_file()


def test_share_archive_excludes_local_data_and_runs_from_extracted_folder(tmp_path):
    spec = importlib.util.spec_from_file_location('build_share', ROOT / 'scripts/build_share.py')
    builder = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(builder)
    archive = builder.build_archive(ROOT, tmp_path / 'shared.zip')
    extracted = tmp_path / 'Cópia compartilhada - análise'
    with zipfile.ZipFile(archive) as bundle:
        names = bundle.namelist()
        assert names == sorted(names)
        assert 'RIIMA/riima/core.py' in names
        assert 'RIIMA/riima/assets/branding/riima_icon.ico' in names
        assert 'RIIMA/scripts/generate_example.py' in names
        assert 'RIIMA/PUBLIC_FILES.json' in names
        assert not any(part in name.split('/') for name in names
                       for part in ('.dev', '.git', '__pycache__', 'imagens', 'papers', 'examples'))
        assert not any(name.endswith(('.riima', '.pyc')) for name in names)
        bundle.extractall(extracted)
    workdir = extracted / 'RIIMA'
    manifest = json.loads((workdir / 'SHARE_MANIFEST.json').read_text(encoding='utf-8'))
    import hashlib
    for item in manifest['files']:
        content = (workdir / item['path']).read_bytes()
        assert len(content) == item['bytes']
        assert hashlib.sha256(content).hexdigest() == item['sha256']
    # The subprocess can use cached third-party packages, but never the source checkout.
    env = os.environ.copy()
    env['PYTHONPATH'] = os.pathsep.join(p for p in env.get('PYTHONPATH', '').split(os.pathsep)
                                      if p and Path(p).resolve() != ROOT)
    result = subprocess.run([sys.executable, '-B', '-m', 'riima', '--help'], cwd=workdir,
                            env=env, capture_output=True, text=True, timeout=30)
    assert result.returncode == 0, result.stderr
    result = subprocess.run([sys.executable, '-B', '-c',
                             'from riima import core; import riima; '
                             'assert core.__version__ == riima.__version__; '
                             'print(core.__file__)'], cwd=workdir, env=env,
                            capture_output=True, text=True, timeout=30)
    assert result.returncode == 0, result.stderr
    assert str(workdir / 'riima') in result.stdout


def test_share_builder_rejects_output_inside_source_package(tmp_path):
    spec = importlib.util.spec_from_file_location('build_share', ROOT / 'scripts/build_share.py')
    builder = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(builder)
    import pytest
    with pytest.raises(ValueError, match='dist|saída|output'):
        builder.build_archive(ROOT, ROOT / 'riima/unexpected.zip')


@pytest.mark.parametrize('filename', ['README.md', 'pyproject.toml', 'scripts/build_share.py',
                                     'requirements/dev.txt', 'docs/METHOD.md'])
def test_share_builder_cannot_replace_source_documents(tmp_path, filename):
    spec = importlib.util.spec_from_file_location('build_share', ROOT / 'scripts/build_share.py')
    builder = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(builder)
    source = tmp_path / 'project'
    files = ('README.md', 'requirements.txt', 'pyproject.toml', 'run_gui.bat', filename)
    for name in files:
        path = source / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b'Preserve this source file.\n')
    (source / 'riima').mkdir()
    (source / 'riima/__init__.py').write_text('__version__ = "1.0.0-rc6"\n')
    original = (source / filename).read_bytes()
    with pytest.raises(ValueError, match='zip|saída|output|dist'):
        builder.build_archive(source, source / filename)
    assert (source / filename).read_bytes() == original
