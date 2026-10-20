#!/usr/bin/env python3
"""Install wheel/sdist independently and run the entire extracted sdist suite.

The caller provides built artifacts and an empty workspace. On Linux, execute
under xvfb-run. --wheelhouse allows offline installation with cached wheels.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tarfile
import venv
import zipfile

ROOT = Path(__file__).resolve().parents[1]


def run(command, cwd, env):
    subprocess.run([str(part) for part in command], cwd=cwd, env=env, check=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--artifacts', type=Path, required=True)
    parser.add_argument('--work', type=Path, required=True)
    parser.add_argument('--wheelhouse', type=Path)
    args = parser.parse_args()
    artifacts = args.artifacts.resolve()
    work = args.work.resolve()
    if work.is_relative_to(ROOT):
        parser.error('--work must be outside the source tree.')
    work.mkdir(parents=True, exist_ok=False)
    env = os.environ.copy()
    env.pop('PYTHONPATH', None)
    env.update(PYTHONDONTWRITEBYTECODE='1', MPLBACKEND='Agg',
               MPLCONFIGDIR=str(work / 'mpl'), TMPDIR=str(work / 'tmp'),
               TEMP=str(work / 'tmp'), TMP=str(work / 'tmp'))
    (work / 'tmp').mkdir()
    options = ['--no-index', '--find-links', str(args.wheelhouse.resolve())] if args.wheelhouse else []
    wheel, = artifacts.glob('*.whl')
    sdist, = artifacts.glob('*.tar.gz')
    for kind, artifact in (('wheel', wheel), ('sdist', sdist)):
        prefix = work / kind
        venv.EnvBuilder(with_pip=True).create(prefix)
        python = prefix / ('Scripts/python.exe' if os.name == 'nt' else 'bin/python')
        run([python, '-m', 'pip', 'install', *options, artifact], work, env)
        outside = work / (kind + '-outside')
        outside.mkdir()
        run([python, '-I', ROOT / 'scripts/verify_installation.py'], outside, env)
    source = work / 'source'
    source.mkdir()
    with tarfile.open(sdist) as bundle:
        # Build output is locally generated; reject paths that could escape extraction.
        for item in bundle.getmembers():
            if not (source / item.name).resolve().is_relative_to(source) or item.issym() or item.islnk():
                raise ValueError('Unsafe sdist member')
        bundle.extractall(source)
    extracted, = [path for path in source.iterdir() if path.is_dir()]
    with zipfile.ZipFile(wheel) as bundle:
        wheel_package = {name: bundle.read(name) for name in bundle.namelist()
                         if name.startswith('riima/') and not name.endswith('/')}
    source_package = {path.relative_to(ROOT).as_posix(): path.read_bytes()
                      for path in (ROOT / 'riima').rglob('*')
                      if path.is_file() and path.suffix in ('.py', '.html', '.png', '.ico')
                      and '__pycache__' not in path.parts}
    assert wheel_package == source_package, 'Wheel package differs from the candidate sources'
    for name, content in source_package.items():
        assert (extracted / name).read_bytes() == content, f'Sdist differs: {name}'
    python = work / 'sdist' / ('Scripts/python.exe' if os.name == 'nt' else 'bin/python')
    run([python, '-m', 'pip', 'install', *options, '-r', extracted / 'requirements/dev.txt'], work, env)
    run([python, '-m', 'pytest', 'tests', '-q', '-p', 'no:cacheprovider',
         '--basetemp', work / 'pytest'], extracted, env)
    run([python, extracted / 'scripts/build_share.py', '--output', work / 'share.zip'], work, env)
    with zipfile.ZipFile(work / 'share.zip') as bundle:
        manifest = json.loads(bundle.read('RIIMA/SHARE_MANIFEST.json'))
        assert manifest['license'] == 'BSD-3-Clause'
        for entry in manifest['files']:
            content = bundle.read('RIIMA/' + entry['path'])
            assert len(content) == entry['bytes']
            assert hashlib.sha256(content).hexdigest() == entry['sha256']
            if entry['path'].startswith('riima/'):
                assert content == source_package[entry['path']]
        assert not any(part in name.split('/') for name in bundle.namelist()
                       for part in ('imagens', '.git', '.dev', 'reproducibility', '__pycache__'))
    print('Wheel, sdist, extracted tests and shared ZIP verified.')


if __name__ == '__main__':
    main()
