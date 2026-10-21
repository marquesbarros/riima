#!/usr/bin/env python3
"""Build a reproducible source archive without local data or development caches."""
from __future__ import annotations

import argparse
import ast
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import tempfile
import zipfile

ROOT = Path(__file__).resolve().parents[1]


def distribution_files(root: Path) -> list[Path]:
    """Resolve only the public paths declared in PUBLIC_FILES.json."""
    root = root.resolve()
    manifest = json.loads((root / 'PUBLIC_FILES.json').read_text(encoding='utf-8'))
    if manifest.get('schema') != 'RIIMA-public-files-1' or not isinstance(manifest.get('files'), list):
        raise ValueError('Invalid public file manifest.')
    files = set()
    for entry in manifest['files']:
        name = entry['path']
        relative = PurePosixPath(name)
        if (not name or relative.is_absolute() or '..' in relative.parts
                or '\\' in name or ':' in name or relative.parts[0] == '.git'
                or not isinstance(entry.get('purpose'), str) or not entry['purpose'].strip()):
            raise ValueError(f'Invalid public path or purpose: {name}')
        path = root / name
        if path in files or not path.is_file() or not path.resolve().is_relative_to(root):
            raise ValueError(f'Duplicate, missing or external distribution file: {name}')
        files.add(path)
    if root / 'PUBLIC_FILES.json' not in files:
        raise ValueError('The public file manifest must include itself.')
    return sorted(files, key=lambda path: path.relative_to(root).as_posix())


def package_version(root: Path) -> str:
    """Read the literal version without importing numerical dependencies."""
    for statement in ast.parse((root / 'riima/__init__.py').read_text(encoding='utf-8')).body:
        if isinstance(statement, ast.Assign) and any(
                isinstance(target, ast.Name) and target.id == '__version__' for target in statement.targets):
            return ast.literal_eval(statement.value)
    raise ValueError('Package version is missing')


def content_manifest(contents: dict[str, bytes]) -> dict:
    """Hash a sorted UTF-8 JSON file list; exclude volatile Git and timestamp data."""
    files = [dict(path=name, bytes=len(data), sha256=hashlib.sha256(data).hexdigest())
             for name, data in sorted(contents.items())]
    canonical = json.dumps(files, ensure_ascii=False, sort_keys=True, separators=(',', ':'))
    return dict(source_sha256=hashlib.sha256(canonical.encode('utf-8')).hexdigest(), files=files)


def source_manifest(root: Path) -> dict:
    """Identify the exact selected source bytes without claiming a recoverable commit."""
    root = root.resolve()
    contents = {path.relative_to(root).as_posix(): path.read_bytes()
                for path in distribution_files(root)}
    return dict(schema='RIIMA-source-identity-1', version=package_version(root),
                **content_manifest(contents))


def build_archive(root: Path, output: Path) -> Path:
    root, output = root.resolve(), output.resolve()
    if output.suffix.lower() != '.zip':
        raise ValueError('A saída precisa ser um arquivo .zip.')
    if output.is_relative_to(root):
        raise ValueError('Escolha uma saída externa à árvore de fontes.')
    contents = {path.relative_to(root).as_posix(): path.read_bytes() for path in distribution_files(root)}
    manifest = {
        'schema': 'RIIMA-shared-source-1', 'version': package_version(root),
        'contents': 'Public source, documentation, tests and synthetic example generator.',
        'license': 'BSD-3-Clause',
        'license_scope': 'Software and bundled branding/template; see THIRD_PARTY.md for excluded materials.',
        **content_manifest(contents),
    }
    contents['SHARE_MANIFEST.json'] = (json.dumps(manifest, ensure_ascii=False, indent=2) + '\n').encode('utf-8')
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(dir=output.parent, prefix='.riima-share-', suffix='.zip', delete=False) as handle:
            temporary = Path(handle.name)
        with zipfile.ZipFile(temporary, 'w', compression=zipfile.ZIP_DEFLATED) as bundle:
            for name, data in sorted(contents.items()):
                info = zipfile.ZipInfo('RIIMA/' + name, date_time=(1980, 1, 1, 0, 0, 0))
                info.compress_type = zipfile.ZIP_DEFLATED
                info.external_attr = 0o644 << 16
                bundle.writestr(info, data)
        with zipfile.ZipFile(temporary) as bundle:
            if bundle.testzip() is not None:
                raise ValueError('Archive integrity check failed')
        os.replace(temporary, output)
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True, help='ZIP destination outside the source tree.')
    args = parser.parse_args()
    output = build_archive(ROOT, args.output)
    print(output)
    print(f'SHA-256: {hashlib.sha256(output.read_bytes()).hexdigest()}')


if __name__ == '__main__':
    main()
