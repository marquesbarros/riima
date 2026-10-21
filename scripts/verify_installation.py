#!/usr/bin/env python3
"""Smoke-test an installed artifact from outside its source checkout.

Use with python -I; only the installed package and synthetic generated inputs
are used. Missing resources, CLI errors and report failures are fatal.
"""
from __future__ import annotations

import importlib.metadata
from importlib.resources import files
from pathlib import Path
import subprocess
import sys


def main() -> None:
    if Path.cwd().resolve().is_relative_to(Path(__file__).resolve().parents[1]):
        raise ValueError('Run installation verification from outside the source tree.')
    import numpy as np
    from PIL import Image
    import riima
    from riima import core, report, review

    package = Path(riima.__file__).resolve()
    if not package.is_relative_to(Path(sys.prefix).resolve()):
        raise AssertionError(f'Package is outside the isolated environment: {package}')
    metadata = importlib.metadata.metadata('riima')
    assert metadata['License'] or metadata['License-Expression']
    for resource in ('templates/descriptive_report.html',
                     'assets/branding/riima_logo.png', 'assets/branding/riima_icon.ico'):
        assert files('riima').joinpath(resource).read_bytes(), resource
    image = Path.cwd() / 'synthetic.png'
    rgb = np.random.default_rng(31).integers(50, 220, (40, 48, 3), dtype=np.uint8)
    Image.fromarray(rgb).save(image)
    completed = subprocess.run(
        [sys.executable, '-I', '-m', 'riima', '--image', str(image), '--clusters', '3',
         '--sample', '500', '--output', str(Path.cwd() / 'cli-output')],
        check=True, capture_output=True, text=True, timeout=90)
    assert (Path.cwd() / 'cli-output/clusters.npy').is_file(), completed.stdout
    prepared = core.prepare_analysis([str(image)], str(Path.cwd() / 'report-output'),
                                     optical_cfg=core.Config(clusters=3, sample_pixels=500))
    prepared['review'] = review.ClusterReview.from_prepared(prepared)
    target = report.export_report(prepared, Path.cwd() / 'report.html')
    html = Path(target).read_text(encoding='utf-8')
    assert 'data:image/' in html and '${fig' not in html
    assert Path.cwd().joinpath('report-output/metrics/group_summary.csv').is_file()
    from riima.gui import gui
    def check_gui(app):
        root = app['root']
        try:
            root.withdraw()
            root.update_idletasks()
            assert root.winfo_exists()
            assert app['preview_canvas'].winfo_exists()
            assert app['session']['prepared'] is None
        finally:
            root.destroy()
    gui(_test_hook=check_gui)
    print(f'Installed artifact verified: {package}; {metadata["Version"]}')


if __name__ == '__main__':
    main()
