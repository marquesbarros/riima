"""Scientific exports preserve the last complete generation on recoverable errors."""
import csv
import json
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from riima import core, metrics, review, report
from test_riima_review import prepared_dataset


def file_bytes(folder):
    return {p.relative_to(folder).as_posix(): p.read_bytes()
            for p in folder.rglob('*') if p.is_file()}


@pytest.mark.parametrize('failure', ['html_write', 'html_publish'])
def test_report_failure_preserves_metrics_and_external_html(tmp_path, monkeypatch, failure):
    prepared = prepared_dataset(tmp_path)
    target = tmp_path / 'Relatório ç.html'
    monkeypatch.setattr(report, 'build_report', lambda *args, **kwargs: '<html>complete report</html>')
    report.export_report(prepared, target)
    before, state = file_bytes(tmp_path), scientific_state(prepared)
    with monkeypatch.context() as patch:
        if failure == 'html_write':
            original = report.tempfile.NamedTemporaryFile
            class FailedFile:
                def __init__(self, *args, **kwargs):
                    self.handle = original(*args, **kwargs)
                def __enter__(self):
                    self.handle.__enter__()
                    return self
                @property
                def name(self):
                    return self.handle.name
                def write(self, text):
                    self.handle.write('<html>partial')
                    raise OSError('HTML write failed')
                def __exit__(self, *args):
                    return self.handle.__exit__(*args)
            patch.setattr(report.tempfile, 'NamedTemporaryFile', FailedFile)
        else:
            original = report.os.replace
            def fail(source, destination):
                if Path(destination) == target:
                    raise OSError('HTML publication failed')
                return original(source, destination)
            patch.setattr(report.os, 'replace', fail)
        with pytest.raises(OSError, match='HTML'):
            report.export_report(prepared, target)
    assert file_bytes(tmp_path) == before
    assert scientific_state(prepared) == state
    assert report.export_report(prepared, target) == target
    assert file_bytes(tmp_path) == before


def scientific_state(prepared):
    state = prepared['review']
    def snapshot(value):
        if isinstance(value, np.ndarray):
            return (value.dtype.str, value.shape, value.tobytes(), value.flags.writeable)
        if isinstance(value, dict):
            return {key: snapshot(item) for key, item in value.items()}
        if isinstance(value, (list, tuple)):
            return [snapshot(item) for item in value]
        if isinstance(value, set):
            return set(value)
        return value
    return dict(config=vars(prepared['config']).copy(), interpretation=prepared['interpretation'].to_dict(),
                groups={k: set(v) for k, v in state.groups.items()}, completed=set(state.completed),
                dirty=set(state.dirty), selected=(state.selected_image, state.selected_cluster),
                items=snapshot(state.items), centers=snapshot(state.centers), palette=snapshot(state.palette),
                models={key: review._model_text(value) for key, value in state.models.items()})


def fail_writes(patch, kind):
    if kind == 'npy':
        def fail(stream, *args, **kwargs):
            stream.write(b'incomplete array')
            raise OSError('simulated NPY write failure')
        patch.setattr(np.lib.format, 'write_array', fail)
    elif kind == 'png':
        def fail(self, path, *args, **kwargs):
            Path(path).write_bytes(b'incomplete image')
            raise OSError('simulated PNG write failure')
        patch.setattr(Image.Image, 'save', fail)
    elif kind == 'json':
        original = Path.write_text
        def fail(path, text, *args, **kwargs):
            if path.name in ('interpretation.json', 'analysis_settings.json', 'metrics_info.json'):
                original(path, '{', encoding='utf-8')
                raise OSError('simulated JSON write failure')
            return original(path, text, *args, **kwargs)
        patch.setattr(Path, 'write_text', fail)
    elif kind == 'csv':
        original = csv.writer
        calls = [0]
        class Writer:
            def __init__(self, stream, *args, **kwargs):
                self.stream, self.writer = stream, original(stream, *args, **kwargs)
            def writerow(self, row):
                calls[0] += 1
                if calls[0] == 2:
                    self.stream.write('incomplete row')
                    raise OSError('simulated CSV write failure')
                return self.writer.writerow(row)
            def writerows(self, rows):
                for row in rows:
                    self.writerow(row)
        patch.setattr(csv, 'writer', Writer)


@pytest.mark.parametrize('exporter, kind', [
    ('review', 'npy'), ('review', 'csv'), ('review', 'png'),
    ('session', 'npy'), ('session', 'csv'), ('session', 'png'), ('session', 'json'),
    ('metrics', 'csv'), ('metrics', 'json'),
])
@pytest.mark.parametrize('existing', [False, True])
def test_failed_writes_preserve_the_entire_previous_generation(tmp_path, monkeypatch, exporter, kind, existing):
    prepared = prepared_dataset(tmp_path)
    output = Path(prepared['output'])
    export = {'review': review.save_review_outputs, 'session': core.save_session_outputs,
              'metrics': metrics.export_metrics}[exporter]
    if existing:
        export(prepared)
        (output / 'user-file.txt').write_bytes(b'keep this unrelated file')
    before, state = file_bytes(output), scientific_state(prepared)
    prepared['interpretation'].rename_cluster(0, 'Updated group')
    state = scientific_state(prepared)
    with monkeypatch.context() as patch:
        fail_writes(patch, kind)
        with pytest.raises(OSError, match='simulated'):
            export(prepared)
    assert file_bytes(output) == before
    assert scientific_state(prepared) == state
    export(prepared)
    assert scientific_state(prepared) == state
    if exporter == 'review':
        np.testing.assert_array_equal(np.load(output / 'review/image_000/labels.npy'),
                                      prepared['review'].items[0]['labels'])
    elif exporter == 'session':
        np.testing.assert_array_equal(np.load(output / 'clusters.npy'), prepared['optical']['labels'])
        assert json.loads((output / 'interpretation.json').read_text())['names']['0'] == 'Updated group'
    else:
        assert (output / 'metrics/metrics_info.json').is_file()


def test_failed_final_preparation_callback_preserves_optical_outputs(tmp_path):
    prepared = prepared_dataset(tmp_path, count=1)
    paths = [prepared['review'].items[0]['path']]
    output = Path(prepared['output'])
    core.run_analysis(paths, str(output), core.Config(clusters=2, sample_pixels=300), mask_mode='overlap_only')
    before = file_bytes(output)
    def fail(stage):
        if stage.startswith('Preparando'):
            raise OSError('simulated final callback failure')
    with pytest.raises(OSError, match='simulated'):
        core.prepare_analysis(paths, str(output), optical_cfg=core.Config(clusters=3, sample_pixels=300),
                              mask_mode='overlap_only', stage_callback=fail)
    assert file_bytes(output) == before


def test_report_and_metrics_are_one_generation_when_rendering_fails(tmp_path, monkeypatch):
    prepared = prepared_dataset(tmp_path)
    output = Path(prepared['output'])
    target = tmp_path / 'Relatório em outra pasta' / 'resultado.html'
    target.parent.mkdir()
    target.write_bytes(b'previous complete HTML')
    metrics.export_metrics(prepared)
    before = file_bytes(output)
    prepared['interpretation'].rename_cluster(0, 'New description')
    state = scientific_state(prepared)
    def fail(*args, **kwargs):
        raise OSError('simulated chart failure')
    monkeypatch.setattr(report, 'build_report', fail)
    with pytest.raises(OSError, match='simulated'):
        report.export_report(prepared, target)
    assert file_bytes(output) == before
    assert target.read_bytes() == b'previous complete HTML'
    assert scientific_state(prepared) == state


def test_cli_failure_after_merge_preserves_the_previous_generation(tmp_path, monkeypatch):
    from riima import gui
    prepared = prepared_dataset(tmp_path, count=1)
    path = prepared['review'].items[0]['path']
    output = Path(prepared['output'])
    core.run_analysis([path], str(output), optical_cfg=core.Config(clusters=2, sample_pixels=300))
    before = file_bytes(output)
    monkeypatch.setattr('sys.argv', ['riima', '--image', path, '--output', str(output),
                                     '--clusters', '2', '--sample', '300', '--merge', '0+1'])
    original = core.save_session_outputs
    calls = []
    def fail(data):
        calls.append(data)
        if len(calls) == 2:
            raise OSError('merged export failed')
        return original(data)
    monkeypatch.setattr(core, 'save_session_outputs', fail)
    with pytest.raises(OSError, match='merged export'):
        gui.main()
    assert len(calls) == 2
    assert file_bytes(output) == before
