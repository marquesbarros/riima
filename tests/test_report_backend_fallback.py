"""Keep export usable when an optional graphics DLL cannot be loaded."""
import base64
import builtins
from html.parser import HTMLParser
import xml.etree.ElementTree as ET

import numpy as np
import pytest

from riima import report
from test_riima_review import prepared_dataset


class ReportImages(HTMLParser):
    def __init__(self):
        super().__init__()
        self.images = []

    def handle_starttag(self, tag, attrs):
        if tag == 'img':
            self.images.append(dict(attrs))


@pytest.mark.parametrize('failure', [
    ImportError('DLL load failed while importing _c_internal_utils: '
                'Uma política de Controle de Aplicativo bloqueou este arquivo.'),
    OSError(126, 'The specified module could not be found'),
    ModuleNotFoundError("No module named 'matplotlib'"),
])
def test_report_exports_all_charts_when_optional_graphics_cannot_import(tmp_path, monkeypatch, failure):
    prepared = prepared_dataset(tmp_path)
    state = prepared['review']
    state.palette[0], state.palette[1] = (9, 136, 66), (211, 37, 111)
    prepared['interpretation'].names[0] = '<script>Classe & teste</script>'
    state.approve(0)
    labels = [item['labels'].copy() for item in state.items]
    locked = [item['locked'].copy() for item in state.items]
    originals = [item['rgb'].copy() for item in state.items]
    original_import = builtins.__import__

    def blocked_import(name, *args, **kwargs):
        if name == 'matplotlib' or name.startswith('matplotlib.'):
            raise failure
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, '__import__', blocked_import)
    output = report.export_report(prepared, tmp_path / 'relatório.html')
    html = output.read_text(encoding='utf-8')
    assert html.index('id="classificacao"') < html.index('id="tamanho"')
    assert '#098842' in html and '#d3256f' in html
    assert '<script>Classe' not in html
    parser = ReportImages()
    parser.feed(html)
    figures = [item for item in parser.images if item.get('class') != 'classification-map']
    assert len(figures) >= 14
    for image in figures:
        assert image['src'].startswith('data:image/svg+xml;base64,')
        svg = ET.fromstring(base64.b64decode(image['src'].split(',', 1)[1]))
        assert svg.tag == '{http://www.w3.org/2000/svg}svg'
        assert not svg.findall('.//{http://www.w3.org/2000/svg}script')
    for i, item in enumerate(state.items):
        np.testing.assert_array_equal(item['labels'], labels[i])
        np.testing.assert_array_equal(item['locked'], locked[i])
        np.testing.assert_array_equal(item['rgb'], originals[i])
    assert state.completed == {0}


@pytest.mark.parametrize('error_type', [ValueError, ImportError, OSError])
def test_optional_backend_does_not_hide_errors_in_chart_data(tmp_path, monkeypatch, error_type):
    from riima import report_charts
    original = tmp_path / 'previous.html'
    original.write_text('previous report', encoding='utf-8')

    def broken(*args, **kwargs):
        raise error_type('Invalid chart data')

    monkeypatch.setattr(report_charts, '_make_matplotlib_charts', broken, raising=False)
    with pytest.raises(error_type, match='Invalid chart data'):
        report.export_report(prepared_dataset(tmp_path), original)
    assert original.read_text(encoding='utf-8') == 'previous report'
