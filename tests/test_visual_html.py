"""Portable document styling preserves the live classification and safe labels."""
import base64
from html import escape
from html.parser import HTMLParser
import io
import re

import numpy as np
import pytest
from PIL import Image

from riima import metrics, report, report_charts, viewer
from test_riima_review import prepared_dataset


class Document(HTMLParser):
    def __init__(self, html):
        super().__init__()
        self.images = []
        self.resources = []
        self.styles = []
        self._in_style = False
        self.feed(html)

    def handle_starttag(self, tag, attrs):
        attrs = dict(attrs)
        if tag == 'img':
            self.images.append(attrs)
        if tag in ('link', 'script'):
            self.resources.append((tag, attrs))
        if tag == 'style':
            self._in_style = True

    def handle_endtag(self, tag):
        if tag == 'style':
            self._in_style = False

    def handle_data(self, data):
        if self._in_style:
            self.styles.append(data)


@pytest.fixture
def styled_report(tmp_path, monkeypatch):
    prepared = prepared_dataset(tmp_path, count=1)
    name = '<script>Classe & "teste"</script>' + 'NomeLongo' * 30
    prepared['interpretation'].names[0] = name
    prepared['review'].palette[0] = (9, 136, 66)
    prepared['review'].palette[1] = (211, 37, 111)
    buffer = io.BytesIO()
    Image.new('RGB', (2, 2), 'white').save(buffer, format='PNG')
    page = base64.b64encode(buffer.getvalue()).decode('ascii')
    keys = ('fig1a', 'fig1b', 'fig1c', 'fig2', 'fig3a', 'fig3b', 'fig4a',
            'fig4b', 'fig5', 'fig6', 'fig7', 'fig8', 'fig9', 'fig10')
    monkeypatch.setattr(report_charts, 'make_charts', lambda *_: {key: [page] for key in keys})
    directory = metrics.export_metrics(prepared, tmp_path / 'metrics')
    html = report.build_report(prepared, directory)
    return html, Document(html), prepared, name


def test_report_embeds_shared_document_style_and_local_font_fallbacks(styled_report):
    from riima.visual_style import css_font_family, document_css
    html, document, _, _ = styled_report
    css = ''.join(document.styles)
    assert document_css() in css
    assert css_font_family() in css
    assert 'font-family:var(--font-family)' in css
    assert 'font-size:var(--font-body)' in css
    assert 'font-size:var(--font-table)' in css
    assert '${visual_style}' not in html
    assert not document.resources
    assert '@import' not in css and '@font-face' not in css
    assert 'url(' not in css


def test_report_style_wraps_long_names_and_keeps_rasters_undecorated(styled_report):
    _, document, _, _ = styled_report
    css = ''.join(document.styles)
    assert 'overflow-wrap:anywhere' in css
    assert 'minmax(0,1fr)' in css
    image_rules = re.search(r'(?m)^img\s*\{([^}]+)\}', css).group(1)
    assert 'max-width:100%' in image_rules
    assert 'height:auto' in image_rules
    assert 'border-radius:0' in image_rules
    assert 'box-shadow:none' in image_rules
    assert '@media (max-width:760px)' in css


def test_report_prints_with_dark_text_without_requiring_background_ink(styled_report):
    _, document, _, _ = styled_report
    css = ''.join(document.styles)
    print_css = css.split('@media print', 1)[1]
    assert re.search(r'nav\s*\{[^}]*display:none', print_css)
    header = re.search(r'header\s*\{([^}]+)\}', print_css).group(1)
    assert 'background:white' in header
    assert 'color:var(--ink)' in header
    assert 'box-shadow:none' in print_css
    assert 'break-after:avoid' in print_css
    assert 'thead' in print_css and 'table-header-group' in print_css
    assert 'print-color-adjust:exact' in print_css


def test_report_navigation_stays_visible_without_covering_anchor_titles(styled_report):
    _, document, _, _ = styled_report
    css = ''.join(document.styles).split('@media print', 1)[0]
    nav = re.search(r'(?m)^nav\s*\{([^}]+)\}', css).group(1)
    assert 'position:sticky' in nav
    assert 'top:var(--space-sm)' in nav
    assert 'max-height:30vh' in nav
    assert 'overflow-y:auto' in nav
    assert re.search(r'\[id\]\s*\{[^}]*scroll-margin-top:calc\(30vh \+ 2 \* var\(--space-md\)\)', css)


def test_print_keeps_figure_captions_tables_and_legends_together(styled_report):
    _, document, _, _ = styled_report
    print_css = ''.join(document.styles).split('@media print', 1)[1]
    assert re.search(r'\.figure-card, \.table-card\s*\{[^}]*break-inside:avoid', print_css)
    assert re.search(r'\.legend\s*\{[^}]*break-inside:avoid', print_css)
    assert re.search(r'\.classification-map\s*\{[^}]*max-height:70mm', print_css)


def test_shared_style_keeps_safe_names_group_colors_and_classification_pixels(styled_report):
    html, document, prepared, name = styled_report
    assert escape(name) in html
    assert '<script>Classe' not in html
    assert 'background:#098842' in html
    assert 'background:#d3256f' in html
    maps = [img for img in document.images if img.get('class') == 'classification-map']
    assert len(maps) == 1
    decoded = np.asarray(Image.open(io.BytesIO(base64.b64decode(maps[0]['src'].split(',', 1)[1]))))
    np.testing.assert_array_equal(decoded, viewer.render_workspace(prepared, 'Classificação'))
    assert all(img.get('alt') and img['src'].startswith('data:') for img in document.images)
    for text in ('Área relativa (%) = 100 × pixels do grupo / pixels válidos',
                 'x = A cos(2θ); y = A sin(2θ)', 'Compacidade = 4πA / P²'):
        assert text in html
