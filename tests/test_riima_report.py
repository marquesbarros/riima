"""A descriptive report follows applied masks, stable IDs and live RIIMA colors."""
import base64
import csv
from pathlib import Path
import io
from html.parser import HTMLParser

import numpy as np
import pytest
from PIL import Image

from riima import metrics
from riima import review
from riima import viewer
from test_riima_review import prepared_dataset


def test_standardization_keeps_missing_values_and_does_not_rank_constants():
    from riima.report_data import standardize
    z, varying = standardize(np.array([[1, 5, np.nan], [3, 5, 4]], float))
    np.testing.assert_allclose(z[:, 0], [-1, 1])
    np.testing.assert_allclose(z[:, 1], [0, 0])
    assert np.isnan(z[0, 2])
    assert varying.tolist() == [True, False, False]


def test_ward_distances_use_cluster_sizes_and_identical_profiles_are_finite():
    from riima.report_data import ward_linkage, similarity_matrix
    link, order = ward_linkage(np.array([[0], [2], [10]], float))
    np.testing.assert_allclose(link, [[0, 1, 2, 2], [2, 3, np.sqrt(108), 3]])
    assert order == [2, 0, 1]
    np.testing.assert_array_equal(similarity_matrix(np.zeros((3, 2))), np.ones((3, 3)))
    assert ward_linkage(np.zeros((1, 2)))[1] == [0]


def test_similarity_imputes_missing_attributes_before_standardization_like_the_reference():
    from riima.report_data import standardize_imputed, ward_linkage
    profile = np.array([[0, 0], [1, np.nan], [2, 1], [2.2, np.nan]], float)
    z = standardize_imputed(profile)
    np.testing.assert_allclose(z[:, 1], [-np.sqrt(2), 0, np.sqrt(2), 0])
    linkage, _ = ward_linkage(z)
    assert linkage[0, :2].tolist() == [1, 3]
    assert linkage[0, 2] == pytest.approx(1.36753, rel=1e-5)


def test_gaussian_density_handles_singular_and_small_samples():
    from riima.report_data import gaussian_density
    xs = np.linspace(-5, 5, 1001)
    values = gaussian_density(np.array([-1., 0., 1.]), xs)
    assert np.sum((values[:-1] + values[1:]) * .5 * np.diff(xs)) == pytest.approx(1, abs=.0001)
    assert values[500] == pytest.approx(.31815622, rel=.001)
    assert gaussian_density(np.ones(5), xs) is None
    assert gaussian_density(np.array([]), xs) is None


def test_report_profiles_keep_same_named_groups_distinct_and_use_exported_metrics(tmp_path):
    from riima.report_data import load_metrics
    prepared = prepared_dataset(tmp_path)
    prepared['interpretation'].names = {0: 'Mesmo nome', 1: 'Mesmo nome'}
    folder = metrics.export_metrics(prepared, tmp_path / 'metrics')
    data = load_metrics(folder)
    assert [g['id'] for g in data['groups']] == [0, 1]
    assert data['groups'][0]['name'] == data['groups'][1]['name'] == 'Mesmo nome'
    rows = list(csv.DictReader((folder / 'group_summary.csv').open(encoding='utf-8-sig')))
    assert sum(g['pixels'] for g in data['groups']) == sum(int(r['pixels']) for r in rows)
    assert data['valid_pixels'] == 16 * 24 + 20 * 28
    assert data['profile'].shape[0] == 2
    assert set(data['features']) == {'L*', 'a*', 'b*', 'T'}


class ImagesInReport(HTMLParser):
    def __init__(self):
        super().__init__()
        self.images = []

    def handle_starttag(self, tag, attrs):
        if tag == 'img':
            self.images.append(dict(attrs))


def test_report_of_resumed_registered_views_has_one_final_map_and_all_optical_attributes(tmp_path):
    import cv2
    from riima import core
    from riima import report
    rgb = np.random.default_rng(13).integers(50, 220, (40, 48, 3), np.uint8)
    paths = []
    for i in range(2):
        path = tmp_path / f'illuminacao_{i}.png'
        cv2.imwrite(str(path), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
        paths.append(str(path))
    prepared = core.prepare_analysis(paths, str(tmp_path / 'out'),
                                     optical_cfg=core.Config(clusters=2, sample_pixels=300),
                                     mask_mode='overlap_only')
    prepared['review'] = review.ClusterReview.from_prepared(prepared)
    prepared['review'].palette[0] = (9, 136, 66)
    prepared['review'].selected_image = 1
    historical = Path(prepared['output'])
    historical.mkdir(exist_ok=True)
    sentinel = historical / 'previous-result.txt'
    sentinel.write_bytes(b'previous scientific results')
    historical_bytes = {p.relative_to(historical): p.read_bytes()
                        for p in historical.rglob('*') if p.is_file()}
    archive = tmp_path / 'registered.riima'
    review.save_project(prepared, archive)
    resumed = review.load_project(archive)
    active = Path(resumed['output'])
    assert active == tmp_path / 'registered_output'
    assert resumed['output_history'] == [str(historical)]
    target = report.export_report(resumed, tmp_path / 'registered.html')
    html = target.read_text(encoding='utf-8')
    parser = ImagesInReport()
    parser.feed(html)
    assert len([img for img in parser.images if img.get('class') == 'classification-map']) == 1
    assert 'illuminacao_0.png' in html and 'illuminacao_1.png' in html
    assert '#098842' in html
    ranks = list(csv.DictReader((active / 'metrics/attribute_separation.csv').open(encoding='utf-8-sig')))
    assert len(ranks) == 8
    assert {p.relative_to(historical): p.read_bytes()
            for p in historical.rglob('*') if p.is_file()} == historical_bytes
    assert resumed['review'].selected_image == 1
    assert resumed['review'].items[0]['sources'] is None


def test_report_exports_live_classification_colors_and_safe_names_before_item_one(tmp_path, monkeypatch):
    from riima import report
    prepared = prepared_dataset(tmp_path)
    state = prepared['review']
    state.palette[0], state.palette[1] = (9, 136, 66), (211, 37, 111)
    prepared['interpretation'].names[0] = '<script>Nome & teste</script>'
    state.exclude(0, [195], True)
    state.refine_pending()
    originals = [item['rgb'].copy() for item in state.items]
    labels = [item['labels'].copy() for item in state.items]
    locked = [item['locked'].copy() for item in state.items]
    monkeypatch.setattr(state, '_fit_model', lambda *a: pytest.fail('report must not train'))
    target = tmp_path / 'descritivo.html'
    exported = report.export_report(prepared, target)
    assert exported == target
    html = target.read_text(encoding='utf-8')
    assert html.index('id="classificacao"') < html.index('id="tamanho"')
    assert '#098842' in html and '#d3256f' in html
    assert '<script>Nome' not in html
    assert '&lt;script&gt;Nome &amp; teste&lt;/script&gt;' in html
    parser = ImagesInReport()
    parser.feed(html)
    maps = [img for img in parser.images if img.get('class') == 'classification-map']
    assert len(maps) == 2
    decoded = np.asarray(Image.open(io.BytesIO(base64.b64decode(maps[0]['src'].split(',', 1)[1]))))
    np.testing.assert_array_equal(decoded, viewer.render_workspace(prepared, 'Classificação'))
    assert all(img['src'].startswith('data:image/png;base64,') and img.get('alt') for img in parser.images)
    for section in ('descritores', 'similaridade', 'direcional', 'distribuicoes', 'complementares', 'referencias'):
        assert f'id="{section}"' in html
    assert 'Gráfico 10.' in html
    for item, rgb, current, protected in zip(state.items, originals, labels, locked):
        np.testing.assert_array_equal(item['rgb'], rgb)
        np.testing.assert_array_equal(item['labels'], current)
        np.testing.assert_array_equal(item['locked'], protected)


@pytest.mark.parametrize('empty', [False, True])
def test_report_handles_single_group_isotropic_or_empty_after_merge_and_resume(tmp_path, empty):
    from riima import report
    prepared = prepared_dataset(tmp_path, count=1)
    prepared['interpretation'].merge_clusters([0, 1])
    prepared['review'].sync_groups(prepared['interpretation'].groups)
    item = prepared['review'].items[0]
    if empty:
        prepared['review'].exclude(0, np.arange(item['valid'].size), True)
        prepared['review'].refine_pending()
    else:
        # Square components have no defined direction.
        item['labels'].fill(-1)
        item['labels'][2:12, 2:12] = 0
    archive = tmp_path / 'resume.riima'
    review.save_project(prepared, archive)
    restored = review.load_project(archive)
    output = report.export_report(restored, tmp_path / 'one.html')
    html = output.read_text(encoding='utf-8')
    assert 'id="tamanho"' in html and 'Gráfico 10.' in html
    assert 'Sem dados' in html if empty else 'Não definida' in html


def test_pending_report_and_failed_write_keep_existing_report(tmp_path, monkeypatch):
    from riima import report
    prepared = prepared_dataset(tmp_path, count=1)
    existing = tmp_path / 'report.html'
    existing.write_text('previous export', encoding='utf-8')
    prepared['review'].annotate(0, 0, [212], 1)
    with pytest.raises(ValueError, match='Atualiz'):
        report.export_report(prepared, existing)
    assert existing.read_text(encoding='utf-8') == 'previous export'
    prepared['review'].refine_pending()
    monkeypatch.setattr(report.os, 'replace', lambda *args: (_ for _ in ()).throw(OSError('disk full')))
    with pytest.raises(OSError, match='disk full'):
        report.export_report(prepared, existing)
    assert existing.read_text(encoding='utf-8') == 'previous export'
    assert not list(tmp_path.glob('.riima-report-*'))
