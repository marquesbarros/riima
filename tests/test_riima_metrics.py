"""Geometric and sampled-attribute summaries, with no mineral identification."""
import csv
import json

import numpy as np
import pytest

from riima import metrics
from test_riima_review import prepared_dataset


def test_empty_created_group_does_not_reduce_existing_sampling_budget(tmp_path):
    import cv2
    from riima import core, report_data, review
    colors = np.array([[40 + 22 * i, 60 + 15 * i, 180 - 17 * i] for i in range(8)], np.uint8)
    rgb = np.repeat(np.repeat(colors[None, :, :], 48, axis=0), 48, axis=1)
    image = tmp_path / 'eight-materials.png'
    cv2.imwrite(str(image), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
    prepared = review.prepare_independent_analysis([str(image)], str(tmp_path / 'out'),
        core.Config(clusters=8, feature_blur_sigma=0, sample_pixels=4000), mask_mode='overlap_only')
    state = prepared['review']
    # Fix the applied segmentation so every group exercises the sampling cap.
    state.items[0]['labels'][:] = np.repeat(np.arange(8), 48)[None, :]
    assert np.bincount(state.items[0]['labels'].ravel()).min() > 1250
    before = metrics.collect_metrics(prepared)
    profiles = report_data.build_metrics_data(before)
    created = state.create_group('Sem exemplos', (17, 39, 63))
    after = metrics.collect_metrics(prepared)
    assert after['info']['effective_sample_per_group'] == before['info']['effective_sample_per_group'] == 1250
    for key in ('summary_rows', 'region_rows', 'attribute_rows'):
        assert [row for row in after[key] if row['group'] != created] == before[key]
    assert after['ranks'] == before['ranks']
    current = report_data.build_metrics_data(after)
    for key in ('similarity', 'linkage'):
        np.testing.assert_array_equal(current[key], profiles[key])
    # Once populated, a new category shares the cap; the now-empty legacy
    # category still counts, preserving the historical budgeting contract.
    state.items[0]['labels'][:, :48] = created
    populated = metrics.collect_metrics(prepared)
    assert populated['info']['effective_sample_per_group'] == 10000 // 9
    assert {row['sampled_pixels'] for row in populated['attribute_rows']} == {10000 // 9}


def test_created_empty_group_metrics_export_true_zero_and_current_names(tmp_path):
    prepared = prepared_dataset(tmp_path, count=1)
    state = prepared['review']
    retired = state.create_group('Removido')
    state.delete_group(retired)
    created = state.create_group('Sem medidas', (17, 39, 63))
    state.rename_group(0, 'Material atual')
    data = metrics.collect_metrics(prepared)
    assert {row['group'] for row in data['summary_rows']} == {0, 1, created}
    empty = next(row for row in data['summary_rows'] if row['group'] == created)
    assert empty['group_name'] == 'Sem medidas'
    for key in ('pixels', 'area_percent', 'components_total', 'components_measured', 'components_omitted'):
        assert empty[key] == 0
    for key in ('component_area_mean_px', 'mean_eccentricity', 'mean_anisotropy', 'dominant_direction_deg'):
        assert empty[key] is None
    assert not any(row['group'] == created for row in data['region_rows'] + data['attribute_rows'])
    exported = metrics.export_metrics(prepared, tmp_path / 'current-metrics')
    with (exported / 'group_summary.csv').open(encoding='utf-8-sig') as handle:
        rows = list(csv.DictReader(handle))
    assert {int(row['group']) for row in rows} == {0, 1, created}
    assert rows[0]['group_name'] == 'Material atual'


def test_current_group_metadata_does_not_change_numeric_metrics(tmp_path):
    prepared = prepared_dataset(tmp_path, count=1)
    before = metrics.collect_metrics(prepared, sample_per_group=17)
    state = prepared['review']
    state.rename_group(0, 'Nome interpretativo')
    state.recolor_group(0, (17, 39, 63))
    after = metrics.collect_metrics(prepared, sample_per_group=17)
    for key in ('summary_rows', 'region_rows', 'attribute_rows'):
        strip_names = lambda rows: [{k: v for k, v in row.items() if k != 'group_name'} for row in rows]
        assert strip_names(after[key]) == strip_names(before[key])
    assert after['ranks'] == before['ranks']


def test_rectangle_geometry_and_direction_are_exact_and_circular_regions_have_no_direction():
    mask = np.zeros((25, 45), bool)
    mask[8:13, 5:35] = True
    summary, regions = metrics.mask_metrics(mask)
    assert summary['pixels'] == 150
    assert summary['components_total'] == 1
    assert regions[0]['perimeter_grid_px'] == 70
    assert regions[0]['orientation_deg'] == pytest.approx(0)
    assert regions[0]['anisotropy'] > .9
    assert summary['dominant_direction_deg'] == pytest.approx(0)
    rotated, _ = metrics.mask_metrics(mask.T)
    assert rotated['dominant_direction_deg'] == pytest.approx(90)
    square = np.ones((12, 12), bool)
    _, rows = metrics.mask_metrics(square)
    assert rows[0]['orientation_deg'] is None
    assert rows[0]['anisotropy'] == pytest.approx(0)


def test_component_limit_is_explicit_and_never_truncates_area_counts():
    mask = np.zeros((24, 24), bool)
    mask[::3, ::3] = True
    summary, rows = metrics.mask_metrics(mask, min_area=1, max_regions=5)
    assert summary['pixels'] == 64
    assert summary['components_total'] == 64
    assert summary['components_measured'] == len(rows) == 5
    assert summary['components_omitted'] == 59


def test_metrics_export_uses_current_masks_excludes_background_and_keeps_originals(tmp_path, monkeypatch):
    prepared = prepared_dataset(tmp_path)
    state = prepared['review']
    state.exclude(0, [195], True)
    state.refine_pending()
    original = [item['rgb'].copy() for item in state.items]
    monkeypatch.setattr(state, '_fit_model', lambda *a: pytest.fail('training during metrics export'))
    path = metrics.export_metrics(prepared, tmp_path / 'metrics', sample_per_group=80)
    rows = list(csv.DictReader((path / 'group_summary.csv').open(encoding='utf-8-sig')))
    first = [row for row in rows if row['image_id'] == '0']
    assert sum(int(row['pixels']) for row in first) == state.items[0]['valid'].sum() - 1
    assert (path / 'regions.csv').is_file()
    regions = list(csv.DictReader((path / 'regions.csv').open(encoding='utf-8-sig')))
    assert len(regions) == sum(int(row['components_measured']) for row in rows)
    assert all(row['image_id'] in ('0', '1') for row in regions)
    assert (path / 'attribute_statistics.csv').is_file()
    ranks = list(csv.DictReader((path / 'attribute_separation.csv').open(encoding='utf-8-sig')))
    assert len(ranks) == 4
    assert all(0 <= float(row['separation_eta2']) <= 1 for row in ranks)
    info = json.loads((path / 'metrics_info.json').read_text(encoding='utf-8'))
    assert info['units'] == 'pixels'
    assert info['sample_per_group'] == 80
    for item, rgb in zip(state.items, original):
        np.testing.assert_array_equal(item['rgb'], rgb)


def test_resume_streams_the_same_attributes_without_retaining_full_feature_planes(tmp_path):
    from riima import review
    prepared = prepared_dataset(tmp_path)
    first = metrics.export_metrics(prepared, tmp_path / 'before')
    project = tmp_path / 'resumed.riima'
    review.save_project(prepared, project)
    loaded = review.load_project(project)
    second = metrics.export_metrics(loaded, tmp_path / 'after')
    assert (first / 'attribute_statistics.csv').read_bytes() == (second / 'attribute_statistics.csv').read_bytes()
    assert all(item['sources'] is None for item in loaded['review'].items)


def test_grid_perimeter_includes_holes_and_metrics_do_not_apply_pending_edits(tmp_path):
    mask = np.ones((15, 15), bool)
    mask[5:10, 5:10] = False
    _, rows = metrics.mask_metrics(mask)
    assert rows[0]['perimeter_grid_px'] == 80
    prepared = prepared_dataset(tmp_path)
    state = prepared['review']
    before = state.items[0]['labels'].copy()
    state.exclude(0, [195], True)
    with pytest.raises(ValueError, match='Atualiz'):
        metrics.export_metrics(prepared, tmp_path / 'pending')
    np.testing.assert_array_equal(state.items[0]['labels'], before)


@pytest.mark.parametrize('duplicate_names', [False, True])
def test_registered_metrics_preserve_all_4n_attribute_names_on_resume(tmp_path, duplicate_names):
    import cv2
    from riima import core
    from riima import review
    rng = np.random.default_rng(13)
    rgb = rng.integers(50, 220, (40, 48, 3), np.uint8)
    paths = []
    for i in range(2):
        parent = tmp_path / str(i) if duplicate_names else tmp_path
        parent.mkdir(exist_ok=True)
        path = parent / ('view.png' if duplicate_names else f'view_{i}.png')
        cv2.imwrite(str(path), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
        paths.append(str(path))
    prepared = core.prepare_analysis(paths, str(tmp_path / 'out'),
                                     optical_cfg=core.Config(clusters=2, sample_pixels=300),
                                     mask_mode='overlap_only')
    prepared['review'] = review.ClusterReview.from_prepared(prepared)
    project = tmp_path / 'registered.riima'
    review.save_project(prepared, project)
    restored = review.load_project(project)
    exported = metrics.export_metrics(restored, tmp_path / 'metrics')
    ranks = list(csv.DictReader((exported / 'attribute_separation.csv').open(encoding='utf-8-sig')))
    assert len(ranks) == 8
    assert len({row['feature'] for row in ranks}) == 8
    assert all(item['sources'] is None for item in restored['review'].items)


def test_collect_metrics_is_in_memory_and_preserves_sample_order_and_geometry(tmp_path):
    prepared = prepared_dataset(tmp_path, count=1)
    item = prepared['review'].items[0]
    item['labels'].fill(-1)
    item['labels'][1:4, 1:5] = 0
    item['labels'][8:11, 12:16] = 1
    # Integer source planes make the exact seeded sample independently checkable.
    plane = np.arange(item['labels'].size, dtype=np.float32).reshape(item['labels'].shape)
    item['sources'] = [plane + j * 1000 for j in range(4)]
    before = {p.relative_to(tmp_path): p.read_bytes() for p in tmp_path.rglob('*') if p.is_file()}
    assert callable(getattr(metrics, 'collect_metrics', None)), 'metrics need a pure collection API'
    data = metrics.collect_metrics(prepared, sample_per_group=3, min_area=1, max_regions=1)
    assert {p.relative_to(tmp_path): p.read_bytes() for p in tmp_path.rglob('*') if p.is_file()} == before
    assert [row['group'] for row in data['summary_rows']] == [0, 1]
    assert [row['area_pixels'] for row in data['region_rows']] == [12, 12]
    assert [row['perimeter_grid_px'] for row in data['region_rows']] == [14, 14]
    rng = np.random.default_rng(prepared['config'].seed)
    for group in (0, 1):
        sampled = rng.choice(np.flatnonzero(item['labels'] == group), 3, replace=False)
        rows = [r for r in data['attribute_rows'] if r['group'] == group]
        assert [r['feature'] for r in rows] == ['L*', 'a*', 'b*', 'T']
        for j, row in enumerate(rows):
            assert row['sampled_pixels'] == 3
            assert row['mean'] == pytest.approx(sampled.mean() + j * 1000)
            assert row['std'] == pytest.approx(sampled.std())
            np.testing.assert_allclose([row['q25'], row['median'], row['q75']],
                                       np.percentile(sampled, [25, 50, 75]) + j * 1000)
    assert data['info']['effective_sample_per_group'] == 3
    assert len(data['ranks']) == 4


def test_collect_metrics_rejects_pending_edits_without_writing_or_changing_labels(tmp_path):
    prepared = prepared_dataset(tmp_path, count=1)
    prepared['review'].exclude(0, [195], True)
    before = prepared['review'].items[0]['labels'].copy()
    assert callable(getattr(metrics, 'collect_metrics', None)), 'metrics need a pure collection API'
    with pytest.raises(ValueError, match='Atualiz'):
        metrics.collect_metrics(prepared)
    np.testing.assert_array_equal(prepared['review'].items[0]['labels'], before)
    assert not (tmp_path / 'out/metrics').exists()
