"""Geometric and sampled-attribute summaries, with no mineral identification."""
import csv
import json

import numpy as np
import pytest

from riima import metrics
from test_riima_review import prepared_dataset


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
