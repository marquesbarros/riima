"""Behavioral tests for dataset-wide interactive cluster review."""
from pathlib import Path
import json

import cv2
import numpy as np
import pytest

from riima import core
from riima import review
def prepared_dataset(tmp_path, count=2, clusters=2):
    paths = []
    for i in range(count):
        # Same two materials in distinct fields and unequal geometries.
        rgb = np.empty((16 + i * 4, 24 + i * 4, 3), np.uint8)
        rgb[:, :rgb.shape[1] // 2] = (100, 80, 60)
        rgb[:, rgb.shape[1] // 2:] = (180, 160, 140)
        path = tmp_path / f'field_{i}.png'
        cv2.imwrite(str(path), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
        paths.append(str(path))
    return review.prepare_independent_analysis(
        paths, str(tmp_path / 'out'), core.Config(clusters=clusters, feature_blur_sigma=0,
                                                 sample_pixels=400, chunk_pixels=99),
        mask_mode='overlap_only')


def test_independent_images_share_fit_and_retain_geometry(tmp_path, monkeypatch):
    monkeypatch.setattr(core, 'register_images', lambda *a, **k: pytest.fail('registration'))
    prepared = prepared_dataset(tmp_path)
    state = review.ClusterReview.from_prepared(prepared)
    assert len(state.items) == 2
    assert state.items[0]['initial'].shape == (16, 24)
    assert state.items[1]['initial'].shape == (20, 28)
    assert state.items[0]['initial'][8, 3] == state.items[1]['initial'][8, 3]
    assert state.centers.shape == (2, 4)


def test_markings_learn_across_images_and_negative_stays_available(tmp_path):
    state = review.ClusterReview.from_prepared(prepared_dataset(tmp_path))
    target = int(state.items[0]['initial'][8, 3])
    # Deliberately redefine this cluster as the opposite material using one field.
    state.annotate(0, target, [8 * 24 + 20], 1)
    state.annotate(0, target, [8 * 24 + 3], -1)
    state.refine(target)
    assert state.items[0]['labels'][8, 20] == target
    assert state.items[0]['labels'][8, 3] != target
    assert state.items[1]['labels'][8, 23] == target  # unannotated field learned
    assert state.items[1]['locked'][8, 3] == -1
    assert state.items[1]['valid'][8, 3]


def test_approval_freezes_whole_prediction_and_remaining_excludes_id(tmp_path):
    state = review.ClusterReview.from_prepared(prepared_dataset(tmp_path))
    target = int(state.items[0]['initial'][8, 3])
    raw = [item['initial'].copy() for item in state.items]
    masks = [item['labels'] == target for item in state.items]
    state.approve(target)
    assert state.completed == {target}
    for item, mask in zip(state.items, masks):
        np.testing.assert_array_equal(item['locked'] == target, mask)
        assert not np.any((item['labels'] == target) & (item['locked'] < 0))
    other = 1 - target
    state.annotate(0, other, [8 * 24 + 20], 1)
    state.refine(other)
    for item, mask, initial in zip(state.items, masks, raw):
        assert np.all(item['labels'][mask] == target)
        np.testing.assert_array_equal(item['initial'], initial)
    with pytest.raises(ValueError):
        state.annotate(0, target, [0], 1)


def test_reopening_releases_only_selected_class(tmp_path):
    state = review.ClusterReview.from_prepared(prepared_dataset(tmp_path))
    state.approve(0)
    state.approve(1)
    other = [item['locked'] == 1 for item in state.items]
    state.reopen(0)
    assert state.completed == {1}
    for item, mask in zip(state.items, other):
        np.testing.assert_array_equal(item['locked'] == 1, mask)
        assert not np.any(item['locked'] == 0)


def test_roundtrip_restores_images_models_and_pending_annotations(tmp_path):
    prepared = prepared_dataset(tmp_path)
    state = review.ClusterReview.from_prepared(prepared)
    prepared['review'] = state
    target = int(state.items[0]['initial'][8, 3])
    state.annotate(0, target, [8 * 24 + 20], 1)
    state.annotate(0, target, [8 * 24 + 3], -1)
    state.refine(target)
    state.approve(target)
    state.annotate(1, 1 - target, [8 * 28 + 3], -1)  # pending, not yet refined
    state.selected_image = 1
    state.selected_cluster = 1 - target
    path = tmp_path / 'project.riima'
    review.save_project(prepared, path)
    for original in tmp_path.glob('field_*.png'):
        original.unlink()
    loaded = review.load_project(path)
    restored = loaded['review']
    assert restored.completed == state.completed
    assert restored.selected_image == 1
    assert restored.selected_cluster == 1 - target
    assert restored.dirty == state.dirty
    assert restored.models[target].isTrained()
    for original, actual in zip(state.items, restored.items):
        for key in ('rgb', 'initial', 'labels', 'locked', 'valid'):
            np.testing.assert_array_equal(actual[key], original[key])
        for cluster, annotation in original['annotations'].items():
            np.testing.assert_array_equal(actual['annotations'][cluster], annotation)
    restored.refine(1 - target)
    for item in restored.items:
        assert np.all(item['labels'][item['locked'] == target] == target)


def test_annotation_erasing_mask_and_conflicts(tmp_path):
    state = review.ClusterReview.from_prepared(prepared_dataset(tmp_path))
    state.items[0]['valid'][0, 0] = False
    state.annotate(0, 0, [0, 1], 1)
    assert state.items[0]['annotations'][0].flat[0] == 0
    assert state.items[0]['annotations'][0].flat[1] == 1
    with pytest.raises(ValueError):
        state.annotate(0, 1, [1], 1)
    state.annotate(0, 0, [1], 0)
    state.annotate(0, 1, [1], 1)
    assert state.items[0]['annotations'][1].flat[1] == 1


def test_atomic_save_keeps_previous_project_on_failure(tmp_path, monkeypatch):
    prepared = prepared_dataset(tmp_path)
    prepared['review'] = review.ClusterReview.from_prepared(prepared)
    path = tmp_path / 'project.riima'
    review.save_project(prepared, path)
    previous = path.read_bytes()
    def fail(*args, **kwargs):
        raise OSError('disk full')
    monkeypatch.setattr(review.os, 'replace', fail)
    with pytest.raises(OSError):
        review.save_project(prepared, path)
    assert path.read_bytes() == previous


def test_reject_invalid_project(tmp_path):
    path = tmp_path / 'invalid.riima'
    with path.open('wb') as handle:
        np.savez(handle, metadata=np.array('{"version": 999}'))
    with pytest.raises(ValueError):
        review.load_project(path)


def test_approval_without_marks_preserves_regularized_initial_selection(tmp_path):
    prepared = prepared_dataset(tmp_path, count=1)
    state = review.ClusterReview.from_prepared(prepared)
    # A boundary pixel changed by the existing spatial regularizer differs from its center.
    item = state.items[0]
    target = int(item['initial'][8, 3])
    item['labels'][8, 20] = target
    mask = item['labels'] == target
    state.approve(target)
    np.testing.assert_array_equal(item['locked'] == target, mask)


def test_single_remaining_class_respects_negative_and_all_complete_is_safe(tmp_path):
    state = review.ClusterReview.from_prepared(prepared_dataset(tmp_path, count=1))
    first = int(state.items[0]['labels'][8, 3])
    other = 1 - first
    state.approve(first)
    state.annotate(0, other, [8 * 24 + 20], -1)
    state.refine(other)
    assert state.items[0]['labels'][8, 20] == -1
    assert state.items[0]['valid'][8, 20]
    state.approve(other)
    assert state.items[0]['labels'][8, 20] == -1
    assert state.completed == {0, 1}
    state.reopen(other)
    state.annotate(0, other, [8 * 24 + 20], 1)
    state.refine(other)
    assert state.items[0]['labels'][8, 20] == other


def test_registered_project_roundtrip_keeps_multimodal_feature_space(tmp_path):
    # A real registered analysis remains supported with the original 4N features.
    rng = np.random.default_rng(13)
    rgb = rng.integers(50, 220, (40, 48, 3), np.uint8)
    paths = []
    for i in range(2):
        path = tmp_path / f'view_{i}.png'
        cv2.imwrite(str(path), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
        paths.append(str(path))
    prepared = core.prepare_analysis(paths, str(tmp_path / 'out'),
                                     optical_cfg=core.Config(clusters=2, sample_pixels=300),
                                     mask_mode='overlap_only')
    state = prepared['review'] = review.ClusterReview.from_prepared(prepared)
    state.selected_image = 1
    state.approve(0)
    path = tmp_path / 'registered.riima'
    review.save_project(prepared, path)
    loaded = review.load_project(path)
    assert loaded['review'].mode == 'registered'
    assert len(loaded['review'].center) == 8
    assert len(loaded['optical']['images']) == 2
    np.testing.assert_array_equal(loaded['review'].items[0]['locked'], state.items[0]['locked'])
    loaded['review'].annotate(0, 1, [10], 1)
    loaded['review'].refine(1)


def test_unchanged_classes_retain_current_selection_when_another_is_refined(tmp_path):
    state = review.ClusterReview.from_prepared(prepared_dataset(tmp_path, count=1))
    item = state.items[0]
    first = int(item['initial'][8, 3])
    other = 1 - first
    # Preserve corrections/regularization unless this pixel is rejected or claimed elsewhere.
    initial = item['initial'].copy()
    initial[8, 20] = first
    initial.setflags(write=False)
    item['initial'] = initial
    item['labels'][8, 20] = first
    state.annotate(0, other, [8 * 24 + 22], 1)
    state.refine(other)
    assert item['labels'][8, 20] == first


def test_approval_freezes_displayed_mask_with_competing_learned_models(tmp_path):
    state = review.ClusterReview.from_prepared(prepared_dataset(tmp_path, count=1))
    for c, positive, negative in [(0, 212, 195), (1, 213, 196)]:
        state.annotate(0, c, [positive], 1)
        state.annotate(0, c, [negative], -1)
        state.refine(c)
    displayed = state.items[0]['labels'] == 0
    state.approve(0)
    np.testing.assert_array_equal(state.items[0]['locked'] == 0, displayed)


def test_erasing_all_examples_restores_initial_prediction(tmp_path):
    state = review.ClusterReview.from_prepared(prepared_dataset(tmp_path, count=1))
    c = int(state.items[0]['initial'][8, 3])
    state.annotate(0, c, [212], 1)
    state.annotate(0, c, [195], -1)
    state.refine(c)
    assert state.items[0]['labels'][8, 20] == c
    state.annotate(0, c, [212, 195], 0)
    state.refine(c)
    np.testing.assert_array_equal(state.items[0]['labels'], state.items[0]['initial'])


def test_pending_markings_are_applied_before_approval_and_after_resume(tmp_path):
    prepared = prepared_dataset(tmp_path, count=1)
    state = prepared['review']
    c = int(state.items[0]['initial'][8, 3])
    state.annotate(0, c, [212], 1)
    state.annotate(0, c, [195], -1)
    path = tmp_path / 'pending.riima'
    review.save_project(prepared, path)
    restored = review.load_project(path)['review']
    restored.refine_pending()
    restored.approve(c)
    assert restored.items[0]['locked'][8, 20] == c
    assert restored.items[0]['locked'][8, 3] != c


def test_resume_preserves_feature_statistics_and_registration_provenance(tmp_path):
    rng = np.random.default_rng(13)
    rgb = rng.integers(50, 220, (40, 48, 3), np.uint8)
    paths = []
    for i in range(2):
        path = tmp_path / f'view_{i}.png'
        cv2.imwrite(str(path), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
        paths.append(str(path))
    prepared = core.prepare_analysis(paths, str(tmp_path / 'out'),
                                     optical_cfg=core.Config(clusters=2, sample_pixels=300),
                                     mask_mode='overlap_only')
    prepared['review'] = review.ClusterReview.from_prepared(prepared)
    core.save_session_outputs(prepared)
    original_csv = (Path(prepared['output']) / 'cluster_statistics.csv').read_bytes()
    path = tmp_path / 'statistics.riima'
    review.save_project(prepared, path)
    loaded = review.load_project(path)
    core.save_session_outputs(loaded)
    assert (Path(prepared['output']) / 'cluster_statistics.csv').read_bytes() == original_csv
    np.testing.assert_array_equal(loaded['optical']['overlap'], prepared['optical']['overlap'])
    actual = loaded['optical']['registered_images'][1]['registration']
    original = prepared['optical']['registered_images'][1]['registration']
    for key, value in original.items():
        if isinstance(value, np.ndarray):
            np.testing.assert_array_equal(actual[key], value)
        else:
            assert actual[key] == value


def test_group_review_uses_union_and_persists_stable_identity(tmp_path):
    prepared = prepared_dataset(tmp_path)
    state = prepared['review']
    raw = [item['initial'].copy() for item in state.items]
    prepared['interpretation'].merge_clusters([0, 1])
    state.sync_groups(prepared['interpretation'].groups)
    assert state.class_ids == [0]
    assert state.groups == {0: {0, 1}}
    for item in state.items:
        assert np.all(item['labels'][item['valid']] == 0)
    state.annotate(0, 0, [212], 1)
    state.annotate(0, 0, [195], -1)
    state.refine_pending()
    assert state.items[0]['labels'][8, 20] == 0
    assert state.items[0]['labels'][8, 3] == -1
    assert state.items[1]['labels'][8, 23] == 0
    state.approve(0)
    path = tmp_path / 'group.riima'
    review.save_project(prepared, path)
    loaded = review.load_project(path)['review']
    assert loaded.groups == {0: {0, 1}}
    assert loaded.completed == {0}
    for item, initial in zip(loaded.items, raw):
        np.testing.assert_array_equal(item['initial'], initial)


def test_grouping_cannot_discard_existing_annotations_or_protected_masks(tmp_path):
    state = prepared_dataset(tmp_path)['review']
    state.annotate(0, 0, [212], 1)
    with pytest.raises(ValueError, match='revis'):
        state.sync_groups([{0, 1}])
    assert state.groups == {0: {0}, 1: {1}}
    state.annotate(0, 0, [212], 0)
    state.refine(0)
    state.approve(0)
    with pytest.raises(ValueError, match='revis'):
        state.sync_groups([{0, 1}])


def test_approval_requires_explicit_update_for_pending_marks(tmp_path):
    state = prepared_dataset(tmp_path)['review']
    state.annotate(0, 0, [212], 1)
    before = state.items[0]['labels'].copy()
    with pytest.raises(ValueError, match='Atualiz'):
        state.approve(0)
    np.testing.assert_array_equal(state.items[0]['labels'], before)
    assert not state.completed
    state.refine_pending()
    state.approve(0)


def test_manual_update_trains_all_pending_groups_once(tmp_path, monkeypatch):
    state = prepared_dataset(tmp_path)['review']
    for c, positive, negative in [(0, 212, 195), (1, 196, 213)]:
        state.annotate(0, c, [positive], 1)
        state.annotate(0, c, [negative], -1)
    calls = []
    original = state._predict
    def counted(*args, **kwargs):
        calls.append(1)
        return original(*args, **kwargs)
    monkeypatch.setattr(state, '_predict', counted)
    state.refine_pending()
    assert calls == [1]
    assert state.dirty == set()
    assert set(state.models) == {0, 1}


def test_subsequent_strokes_reuse_annotation_buffer(tmp_path, monkeypatch):
    state = prepared_dataset(tmp_path)['review']
    state.annotate(0, 0, [212], 1)
    buffer = state.items[0]['annotations'][0]
    original = np.zeros
    def guarded(shape, *args, **kwargs):
        if shape == buffer.shape:
            pytest.fail('full annotation buffer allocated on every stroke')
        return original(shape, *args, **kwargs)
    monkeypatch.setattr(np, 'zeros', guarded)
    state.annotate(0, 0, [213], -1)
    assert state.items[0]['annotations'][0] is buffer


def test_regrouping_remaining_classes_never_reintroduces_approved_negative_pixels(tmp_path):
    state = prepared_dataset(tmp_path, count=1, clusters=3)['review']
    target = int(state.items[0]['initial'][8, 3])
    state.annotate(0, target, [195], -1)
    state.refine_pending()
    state.approve(target)
    item = state.items[0]
    assert item['locked'].flat[195] == -1
    assert item['labels'].flat[195] != target
    protected = item['locked'].copy()
    others = set(range(3)) - {target}
    state.sync_groups([{target}, others])
    assert item['labels'].flat[195] != target
    assert not np.any((item['labels'] == target) & (item['locked'] < 0))
    np.testing.assert_array_equal(item['locked'], protected)


def test_model_precedence_survives_resume_and_updates_of_other_groups(tmp_path):
    prepared = prepared_dataset(tmp_path, count=1, clusters=3)
    state = prepared['review']
    for c, positive, negative in [(0, 212, 195), (1, 213, 196)]:
        state.annotate(0, c, [positive], 1)
        state.annotate(0, c, [negative], -1)
    state.refine_pending(preferred=0)
    displayed = state.items[0]['labels'].copy()
    path = tmp_path / 'precedence.riima'
    review.save_project(prepared, path)
    restored = review.load_project(path)['review']
    assert list(restored.models) == [1, 0]
    restored.annotate(0, 2, [194], -1)
    # Navigating to an unchanged group before clicking Update must not reprioritize it.
    restored.refine_pending(preferred=1)
    # An unrelated negative cannot change the winner between these learned groups.
    protected_choices = np.isin(displayed, [0, 1])
    np.testing.assert_array_equal(restored.items[0]['labels'][protected_choices],
                                  displayed[protected_choices])


def test_background_exclusion_is_manual_global_and_resumable(tmp_path):
    prepared = prepared_dataset(tmp_path)
    state = prepared['review']
    originals = [item['rgb'].copy() for item in state.items]
    before = state.items[0]['labels'].copy()
    state.exclude(0, [195], True)
    assert state.has_pending
    np.testing.assert_array_equal(state.items[0]['labels'], before)
    with pytest.raises(ValueError, match='Atualiz'):
        state.approve(0)
    path = tmp_path / 'pending-background.riima'
    review.save_project(prepared, path)
    state = review.load_project(path)['review']
    assert state.has_pending and state.items[0]['excluded'].flat[195]
    state.refine_pending()
    assert not state.has_pending
    assert state.items[0]['labels'].flat[195] == -1
    state.annotate(0, 1, [195, 212], 1)
    state.refine_pending()
    assert state.items[0]['labels'].flat[195] == -1
    prepared['review'] = state
    review.save_project(prepared, path)
    restored = review.load_project(path)['review']
    assert restored.items[0]['applied_excluded'].flat[195]
    restored.annotate(1, 0, [10], -1)
    restored.refine_pending()
    assert restored.items[0]['labels'].flat[195] == -1
    for item, original in zip(restored.items, originals):
        np.testing.assert_array_equal(item['rgb'], original)


def test_background_cannot_change_approved_pixels_and_can_be_erased(tmp_path):
    state = prepared_dataset(tmp_path)['review']
    state.approve(0)
    locked = state.items[0]['locked'].copy()
    protected = np.flatnonzero(locked >= 0)
    free = np.flatnonzero(locked < 0)[0]
    state.exclude(0, np.r_[protected, free], True)
    state.refine_pending()
    np.testing.assert_array_equal(state.items[0]['locked'], locked)
    assert not np.any(state.items[0]['excluded'].ravel()[protected])
    assert state.items[0]['labels'].flat[free] == -1
    state.exclude(0, [free], False)
    state.refine_pending()
    assert state.items[0]['labels'].flat[free] == 1
    np.testing.assert_array_equal(state.items[0]['locked'], locked)


def test_undo_is_one_stroke_across_segments_and_after_manual_update(tmp_path):
    state = prepared_dataset(tmp_path)['review']
    state.begin_stroke()
    state.annotate(0, 0, [195, 196], -1)
    state.annotate(0, 0, [196, 197], -1)
    state.end_stroke()
    state.begin_stroke()
    state.exclude(0, [212], True)
    state.end_stroke()
    assert state.undo()
    assert not state.items[0]['excluded'].flat[212]
    assert state.dirty == {0}
    state.refine_pending()
    applied = state.items[0]['labels'].copy()
    assert state.undo()
    assert not np.any(state.items[0]['annotations'][0])
    assert state.has_pending
    np.testing.assert_array_equal(state.items[0]['labels'], applied)
    state.refine_pending()
    assert not state.has_pending
    assert not state.undo()


def test_background_only_update_keeps_model_precedence_and_failed_update_is_pending(tmp_path, monkeypatch):
    state = prepared_dataset(tmp_path, clusters=3)['review']
    for c, positive, negative in [(0, 212, 195), (1, 213, 196)]:
        state.annotate(0, c, [positive], 1)
        state.annotate(0, c, [negative], -1)
    state.refine_pending(preferred=0)
    original_predict = state._predict
    previous_models = dict(state.models)
    before = state.items[0]['labels'].copy()
    state.exclude(0, [194], True)
    def fail(**kwargs):
        raise RuntimeError('failed prediction')
    monkeypatch.setattr(state, '_predict', fail)
    with pytest.raises(RuntimeError):
        state.refine_pending(preferred=1)
    np.testing.assert_array_equal(state.items[0]['labels'], before)
    assert state.models == previous_models
    assert state.has_pending
    assert not state.items[0]['applied_excluded'].flat[194]
    monkeypatch.setattr(state, '_predict', original_predict)
    state.refine_pending(preferred=1)
    assert list(state.models) == [1, 0]


def test_old_project_migration_hides_applied_examples_but_keeps_pending_visible(tmp_path):
    prepared = prepared_dataset(tmp_path)
    state = prepared['review']
    state.annotate(0, 0, [195], -1)
    state.refine_pending()
    state.annotate(0, 1, [212], -1)
    path = tmp_path / 'legacy.riima'
    review.save_project(prepared, path)
    with np.load(path, allow_pickle=False) as archive:
        arrays = {key: archive[key].copy() for key in archive.files}
    meta = json.loads(str(arrays['metadata']))
    meta['version'] = 2
    # Construct the genuine historical per-image sample layout.
    for i in range(1, len(meta['items'])):
        arrays[f'{i}_sample_X'] = arrays['0_sample_X'].copy()
    meta.pop('exclusion_dirty')
    arrays = {key: value for key, value in arrays.items()
              if not ('applied_annotation' in key or key.endswith('_excluded'))}
    arrays['metadata'] = np.array(json.dumps(meta))
    with path.open('wb') as handle:
        np.savez_compressed(handle, **arrays)
    restored = review.load_project(path)['review']
    assert restored.items[0]['applied_annotations'][0].flat[195] == -1
    assert restored.items[0]['applied_annotations'][1].flat[212] == 0
    assert restored.dirty == {1}


def test_legacy_erased_examples_remain_pending_after_unrelated_undo_and_resave(tmp_path):
    prepared = prepared_dataset(tmp_path)
    state = prepared['review']
    state.annotate(0, 0, [212], 1)
    state.annotate(0, 0, [195], -1)
    state.refine_pending()
    state.annotate(0, 0, [212, 195], 0)
    path = tmp_path / 'legacy-erased.riima'
    review.save_project(prepared, path)
    with np.load(path, allow_pickle=False) as archive:
        arrays = {key: archive[key].copy() for key in archive.files}
    meta = json.loads(str(arrays['metadata']))
    meta['version'] = 2
    for i in range(1, len(meta['items'])):
        arrays[f'{i}_sample_X'] = arrays['0_sample_X'].copy()
    arrays['metadata'] = np.array(json.dumps(meta))
    with path.open('wb') as handle:
        np.savez_compressed(handle, **arrays)
    loaded = review.load_project(path)
    restored = loaded['review']
    restored.exclude(0, [196], True)
    restored.undo()
    assert restored.dirty == {0}
    with pytest.raises(ValueError, match='Atualiz'):
        restored.approve(0)
    review.save_project(loaded, path)
    restored = review.load_project(path)['review']
    restored.exclude(0, [196], True)
    restored.undo()
    assert restored.dirty == {0}
    restored.refine_pending()
    assert restored.dirty == set()
    assert 0 not in restored.models


def test_entire_background_group_can_be_restored_after_save_and_never_competes(tmp_path):
    prepared = prepared_dataset(tmp_path)
    state = prepared['review']
    masks = [item['labels'] == 0 for item in state.items]
    state.set_background([0], True)
    assert state.has_pending
    state.refine_pending()
    for item, mask in zip(state.items, masks):
        assert np.all(item['labels'][mask] == -1)
        assert np.all(item['excluded'][mask])
    state.annotate(0, 1, [212], -1)
    state.refine_pending()
    assert not any(np.any(item['labels'] == 0) for item in state.items)
    path = tmp_path / 'background-group.riima'
    review.save_project(prepared, path)
    restored = review.load_project(path)['review']
    assert restored.background_groups == {0}
    restored.set_background([0], False)
    restored.refine_pending()
    assert restored.background_groups == set()
    for item, mask in zip(restored.items, masks):
        assert not np.any(item['excluded'][mask])
        assert np.all(item['labels'][mask] == 0)


def test_reset_group_clears_learning_without_changing_other_approved_masks(tmp_path):
    state = prepared_dataset(tmp_path)['review']
    state.approve(1)
    state.annotate(0, 0, [195], -1)
    state.refine_pending()
    state.exclude(0, [196], True)
    state.refine_pending()
    locks = [item['locked'].copy() for item in state.items]
    labels = [item['labels'].copy() for item in state.items]
    state.reset_group(0)
    for item, before in zip(state.items, labels):
        np.testing.assert_array_equal(item['labels'], before)
    assert state.has_pending
    state.refine_pending()
    assert 0 not in state.models
    assert not np.any(state.items[0]['annotations'][0])
    assert state.items[0]['labels'].flat[196] == -1
    assert state.items[0]['labels'].flat[195] == 0
    for item, before in zip(state.items, locks):
        np.testing.assert_array_equal(item['locked'], before)
    with pytest.raises(ValueError, match='Reabr'):
        state.reset_group(1)


def test_background_operation_requires_reopen_for_approved_group(tmp_path):
    state = prepared_dataset(tmp_path)['review']
    state.approve(0)
    with pytest.raises(ValueError, match='Reabr'):
        state.set_background([0], True)
    assert not state.has_pending
    assert state.background_groups == set()


def test_background_history_with_restored_and_approved_pixels_can_resume_safely(tmp_path):
    prepared = prepared_dataset(tmp_path)
    state = prepared['review']
    target = int(state.items[0]['labels'].flat[195])
    state.set_background([target], True)
    state.refine_pending()
    state.exclude(0, [195], False)
    state.refine_pending()
    other = int(state.items[0]['labels'].flat[195])
    assert other != target and other >= 0
    state.approve(other)
    path = tmp_path / 'restored-approved.riima'
    review.save_project(prepared, path)
    restored = review.load_project(path)['review']
    locks = restored.items[0]['locked'].copy()
    restored.set_background([target], False)
    restored.refine_pending()
    np.testing.assert_array_equal(restored.items[0]['locked'], locks)
    assert restored.items[0]['labels'].flat[195] == other


def test_reset_and_restore_group_preserve_manual_background_marked_later(tmp_path):
    prepared = prepared_dataset(tmp_path)
    state = prepared['review']
    target = int(state.items[0]['labels'].flat[195])
    state.set_background([target], True)
    state.refine_pending()
    state.exclude(0, [195], False)
    state.refine_pending()
    state.exclude(0, [195], True)
    state.refine_pending()
    path = tmp_path / 'manual-background.riima'
    review.save_project(prepared, path)
    restored = review.load_project(path)['review']
    restored.reset_group(target)
    restored.refine_pending()
    assert restored.items[0]['excluded'].flat[195]
    assert restored.items[0]['labels'].flat[195] == -1


def test_manual_background_ownership_and_undo_survive_group_restoration(tmp_path):
    state = prepared_dataset(tmp_path)['review']
    target = int(state.items[0]['labels'].flat[195])
    state.set_background([target], True)
    state.refine_pending()
    state.begin_stroke()
    state.exclude(0, [195], True)  # Reaffirming a group-excluded pixel makes it manual.
    state.end_stroke()
    assert not state.items[0]['background_masks'][target].flat[195]
    state.undo()
    assert state.items[0]['background_masks'][target].flat[195]
    state.set_background([target], False)
    state.refine_pending()
    assert not state.items[0]['excluded'].flat[195]


def rewrite_project(path, edit):
    """Change a numeric project fixture without using pickle or production helpers."""
    with np.load(path, allow_pickle=False) as archive:
        arrays = {key: archive[key].copy() for key in archive.files}
    metadata = json.loads(str(arrays['metadata']))
    edit(arrays, metadata)
    arrays['metadata'] = np.array(json.dumps(metadata))
    with path.open('wb') as handle:
        np.savez_compressed(handle, **arrays)


@pytest.mark.parametrize('key', ['labels', 'valid', 'centers', 'center', 'scale',
                                  'confidence', 'sample_X', 'reference', 'unknown'])
def test_load_rejects_reserved_or_unknown_optional_arrays(tmp_path, key):
    prepared = prepared_dataset(tmp_path, count=1)
    path = tmp_path / 'reserved-array.riima'
    review.save_project(prepared, path)
    def edit(arrays, meta):
        meta['items'][0]['optional_arrays'].append(key)
        arrays.setdefault(f'0_{key}', np.zeros(1))
    rewrite_project(path, edit)
    with pytest.raises(ValueError, match='opciona'):
        review.load_project(path)


@pytest.mark.parametrize('keys', [None, 'palette', {'palette': True}, [None],
                                  [['palette']], ['palette', 'palette']])
def test_load_rejects_malformed_optional_array_list(tmp_path, keys):
    path = tmp_path / 'optional-list.riima'
    review.save_project(prepared_dataset(tmp_path, count=1), path)
    rewrite_project(path, lambda arrays, meta: meta['items'][0].update(optional_arrays=keys))
    with pytest.raises(ValueError, match='opciona'):
        review.load_project(path)


@pytest.mark.parametrize('version', [1, 2, 3, 4])
def test_optional_arrays_and_workspace_roundtrip_preserve_science(tmp_path, version):
    prepared = prepared_dataset(tmp_path, count=1)
    optical = prepared['optical']
    optical['feature_means'] = np.arange(8, dtype=np.float32).reshape(2, 4)
    optical['overlap'] = np.ones((16, 24), bool)
    optical['palette'] = np.array([[10, 20, 30], [40, 50, 60]], np.uint8)
    prepared['workspace'] = dict(selected_groups=[1], opacity=0, split_fraction=1,
                                 step='organize', display='Máscara selecionada', view_type='compare',
                                 isolate=True, contours=False, show_applied=False, automatic_output=True,
                                 compare_left='Imagem original', compare_right='Classificação atual',
                                 legacy_setting={'name': 'preserved'})
    path = tmp_path / 'valid-optionals.riima'
    review.save_project(prepared, path)
    rewrite_project(path, lambda arrays, meta: meta.update(version=version))
    loaded = review.load_project(path)
    assert loaded['workspace'] == prepared['workspace']
    assert loaded['interpretation'].to_dict() == prepared['interpretation'].to_dict()
    for key in ('feature_means', 'overlap', 'palette', 'labels', 'valid', 'centers', 'center', 'scale'):
        np.testing.assert_array_equal(loaded['optical'][key], optical[key])
    np.testing.assert_array_equal(loaded['review'].items[0]['labels'], prepared['review'].items[0]['labels'])
    assert not loaded['review'].items[0]['initial'].flags.writeable


@pytest.mark.parametrize('key, replacement', [
    ('feature_means', np.zeros((2, 4, 1), np.float32)),
    ('feature_means', np.zeros((2, 4), np.complex64)),
    ('feature_means', np.full((2, 4), np.inf)),
    ('palette', np.zeros((2, 3), np.float32)),
    ('overlap', np.ones((16, 24, 1), bool)),
    ('overlap', np.zeros((16, 24), bool)),
])
def test_optional_arrays_require_compatible_types_and_support(tmp_path, key, replacement):
    prepared = prepared_dataset(tmp_path, count=1)
    prepared['optical']['feature_means'] = np.zeros((2, 4), np.float32)
    prepared['optical']['overlap'] = np.ones((16, 24), bool)
    path = tmp_path / 'optional-geometry.riima'
    review.save_project(prepared, path)
    rewrite_project(path, lambda arrays, meta: arrays.update({f'0_{key}': replacement}))
    with pytest.raises(ValueError, match='[Aa]tribut|[Pp]aleta|[Mm]áscara'):
        review.load_project(path)


@pytest.mark.parametrize('workspace', [
    None, [], 'workspace', {'selected_groups': None}, {'selected_groups': '0'},
    {'selected_groups': [True]}, {'selected_groups': [0.5]},
    {'opacity': None}, {'opacity': '50'}, {'opacity': -1}, {'opacity': 101},
    {'opacity': float('nan')}, {'opacity': float('inf')},
    {'split_fraction': float('-inf')}, {'split_fraction': -0.01}, {'split_fraction': 1.01},
    {'isolate': None}, {'contours': 'false'}, {'show_applied': 1},
    {'automatic_output': []}, {'step': []}, {'display': 'unknown'},
    {'view_type': {}}, {'compare_left': []}, {'compare_right': None},
])
def test_load_rejects_invalid_workspace_before_returning_project(tmp_path, workspace):
    path = tmp_path / 'invalid-workspace.riima'
    review.save_project(prepared_dataset(tmp_path, count=1), path)
    rewrite_project(path, lambda arrays, meta: meta.update(workspace=workspace))
    with pytest.raises(ValueError, match='[Ww]orkspace'):
        review.load_project(path)


@pytest.mark.parametrize('version', [1, 2, 3, 4])
def test_workspace_absent_in_historical_project_uses_defaults(tmp_path, version):
    path = tmp_path / 'legacy-workspace.riima'
    review.save_project(prepared_dataset(tmp_path, count=1), path)
    def edit(arrays, meta):
        meta['version'] = version
        meta.pop('workspace', None)
        meta['items'][0].pop('optional_arrays', None)
    rewrite_project(path, edit)
    loaded = review.load_project(path)
    assert loaded['workspace'] == {}
    assert loaded['review'].selected_cluster == 0


def test_workspace_normalizes_stale_selections_without_removing_legacy_fields(tmp_path):
    prepared = prepared_dataset(tmp_path, count=1)
    prepared['workspace'] = dict(selected_groups=[1, 77, 1, 0], opacity=100., split_fraction=0.,
                                 display='Grupo selecionado', compare_left='Obsolete source',
                                 legacy_setting={'name': 'preserved'})
    path = tmp_path / 'workspace-normalized.riima'
    review.save_project(prepared, path)
    loaded = review.load_project(path)
    assert loaded['workspace'] == dict(selected_groups=[1, 0], opacity=100., split_fraction=0.,
                                       display='Grupo selecionado', compare_left='Obsolete source',
                                       legacy_setting={'name': 'preserved'})
    assert prepared['workspace']['selected_groups'] == [1, 77, 1, 0]


@pytest.mark.parametrize('payload', [b'not a project', b'PK\x03\x04truncated archive'])
def test_corrupt_project_has_a_clear_validation_error(tmp_path, payload):
    path = tmp_path / 'corrupt.riima'
    path.write_bytes(payload)
    with pytest.raises(ValueError, match='Projeto RIIMA inv'):
        review.load_project(path)


def test_regrouping_waits_for_erased_applied_negative_to_be_updated(tmp_path):
    prepared = prepared_dataset(tmp_path, count=1, clusters=3)
    state = prepared['review']
    target = int(state.items[0]['initial'].flat[195])
    state.annotate(0, target, [195], -1)
    state.refine_pending()
    reassigned = int(state.items[0]['labels'].flat[195])
    merge_with = next(c for c in state.class_ids if c not in (target, reassigned))
    partition = [{target, merge_with}, {reassigned}]
    state.annotate(0, target, [195], 0)
    before = state.items[0]['labels'].copy()
    with pytest.raises(ValueError, match='Atualiz'):
        state.sync_groups(partition)
    assert state.dirty == {target}
    assert state.items[0]['applied_annotations'][target].flat[195] == -1
    np.testing.assert_array_equal(state.items[0]['labels'], before)
    state.refine_pending()
    state.sync_groups(partition)
    assert state.items[0]['labels'].flat[195] == min(target, merge_with)
    assert not state.has_pending


def test_regrouping_waits_for_empty_group_reset_to_be_updated(tmp_path):
    state = prepared_dataset(tmp_path, count=1)['review']
    state.reset_group(0)
    with pytest.raises(ValueError, match='Atualiz'):
        state.sync_groups([{0, 1}])
    assert state.groups == {0: {0}, 1: {1}}
    assert state.dirty == {0}
    state.refine_pending()
    state.sync_groups([{0, 1}])
    assert state.groups == {0: {0, 1}}


def test_load_rejects_approved_pixels_without_their_locks(tmp_path):
    prepared = prepared_dataset(tmp_path, count=1)
    prepared['review'].approve(0)
    path = tmp_path / 'missing-locks.riima'
    review.save_project(prepared, path)
    rewrite_project(path, lambda arrays, meta: arrays['0_locked'].fill(-1))
    before = prepared['review'].items[0]['locked'].copy()
    with pytest.raises(ValueError, match='aprovad'):
        review.load_project(path)
    assert prepared['review'].completed == {0}
    np.testing.assert_array_equal(prepared['review'].items[0]['locked'], before)


def test_load_rejects_current_labels_of_removed_group_ids(tmp_path):
    prepared = prepared_dataset(tmp_path, count=1, clusters=3)
    prepared['interpretation'].merge_clusters([0, 1])
    prepared['review'].sync_groups(prepared['interpretation'].groups)
    path = tmp_path / 'inactive-label.riima'
    review.save_project(prepared, path)
    rewrite_project(path, lambda arrays, meta: arrays['0_labels'].__setitem__((0, 0), 1))
    with pytest.raises(ValueError, match='[Rr]ótulos'):
        review.load_project(path)


@pytest.mark.parametrize('key', ['initial', 'labels', 'locked'])
@pytest.mark.parametrize('replacement', ['out-of-range', 'fractional'])
def test_load_rejects_invalid_label_values_before_casting(tmp_path, key, replacement):
    prepared = prepared_dataset(tmp_path, count=1)
    approved = int(prepared['review'].items[0]['labels'][0, 0])
    if key == 'locked':
        prepared['review'].approve(approved)
    path = tmp_path / 'invalid-labels.riima'
    review.save_project(prepared, path)
    def edit(arrays, meta):
        if replacement == 'fractional':
            arrays[f'0_{key}'] = arrays[f'0_{key}'].astype(np.float32)
            arrays[f'0_{key}'][0, 0] = approved + 0.5
        else:
            arrays[f'0_{key}'] = arrays[f'0_{key}'].astype(np.int64)
            arrays[f'0_{key}'][0, 0] = 2 ** 32 + approved
    rewrite_project(path, edit)
    with pytest.raises(ValueError, match='[Rr]ótulos'):
        review.load_project(path)


def test_load_rejects_unapplied_annotations_without_pending_flag(tmp_path):
    prepared = prepared_dataset(tmp_path, count=1)
    prepared['review'].annotate(0, 0, [195], -1)
    path = tmp_path / 'missing-pending.riima'
    review.save_project(prepared, path)
    rewrite_project(path, lambda arrays, meta: meta.update(dirty=[]))
    with pytest.raises(ValueError, match='[Pp]end'):
        review.load_project(path)


def test_load_rejects_pending_changes_to_an_approved_group(tmp_path):
    prepared = prepared_dataset(tmp_path, count=1)
    prepared['review'].approve(0)
    path = tmp_path / 'approved-pending.riima'
    review.save_project(prepared, path)
    rewrite_project(path, lambda arrays, meta: meta.update(dirty=[0]))
    with pytest.raises(ValueError, match='[Pp]end'):
        review.load_project(path)


def test_load_rejects_unapplied_exclusions_without_pending_flag(tmp_path):
    prepared = prepared_dataset(tmp_path, count=1)
    prepared['review'].exclude(0, [195], True)
    path = tmp_path / 'missing-background-pending.riima'
    review.save_project(prepared, path)
    rewrite_project(path, lambda arrays, meta: meta.update(exclusion_dirty=False))
    with pytest.raises(ValueError, match='[Pp]end'):
        review.load_project(path)


def test_load_rejects_background_group_without_recovery_masks(tmp_path):
    prepared = prepared_dataset(tmp_path, count=1)
    prepared['review'].set_background([0], True)
    prepared['review'].refine_pending()
    path = tmp_path / 'missing-background-history.riima'
    review.save_project(prepared, path)
    rewrite_project(path, lambda arrays, meta: meta['items'][0].update(background_masks=[]))
    with pytest.raises(ValueError, match='fundo'):
        review.load_project(path)


@pytest.mark.parametrize('field, value', [
    ('clusters', 1), ('clusters', 256), ('clusters', 2.5),
    ('sample_pixels', 1), ('chunk_pixels', 0), ('chunk_pixels', 1.5),
    ('kmeans_attempts', 0), ('seed', -1),
    ('feature_blur_sigma', -1), ('feature_blur_sigma', float('nan')),
])
def test_load_rejects_configuration_that_cannot_resume_prediction(tmp_path, field, value):
    prepared = prepared_dataset(tmp_path, count=1)
    path = tmp_path / 'invalid-config.riima'
    review.save_project(prepared, path)
    rewrite_project(path, lambda arrays, meta: meta['config'].update({field: value}))
    with pytest.raises(ValueError, match='[Cc]onfig'):
        review.load_project(path)


@pytest.mark.parametrize('key, replacement', [
    ('scale', np.zeros(4, np.float32)),
    ('scale', np.array([1, -1, 1, 1], np.float32)),
    ('scale', np.array([1, np.nan, 1, 1], np.float32)),
    ('scale', np.full(4, 1e-100, np.float64)),
    ('center', np.zeros((1, 4), np.float32)),
    ('center', np.array([1, np.inf, 1, 1], np.float32)),
    ('centers', np.zeros((1, 4), np.float32)),
    ('centers', np.zeros((2, 3), np.float32)),
    ('centers', np.full((2, 4), np.nan, np.float32)),
    ('0_sample_X', np.zeros((10, 3), np.float32)),
    ('0_sample_X', np.full((10, 4), np.inf, np.float32)),
])
def test_load_rejects_incompatible_or_nonfinite_feature_space(tmp_path, key, replacement):
    prepared = prepared_dataset(tmp_path, count=1)
    path = tmp_path / 'invalid-feature-space.riima'
    review.save_project(prepared, path)
    rewrite_project(path, lambda arrays, meta: arrays.update({key: replacement}))
    with pytest.raises(ValueError, match='[Aa]tribut|[Ee]scala'):
        review.load_project(path)


@pytest.mark.parametrize('key, replacement', [
    ('0_rgb', np.zeros((16, 24, 3), np.float32)),
    ('0_image_0', np.zeros((16, 24, 3), np.float32)),
    ('0_image_0', np.zeros((16, 23, 3), np.uint8)),
    ('0_valid', np.ones((16, 24), np.uint8)),
    ('0_confidence', np.ones((16, 23), np.float32)),
    ('0_confidence', np.full((16, 24), np.nan, np.float32)),
    ('0_confidence', np.full((16, 24), 2, np.float32)),
    ('palette', np.zeros((1, 3), np.uint8)),
    ('palette', np.zeros((2, 3), np.float32)),
])
def test_load_rejects_invalid_image_mask_confidence_or_palette(tmp_path, key, replacement):
    prepared = prepared_dataset(tmp_path, count=1)
    path = tmp_path / 'invalid-image-data.riima'
    review.save_project(prepared, path)
    rewrite_project(path, lambda arrays, meta: arrays.update({key: replacement}))
    with pytest.raises(ValueError, match='[Ii]magem|[Mm]áscara|[Cc]onfiança|[Pp]aleta'):
        review.load_project(path)


@pytest.mark.parametrize('metadata_change', [
    {'image_count': 0}, {'image_count': 1.5},
    {'image_names': []}, {'reference_index': 1}, {'mask_mode': 'unknown'},
])
def test_load_rejects_image_metadata_that_cannot_be_displayed(tmp_path, metadata_change):
    prepared = prepared_dataset(tmp_path, count=1)
    path = tmp_path / 'invalid-image-info.riima'
    review.save_project(prepared, path)
    rewrite_project(path, lambda arrays, meta: meta['items'][0].update(metadata_change))
    with pytest.raises(ValueError, match='[Ii]magem|[Mm]odo'):
        review.load_project(path)


@pytest.mark.parametrize('key, replacement', [
    ('feature_means', np.zeros((2, 3), np.float32)),
    ('feature_means', np.full((2, 4), np.nan, np.float32)),
    ('palette', np.zeros((1, 3), np.uint8)),
    ('overlap', np.ones((16, 23), bool)),
    ('overlap', np.ones((16, 24), np.uint8)),
])
def test_load_rejects_incompatible_optional_optical_arrays(tmp_path, key, replacement):
    prepared = prepared_dataset(tmp_path, count=1)
    prepared['optical']['feature_means'] = np.zeros((2, 4), np.float32)
    prepared['optical']['overlap'] = prepared['optical']['valid'].copy()
    path = tmp_path / 'invalid-optical-data.riima'
    review.save_project(prepared, path)
    rewrite_project(path, lambda arrays, meta: arrays.update({f'0_{key}': replacement}))
    with pytest.raises(ValueError, match='[Aa]tribut|[Pp]aleta|[Mm]áscara'):
        review.load_project(path)


@pytest.mark.parametrize('version', [1, 2, 3])
def test_legacy_projects_preserve_models_locks_metadata_and_pending_evidence(tmp_path, version):
    prepared = prepared_dataset(tmp_path, count=1)
    state = prepared['review']
    prepared['interpretation'].rename_cluster(1, 'Legacy name')
    prepared['interpretation'].recolor_cluster(1, (13, 29, 71))
    state.palette[1] = (13, 29, 71)
    state.annotate(0, 0, [212], 1)
    state.annotate(0, 0, [195], -1)
    state.refine_pending()
    state.approve(0)
    state.annotate(0, 1, [194], -1)
    state.selected_cluster = 1
    path = tmp_path / 'legacy-roundtrip.riima'
    review.save_project(prepared, path)
    def edit(arrays, meta):
        meta['version'] = version
        if version < 3:
            meta.pop('exclusion_dirty', None)
            meta.pop('unresolved_dirty', None)
            for key in list(arrays):
                if 'applied_annotation' in key or key.endswith('_excluded'):
                    arrays.pop(key)
    rewrite_project(path, edit)
    loaded = review.load_project(path)
    restored = loaded['review']
    assert loaded['interpretation'].to_dict() == prepared['interpretation'].to_dict()
    assert restored.completed == {0}
    assert restored.dirty == {1}
    assert restored.models[0].isTrained()
    assert restored.selected_cluster == 1
    np.testing.assert_array_equal(restored.palette, state.palette)
    np.testing.assert_array_equal(restored.items[0]['locked'], state.items[0]['locked'])
    assert restored.items[0]['annotations'][1].flat[194] == -1
    assert restored.items[0]['applied_annotations'][1].flat[194] == 0
    restored.refine_pending()
    np.testing.assert_array_equal(restored.items[0]['locked'], state.items[0]['locked'])
    review.save_project(loaded, path)
    assert not review.load_project(path)['review'].has_pending


@pytest.mark.parametrize('prior_model', [False, True])
def test_failed_single_group_refinement_preserves_model_and_pending_evidence(tmp_path, monkeypatch, prior_model):
    prepared = prepared_dataset(tmp_path, count=1)
    state = prepared['review']
    state.annotate(0, 0, [212], 1)
    state.annotate(0, 0, [195], -1)
    if prior_model:
        state.refine(0)
        state.approve(1)
        state.annotate(0, 0, [195], 1)
        state.annotate(0, 0, [212], -1)
    before_models = dict(state.models)
    before_labels = state.items[0]['labels'].copy()
    before_locks = state.items[0]['locked'].copy()
    before_applied = state.items[0]['applied_annotations'].get(0, np.zeros((16, 24), np.int8)).copy()
    def fail(**kwargs):
        raise RuntimeError('failed prediction')
    monkeypatch.setattr(state, '_predict', fail)
    with pytest.raises(RuntimeError, match='failed prediction'):
        state.refine(0)
    assert state.models == before_models
    assert state.dirty == {0}
    np.testing.assert_array_equal(state.items[0]['labels'], before_labels)
    np.testing.assert_array_equal(state.items[0]['locked'], before_locks)
    np.testing.assert_array_equal(state.items[0]['applied_annotations'].get(0, np.zeros((16, 24), np.int8)),
                                  before_applied)
    path = tmp_path / 'failed-refinement.riima'
    review.save_project(prepared, path)
    assert review.load_project(path)['review'].has_pending
