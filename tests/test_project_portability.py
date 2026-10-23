"""Portable destinations and one stored independent training sample."""
import json
from pathlib import Path
import shutil

import numpy as np
import pytest

from riima import core, review
from test_riima_review import prepared_dataset, rewrite_project


@pytest.mark.parametrize('mode', ['independent', 'registered'])
@pytest.mark.parametrize('mask_mode', ['auto', 'overlap_only'])
def test_mask_view_roundtrip_preserves_saved_analysis_and_manual_background(tmp_path, mode, mask_mode):
    source = prepared_dataset(tmp_path, count=1)
    paths = [source['review'].items[0]['path']]
    cfg = core.Config(clusters=2, feature_blur_sigma=0, sample_pixels=300, chunk_pixels=99)
    if mode == 'independent':
        prepared = review.prepare_independent_analysis(paths, str(tmp_path / 'mask-out'), cfg,
                                                      mask_mode=mask_mode)
    else:
        prepared = core.prepare_analysis(paths, str(tmp_path / 'mask-out'), optical_cfg=cfg,
                                         mask_mode=mask_mode, publish_outputs=False)
        prepared['review'] = review.ClusterReview.from_prepared(prepared)
    state = prepared['review']
    state.exclude(0, [0])
    state.refine_pending()
    state.exclude(0, [0], False)
    state.exclude(0, [1])
    state.refine_pending()
    prepared['workspace'] = {'step': 'workbench', 'display': 'Máscara', 'selected_groups': [0]}
    path = tmp_path / 'mask-view.riima'
    review.save_project(prepared, path)

    loaded = review.load_project(path)

    assert loaded['workspace']['display'] == 'Máscara'
    assert loaded['workspace']['step'] == 'workbench'
    assert loaded['review'].mode == mode
    actual, expected = loaded['review'].items[0], state.items[0]
    assert actual['optical']['mask_mode'] == mask_mode
    assert not actual['excluded'].ravel()[0] and actual['excluded'].ravel()[1]
    for key in ('valid', 'excluded', 'applied_excluded', 'display_valid', 'labels', 'initial'):
        np.testing.assert_array_equal(actual[key], expected[key])


@pytest.mark.parametrize('count,clusters,budget', [(3, 2, 5), (4, 3, 10), (6, 2, 5)])
def test_independent_impossible_budget_is_rejected_before_read_or_callback(tmp_path, monkeypatch, count, clusters, budget):
    monkeypatch.setattr(core, 'read_rgb', lambda *args: pytest.fail('image processing before budget validation'))
    with pytest.raises(ValueError, match='amostra|orçamento'):
        review.prepare_independent_analysis(['absent.png'] * count, tmp_path / 'out',
            core.Config(clusters=clusters, sample_pixels=budget),
            stage_callback=lambda *args: pytest.fail('callback before budget validation'))


@pytest.mark.parametrize('budget', [4, 5, 10, 1000])
def test_valid_budget_retains_per_field_sampling_policy(tmp_path, budget):
    seed = prepared_dataset(tmp_path)
    paths = [item['path'] for item in seed['review'].items]
    cfg = core.Config(clusters=2, sample_pixels=budget, feature_blur_sigma=0)
    actual = review.prepare_independent_analysis(paths, tmp_path / 'out', cfg, mask_mode='overlap_only')
    expected = sum(min(budget // 2, item['valid'].sum()) for item in seed['review'].items)
    assert len(actual['optical']['sample_X']) == expected <= budget


@pytest.mark.parametrize('legacy_version', [None, 1, 2, 3, 4])
@pytest.mark.parametrize('keep_original', [False, True])
def test_copy_or_move_never_restores_old_destination(tmp_path, monkeypatch, legacy_version, keep_original):
    source = tmp_path / 'original'
    source.mkdir()
    prepared = prepared_dataset(source)
    original_output = Path(prepared['output'])
    original_output.mkdir()
    (original_output / 'keep.txt').write_text('original')
    path = source / 'project.riima'
    review.save_project(prepared, path)
    if legacy_version:
        def legacy(arrays, meta):
            meta['version'] = legacy_version
            meta.pop('sample_layout', None)
            meta.pop('output_history', None)
            arrays['1_sample_X'] = arrays['0_sample_X'].copy()
        rewrite_project(path, legacy)
    destination = tmp_path / 'moved'
    destination.mkdir()
    moved = destination / path.name
    shutil.copyfile(path, moved)
    if not keep_original:
        shutil.rmtree(source)
    unrelated_cwd = tmp_path / 'cwd'
    unrelated_cwd.mkdir()
    monkeypatch.chdir(unrelated_cwd)
    loaded = review.load_project(moved)
    assert Path(loaded['output']).parent == destination
    assert loaded['output_history'] == [str(original_output)]
    review.save_review_outputs(loaded)
    assert (Path(loaded['output']) / 'review').is_dir()
    if keep_original:
        assert sorted(p.name for p in original_output.iterdir()) == ['keep.txt']
    else:
        assert not source.exists()


def test_resume_skips_existing_destinations_and_preserves_history_on_resave(tmp_path):
    prepared = prepared_dataset(tmp_path)
    path = tmp_path / 'project.riima'
    review.save_project(prepared, path)
    first = review.load_project(path)
    Path(first['output']).mkdir()
    (Path(first['output']) / 'keep').write_text('do not overwrite')
    second = review.load_project(path)
    assert first['output'] != second['output']
    assert not Path(second['output']).exists()
    review.save_project(first, path)
    third = review.load_project(path)
    assert third['output_history'] == [prepared['output'], first['output']]
    assert (Path(first['output']) / 'keep').read_text() == 'do not overwrite'


def test_new_archive_and_roundtrip_share_only_one_sample(tmp_path):
    prepared = prepared_dataset(tmp_path, count=3)
    path = tmp_path / 'shared.riima'
    review.save_project(prepared, path)
    with np.load(path, allow_pickle=False) as archive:
        meta = json.loads(str(archive['metadata']))
        assert meta['version'] == 6
        assert meta['sample_layout'] == 'shared'
        assert [key for key in archive.files if key.endswith('_sample_X')] == ['0_sample_X']
    loaded = review.load_project(path)
    samples = [item['optical']['sample_X'] for item in loaded['review'].items]
    assert all(sample is samples[0] for sample in samples)
    assert sum(sample.nbytes for sample in {id(x): x for x in samples}.values()) == samples[0].nbytes
    np.testing.assert_array_equal(samples[0], prepared['optical']['sample_X'])
    loaded['review'].annotate(0, 0, [195], -1)
    loaded['review'].annotate(0, 0, [212], 1)
    loaded['review'].refine_pending()
    review.save_project(loaded, path)
    restored = review.load_project(path)
    assert restored['review'].models[0].isTrained()
    for original, actual in zip(loaded['review'].items, restored['review'].items):
        np.testing.assert_array_equal(original['labels'], actual['labels'])


@pytest.mark.parametrize('version', [1, 2, 3, 4])
def test_legacy_duplicate_samples_are_validated_and_reused(tmp_path, version):
    path = tmp_path / 'legacy.riima'
    review.save_project(prepared_dataset(tmp_path), path)
    def legacy(arrays, meta):
        meta['version'] = version
        meta.pop('sample_layout', None)
        arrays['1_sample_X'] = arrays['0_sample_X'].copy()
    rewrite_project(path, legacy)
    state = review.load_project(path)['review']
    assert state.items[0]['optical']['sample_X'] is state.items[1]['optical']['sample_X']
    rewrite_project(path, lambda arrays, meta: arrays.update({'1_sample_X': np.full((5, 4), np.inf)}))
    with pytest.raises(ValueError, match='[Aa]tribut'):
        review.load_project(path)


@pytest.mark.parametrize('corruption', ['missing', 'extra', 'wrong_layout', 'nonfinite', 'dimensions', 'too_short'])
def test_v5_shared_sample_corruption_is_rejected(tmp_path, corruption):
    path = tmp_path / 'bad.riima'
    review.save_project(prepared_dataset(tmp_path), path)
    def edit(arrays, meta):
        if corruption == 'missing':
            arrays.pop('0_sample_X')
        elif corruption == 'extra':
            arrays['1_sample_X'] = arrays['0_sample_X'].copy()
        elif corruption == 'wrong_layout':
            meta['sample_layout'] = 'per_item'
        else:
            arrays['0_sample_X'] = {'nonfinite': np.full((10, 4), np.inf),
                                   'dimensions': np.zeros((10, 3)), 'too_short': np.zeros((1, 4))}[corruption]
    rewrite_project(path, edit)
    with pytest.raises(ValueError, match='Projeto RIIMA inválido'):
        review.load_project(path)


@pytest.mark.parametrize('history', [None, '', {}, [None], [42], [''], [False]])
def test_v5_rejects_malformed_output_history_before_returning_state(tmp_path, history):
    path = tmp_path / 'history.riima'
    review.save_project(prepared_dataset(tmp_path), path)
    rewrite_project(path, lambda arrays, meta: meta.update(output_history=history))
    with pytest.raises(ValueError, match='Histórico'):
        review.load_project(path)


@pytest.mark.parametrize('field', ['output_history', 'sample_layout'])
def test_v5_requires_explicit_new_schema_metadata(tmp_path, field):
    path = tmp_path / 'missing-field.riima'
    review.save_project(prepared_dataset(tmp_path), path)
    rewrite_project(path, lambda arrays, meta: meta.pop(field))
    with pytest.raises(ValueError, match='Projeto RIIMA inválido'):
        review.load_project(path)


@pytest.mark.parametrize('version', [1, 2, 3, 4])
def test_distinct_legacy_samples_survive_new_format_resave(tmp_path, version):
    path = tmp_path / 'distinct.riima'
    review.save_project(prepared_dataset(tmp_path), path)
    def legacy(arrays, meta):
        meta['version'] = version
        meta.pop('sample_layout', None)
        arrays['1_sample_X'] = arrays['0_sample_X'].copy() + 1
    rewrite_project(path, legacy)
    prepared = review.load_project(path)
    samples = [item['optical']['sample_X'] for item in prepared['review'].items]
    assert samples[0] is not samples[1]
    review.save_project(prepared, path)
    with np.load(path, allow_pickle=False) as archive:
        assert json.loads(str(archive['metadata']))['sample_layout'] == 'per_item'
    restored = review.load_project(path)
    for original, item in zip(samples, restored['review'].items):
        np.testing.assert_array_equal(original, item['optical']['sample_X'])


def test_historical_output_that_matches_default_is_never_activated(tmp_path):
    prepared = prepared_dataset(tmp_path)
    prepared['output'] = str(tmp_path / 'project_output')
    path = tmp_path / 'project.riima'
    review.save_project(prepared, path)
    loaded = review.load_project(path)
    assert loaded['output'] == str(tmp_path / 'project_output_2')
    assert loaded['output_history'] == [prepared['output']]
    assert not Path(loaded['output']).exists()


def test_loading_in_readonly_location_never_attempts_to_create_directories(tmp_path, monkeypatch):
    path = tmp_path / 'readonly.riima'
    review.save_project(prepared_dataset(tmp_path), path)
    before = {entry.name: entry.read_bytes() for entry in tmp_path.iterdir() if entry.is_file()}
    monkeypatch.setattr(Path, 'mkdir', lambda *args, **kwargs: pytest.fail('directory creation while reading'))
    monkeypatch.setattr(review.tempfile, 'mkstemp', lambda *args, **kwargs: pytest.fail('temporary file while reading'))
    loaded = review.load_project(path)
    assert loaded['output_pending'] is True
    assert not Path(loaded['output']).exists()
    assert {entry.name: entry.read_bytes() for entry in tmp_path.iterdir() if entry.is_file()} == before


def test_v5_registered_sample_roundtrip_retains_feature_dimensions(tmp_path):
    seed = prepared_dataset(tmp_path, count=1)
    prepared = core.prepare_analysis([seed['review'].items[0]['path']], tmp_path / 'registered',
        optical_cfg=core.Config(clusters=2, sample_pixels=100), mask_mode='overlap_only')
    prepared['review'] = review.ClusterReview.from_prepared(prepared)
    path = tmp_path / 'registered.riima'
    review.save_project(prepared, path)
    with np.load(path, allow_pickle=False) as archive:
        meta = json.loads(str(archive['metadata']))
        assert meta['version'] == 6
        assert meta['sample_layout'] == 'per_item'
    loaded = review.load_project(path)
    assert loaded['analysis_mode'] == 'registered'
    np.testing.assert_array_equal(loaded['optical']['sample_X'], prepared['optical']['sample_X'])
    assert loaded['output_history'] == [str(tmp_path / 'registered')]


def test_compact_archive_removes_exact_duplicates_and_restores_independent_masks(tmp_path):
    prepared = prepared_dataset(tmp_path, count=1)
    item = prepared['review'].items[0]
    # These equal arrays have distinct editing roles and must not alias in memory.
    path = tmp_path / 'compact.riima'
    review.save_project(prepared, path)
    with np.load(path, allow_pickle=False) as archive:
        meta = json.loads(str(archive['metadata']))
        aliases = meta['array_aliases']
        assert '0_image_0' in aliases or '0_rgb' in aliases
        assert '0_applied_excluded' in aliases or '0_excluded' in aliases
        assert archive['0_labels'].dtype == np.int16
        assert len(archive.files) < 17
    loaded = review.load_project(path)['review'].items[0]
    for key in ('rgb', 'initial', 'labels', 'valid', 'locked', 'excluded', 'applied_excluded'):
        np.testing.assert_array_equal(loaded[key], item[key])
        assert loaded[key].dtype == item[key].dtype
    loaded['excluded'][0, 0] = True
    assert not loaded['applied_excluded'][0, 0]
    loaded['labels'][0, 0] = -1
    assert loaded['initial'][0, 0] >= 0


@pytest.mark.parametrize('corruption', ['missing', 'cycle', 'unknown', 'reserved', 'collision', 'bad_dtype'])
def test_compact_storage_rejects_invalid_descriptors(tmp_path, corruption):
    path = tmp_path / 'invalid-compact.riima'
    review.save_project(prepared_dataset(tmp_path, count=1), path)
    with np.load(path, allow_pickle=False) as archive:
        arrays = {key: archive[key].copy() for key in archive.files}
    meta = json.loads(str(arrays['metadata']))
    aliases = meta['array_aliases']
    alias = next(iter(aliases))
    if corruption == 'missing':
        aliases[alias] = 'absent'
    elif corruption == 'cycle':
        aliases[alias] = alias
    elif corruption == 'unknown':
        aliases['999_rgb'] = aliases[alias]
    elif corruption == 'reserved':
        aliases['metadata'] = aliases[alias]
    elif corruption == 'collision':
        arrays[alias] = arrays[aliases[alias]].copy()
    else:
        meta['integer_dtypes']['0_labels'] = 'float32'
    arrays['metadata'] = np.array(json.dumps(meta))
    with path.open('wb') as handle:
        np.savez_compressed(handle, **arrays)
    with pytest.raises(ValueError, match='Projeto RIIMA inválido'):
        review.load_project(path)


def test_compact_archive_keeps_float_bit_patterns_and_self_contained_images(tmp_path):
    prepared = prepared_dataset(tmp_path, count=1)
    optical = prepared['optical']
    optical['confidence'] = np.linspace(0, 1, optical['labels'].size, dtype=np.float64).reshape(optical['labels'].shape)
    optical['sample_X'] = optical['sample_X'].astype(np.float64)
    optical['sample_X'][0, 0] = np.nextafter(optical['sample_X'][0, 0], np.inf)
    prepared['workspace'] = {'selected_groups': [0], 'extension': {'author': 'petrographer'}}
    prepared['interpretation'].rename_cluster(0, 'Annotated material')
    path = tmp_path / 'self-contained.riima'
    review.save_project(prepared, path)
    for item in prepared['review'].items:
        Path(item['path']).unlink()
    loaded = review.load_project(path)
    for key in ('sample_X', 'confidence'):
        actual, expected = loaded['optical'][key], optical[key]
        assert actual.dtype == expected.dtype
        assert actual.tobytes() == expected.tobytes()
    np.testing.assert_array_equal(loaded['optical']['images'][0], optical['images'][0])
    assert loaded['workspace'] == prepared['workspace']
    assert loaded['interpretation'].to_dict() == prepared['interpretation'].to_dict()


@pytest.mark.parametrize('version', [1, 2, 3, 4, 5, 6])
def test_all_storage_schemas_keep_models_locks_and_pending_annotations(tmp_path, version):
    prepared = prepared_dataset(tmp_path, count=1)
    state = prepared['review']
    state.annotate(0, 0, [212], 1)
    state.annotate(0, 0, [195], -1)
    state.refine_pending()
    state.approve(0)
    state.annotate(0, 1, [194], -1)
    path = tmp_path / 'all-schemas.riima'
    review.save_project(prepared, path)
    if version < 6:
        # Build the real historical physical-array layout independently of the
        # production decoder, rather than relabelling compact members as legacy.
        with np.load(path, allow_pickle=False) as archive:
            arrays = {key: archive[key].copy() for key in archive.files}
        meta = json.loads(str(arrays.pop('metadata')))
        for key, target in meta.pop('array_aliases').items():
            arrays[key] = arrays[target].copy()
        for key, dtype in meta.pop('integer_dtypes').items():
            arrays[key] = arrays[key].astype(np.dtype(dtype))
        meta['version'] = version
        arrays['metadata'] = np.array(json.dumps(meta))
        with path.open('wb') as handle:
            np.savez_compressed(handle, **arrays)
    loaded = review.load_project(path)['review']
    assert loaded.completed == {0}
    assert loaded.dirty == {1}
    assert loaded.has_pending
    np.testing.assert_array_equal(loaded.items[0]['locked'], state.items[0]['locked'])
    np.testing.assert_array_equal(loaded.items[0]['labels'], state.items[0]['labels'])
    for group, marks in state.items[0]['annotations'].items():
        np.testing.assert_array_equal(loaded.items[0]['annotations'][group], marks)
    features = state._features(state.items[0], np.arange(384))
    np.testing.assert_array_equal(loaded.models[0].predict(features)[1], state.models[0].predict(features)[1])


@pytest.mark.parametrize('order', [None, {}, 0, [0, 0], [2], [True], ['0']])
def test_project_rejects_malformed_refinement_precedence(tmp_path, order):
    path = tmp_path / 'invalid-precedence.riima'
    review.save_project(prepared_dataset(tmp_path), path)
    rewrite_project(path, lambda arrays, meta: meta.update(refinement_order=order))
    with pytest.raises(ValueError, match='precedência'):
        review.load_project(path)


@pytest.mark.parametrize('version', [1, 2, 3, 4, 5, 6])
def test_missing_refinement_order_preserves_historical_precedence_without_hidden_prediction(tmp_path, version, monkeypatch):
    prepared = prepared_dataset(tmp_path)
    state = prepared['review']
    state.annotate(0, 0, [212], 1)
    state.annotate(0, 0, [195], -1)
    state.refine_pending()
    state.annotate(0, 1, [213], 1)
    state.refine_pending(preferred=1)
    path = tmp_path / 'historical-precedence.riima'
    review.save_project(prepared, path)
    expected = [item['labels'].copy() for item in state.items]
    def legacy(arrays, meta):
        meta['version'] = version
        meta.pop('refinement_order')
        if version < 5:
            arrays['1_sample_X'] = arrays['0_sample_X'].copy()
    rewrite_project(path, legacy)
    with monkeypatch.context() as context:
        context.setattr(review.ClusterReview, '_features', lambda *args: pytest.fail('hidden feature computation on opening'))
        loaded = review.load_project(path)['review']
    assert loaded.refinement_order == [1, 0]  # historical supports before binary forests
    for item, labels in zip(loaded.items, expected):
        np.testing.assert_array_equal(item['labels'], labels)
    loaded.exclude(0, [196], True)
    loaded.refine_pending()
    assert loaded.refinement_order == [1, 0]
    assert loaded.items[1]['labels'][8, 20] == 0


@pytest.mark.parametrize('corruption', ['duplicate_id', 'duplicate_member', 'bad_member', 'unknown_name',
                                         'active_retired', 'missing_provenance', 'bad_color', 'low_next_id'])
def test_v7_rejects_corrupted_stable_catalog(tmp_path, corruption):
    prepared = prepared_dataset(tmp_path)
    state = prepared['review']
    group = state.create_group('Extra')
    state.delete_group(group)
    state.delete_group(0)
    path = tmp_path / 'bad-catalog.riima'
    review.save_project(prepared, path)
    def corrupt(arrays, meta):
        if corruption == 'duplicate_id':
            meta['group_catalog'].append(dict(meta['group_catalog'][0]))
        elif corruption == 'duplicate_member':
            meta['group_catalog'][0]['members'] = [1, 1]
        elif corruption == 'bad_member':
            meta['group_catalog'][0]['members'] = [100]
        elif corruption == 'unknown_name':
            meta['group_names']['700'] = 'Órfão'
        elif corruption == 'active_retired':
            meta['retired_groups']['1'] = meta['retired_groups']['0']
        elif corruption == 'missing_provenance':
            meta['retired_groups'].pop('0')
        elif corruption == 'bad_color':
            meta['retired_groups']['0']['color'] = [1, 2, 999]
        else:
            meta['next_group_id'] = 1
    rewrite_project(path, corrupt)
    with pytest.raises(ValueError, match='Projeto RIIMA inválido'):
        review.load_project(path)


def test_v7_high_group_ids_roundtrip_without_compacting_outside_v6_bounds(tmp_path):
    prepared = prepared_dataset(tmp_path, count=1)
    state = prepared['review']
    for index in range(254):
        group = state.create_group(color=(index % 256, 100, 200))
        if index < 253:
            state.delete_group(group)
    assert group == 255
    state.annotate(0, group, [212], 1)
    state.refine_pending()
    state.approve(group)
    path = tmp_path / 'high-id.riima'
    review.save_project(prepared, path)
    with np.load(path, allow_pickle=False) as archive:
        meta = json.loads(str(archive['metadata']))
        assert meta['version'] == 7
        assert '0_labels' not in meta['integer_dtypes']
        assert '0_locked' not in meta['integer_dtypes']
    actual = review.load_project(path)['review']
    assert actual.completed == {255} and actual.next_group_id == 256
    np.testing.assert_array_equal(actual.items[0]['labels'], state.items[0]['labels'])
    np.testing.assert_array_equal(actual.items[0]['locked'], state.items[0]['locked'])
