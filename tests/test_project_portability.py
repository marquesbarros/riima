"""Portable destinations and one stored independent training sample."""
import json
from pathlib import Path
import shutil

import numpy as np
import pytest

from riima import core, review
from test_riima_review import prepared_dataset, rewrite_project


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
        assert meta['version'] == 5
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
        assert meta['version'] == 5
        assert meta['sample_layout'] == 'per_item'
    loaded = review.load_project(path)
    assert loaded['analysis_mode'] == 'registered'
    np.testing.assert_array_equal(loaded['optical']['sample_X'], prepared['optical']['sample_X'])
    assert loaded['output_history'] == [str(tmp_path / 'registered')]
