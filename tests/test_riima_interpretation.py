import csv
import json
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from riima import core
def optical():
    return dict(labels=np.array([[0, 1, 2], [0, 1, 2]], dtype=np.uint8),
                valid=np.array([[1, 1, 1], [1, 1, 0]], dtype=bool),
                confidence=np.ones((2, 3), dtype=np.float32), clusters=3,
                palette=core.cluster_palette(3), centers=np.arange(12).reshape(3, 4),
                feature_means=np.arange(12).reshape(3, 4), image_names=['example'],
                reference_index=0, mask_mode='auto')


def test_raw_immutable_and_merge_undo_preserves_metadata():
    raw = optical()
    before = raw['labels'].copy()
    state = core.Interpretation(raw)
    with pytest.raises(ValueError):
        state.raw['labels'][0, 0] = 2
    state.rename_cluster(2, 'C')
    state.recolor_cluster(2, (1, 2, 3))
    state.merge_clusters([0, 1])
    assert state.groups == [{0, 1}, {2}]
    assert state.names[1] == 'C'
    assert state.colors[1] == (1, 2, 3)
    assert state.interpreted()['labels'][0].tolist() == [0, 0, 1]
    state.undo()
    assert state.groups == [{0}, {1}, {2}]
    assert state.names[2] == 'C'
    np.testing.assert_array_equal(raw['labels'], before)


def test_families_manual_membership_and_area():
    state = core.Interpretation(optical())
    assert state.families == []
    state.create_family([0, 2], 'Example', (20, 30, 40))
    result = state.family_result()
    assert result['stats'][0]['pixels'] == 3
    assert result['stats'][0]['area'] == 60
    assert result['family_map'].tolist() == [[1, 0, 1], [1, 0, 0]]
    state.add_to_family(0, [1])
    assert state.family_result()['stats'][0]['pixels'] == 5
    state.remove_from_family(0, [2])
    assert state.families[0]['clusters'] == [0, 1]


def test_conflicting_family_merge_is_atomic():
    state = core.Interpretation(optical())
    state.create_family([0], 'A')
    state.create_family([1], 'B')
    with pytest.raises(ValueError):
        state.merge_clusters([0, 1])
    assert state.groups == [{0}, {1}, {2}]
    state.merge_families([0, 1])
    state.merge_clusters([0, 1])
    assert state.families[0]['clusters'] == [0, 1]


def test_reset_without_clustering(monkeypatch):
    state = core.Interpretation(optical())
    monkeypatch.setattr(core, 'fit_kmeans', lambda *a, **k: pytest.fail('reset refit'))
    state.rename_cluster(0, 'Name')
    state.recolor_cluster(0, (1, 2, 3))
    state.merge_clusters([0, 2])
    state.create_family([0], 'A')
    state.reset()
    assert state.groups == [{0}, {1}, {2}]
    assert not state.names and not state.colors and not state.families


def test_export_roundtrip_consistent_and_reset_overwrites(tmp_path):
    raw = optical()
    state = core.Interpretation(raw)
    state.merge_clusters([0, 1])
    state.rename_cluster(0, 'Manual')
    state.recolor_cluster(0, (1, 2, 3))
    state.create_family([0], 'Family', (4, 5, 6))
    prepared = dict(optical=raw, interpretation=state, output=str(tmp_path), config=core.Config(clusters=3))
    core.save_session_outputs(prepared)
    np.testing.assert_array_equal(np.load(tmp_path / 'clusters.npy'), raw['labels'])
    np.testing.assert_array_equal(np.load(tmp_path / 'families.npy'), state.family_result()['family_map'])
    rows = list(csv.DictReader((tmp_path / 'family_statistics.csv').open(encoding='utf-8')))
    assert rows[0]['name'] == 'Family'
    assert int(rows[0]['pixels']) == 4
    restored = core.load_interpretation(raw, tmp_path)
    assert restored.to_dict() == state.to_dict()
    state.reset()
    core.save_session_outputs(prepared)
    assert not core.load_interpretation(raw, tmp_path).families
    assert not np.load(tmp_path / 'families.npy').any()


def test_legacy_migration_preserves_safe_clusters_warns_for_lost_families(tmp_path):
    (tmp_path / 'optical_merge.json').write_text(json.dumps({'groups': [[0, 1], [2]]}))
    (tmp_path / 'display_metadata.json').write_text(json.dumps({'optical_group_names': {'0': 'Safe'}}))
    (tmp_path / 'family_merge.json').write_text(json.dumps({'groups': [[0, 1]]}))
    with pytest.warns(UserWarning):
        state = core.load_interpretation(optical(), tmp_path)
    assert state.groups == [{0, 1}, {2}]
    assert state.names[0] == 'Safe'
    assert state.families == []


def test_legacy_bad_display_entry_does_not_discard_valid_merges(tmp_path):
    (tmp_path / 'optical_merge.json').write_text(json.dumps({'groups': [[0, 1], [2]]}))
    (tmp_path / 'display_metadata.json').write_text(json.dumps({
        'optical_group_names': {'0': 'Safe', '7': 'Invalid'},
        'optical_group_colors': {'1': [1, 2, 3]}}))
    with pytest.warns(UserWarning):
        state = core.load_interpretation(optical(), tmp_path)
    assert state.groups == [{0, 1}, {2}]
    assert state.names == {0: 'Safe'}
    assert state.colors == {1: (1, 2, 3)}


def test_load_rejects_metadata_from_a_different_cluster_solution(tmp_path):
    raw = optical()
    wrong = raw['labels'].copy()
    wrong[0, 0] = 1
    np.save(tmp_path / 'clusters.npy', wrong)
    (tmp_path / 'interpretation.json').write_text(json.dumps(core.Interpretation(raw).to_dict()))
    with pytest.raises(ValueError, match='clusters'):
        core.load_interpretation(raw, tmp_path)


def test_load_rejects_reordered_groups_instead_of_silently_reassigning_names(tmp_path):
    payload = core.Interpretation(optical()).to_dict()
    payload['groups'] = [[2], [0, 1]]
    payload['names'] = {'0': 'Only raw 2'}
    (tmp_path / 'interpretation.json').write_text(json.dumps(payload))
    with pytest.raises(ValueError, match='order'):
        core.load_interpretation(optical(), tmp_path)
