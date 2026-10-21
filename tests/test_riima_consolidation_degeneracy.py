"""R-07: flag undefined separation without changing published raw scores."""
import warnings

import numpy as np
import pytest

from riima import core


@pytest.mark.parametrize('distinct', [1, 2])
def test_fit_warns_when_k_exceeds_distinct_vectors_without_changing_clustering(distinct):
    X = np.repeat(np.arange(distinct, dtype=np.float32), 12)[:, None]
    with pytest.warns(RuntimeWarning, match='coincidentes'):
        labels, centers = core.fit_kmeans(X, 3, 1, 13)
    # The existing fit is still returned, including its nonempty fitted labels.
    assert labels.shape == (len(X),)
    assert labels.dtype == np.uint8
    np.testing.assert_array_equal(np.unique(centers), np.arange(distinct, dtype=np.float32))
    assert len(np.unique(centers, axis=0)) < len(centers)


def test_exact_coincident_center_tie_warns_and_preserves_raw_confidence():
    X = np.zeros((20, 4), np.float32)
    centers = np.zeros((3, 4), np.float32)
    labels, d1, d2 = core.nearest_two_centers(X, centers)
    with pytest.warns(RuntimeWarning, match='confiança.*0/0'):
        confidence = core.compute_optical_confidence(d1, d2)
    np.testing.assert_array_equal(labels, np.zeros(20, np.uint8))
    np.testing.assert_array_equal(confidence, np.ones(20, np.float32))
    with pytest.warns(RuntimeWarning, match='Davies-Bouldin.*separação'):
        assert core._davies_bouldin(X, labels, centers) == 0


def test_valid_distances_and_positive_distance_ties_preserve_confidence_without_warning():
    X = np.array([[0.], [1.], [2.]], np.float32)
    centers = np.array([[0.], [4.]], np.float32)
    with warnings.catch_warnings():
        warnings.simplefilter('error')
        labels, d1, d2 = core.nearest_two_centers(X, centers)
        confidence = core.compute_optical_confidence(d1, d2)
    np.testing.assert_array_equal(labels, [0, 0, 0])
    np.testing.assert_array_equal(d1, [0, 1, 4])
    np.testing.assert_array_equal(d2, [16, 9, 4])
    np.testing.assert_allclose(confidence, [1, 8 / 9, 0], atol=1e-7)


def test_all_degenerate_diagnostics_are_explicit_and_cannot_be_recommended():
    with pytest.warns(RuntimeWarning):
        result = core.evaluate_cluster_counts(np.zeros((20, 4), np.float32), [2, 3], attempts=1)
    assert result['diagnostic_valid'] == [False, False]
    assert all(result['diagnostic_warnings'])
    assert result['davies_bouldin'] == [0, 0]  # Preserve the existing raw convention.
    assert core.recommend_cluster_count(result['k_values'], result['silhouette'],
                                        diagnostic_valid=result['diagnostic_valid']) is None


def test_valid_and_degenerate_candidates_keep_raw_scores_but_only_valid_k_is_recommended():
    X = np.repeat(np.array([[0., 0.], [10., 10.]], np.float32), 10, axis=0)
    with pytest.warns(RuntimeWarning):
        result = core.evaluate_cluster_counts(X, [2, 3], attempts=1)
    assert result['diagnostic_valid'] == [True, False]
    assert result['diagnostic_warnings'][0] == []
    assert result['diagnostic_warnings'][1]
    assert result['silhouette'][0] == pytest.approx(1)
    assert result['davies_bouldin'][0] == 0
    # A manually supplied high invalid score must not win the advisory ranking.
    assert core.recommend_cluster_count([2, 3], [0.5, 1.],
                                        diagnostic_valid=result['diagnostic_valid']) == 2
    assert core.recommend_cluster_count([2, 3], [0.5, 1.]) == 3


def test_recommendation_rejects_misaligned_validity_mask():
    with pytest.raises(ValueError, match='diagnostic_valid'):
        core.recommend_cluster_count([2, 3], [0.5, 0.6], diagnostic_valid=[True])


def test_constant_image_fit_exposes_degeneracy_and_preserves_prediction_arrays():
    sources = [np.full((12, 12), 120, np.float32) for _ in range(4)]
    valid = np.ones((12, 12), bool)
    cfg = core.Config(clusters=3, sample_pixels=144, kmeans_attempts=1)
    with pytest.warns(RuntimeWarning, match='coincidentes|confiança'):
        result = core.cluster_optical_affinity(sources, valid, cfg)
    np.testing.assert_array_equal(result['labels'], np.zeros(144, np.uint8))
    np.testing.assert_array_equal(result['confidence'], np.ones(144, np.float32))
    np.testing.assert_array_equal(result['centers'], np.zeros((3, 4), np.float32))
