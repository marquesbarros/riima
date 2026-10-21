"""Regression cases R-06/R-16: defined axial directions and local moments."""
import numpy as np
import pytest

from riima.metrics import mask_metrics


@pytest.mark.parametrize('pattern', [np.ones((3, 3), bool), np.array([
    [1, 1, 1, 1], [0, 1, 0, 1], [0, 1, 0, 1], [0, 1, 0, 0]], bool)])
def test_isotropic_regions_never_supply_an_aggregate_direction(pattern):
    mask = np.zeros((12, 7), bool)
    # The asymmetric nine-pixel pattern has exact isotropic covariance 10/9 I,
    # with centroid (5/3, 1); roundoff persists even with local moments.
    mask[3:3 + pattern.shape[0], 2:2 + pattern.shape[1]] = pattern
    summary, rows = mask_metrics(mask)
    assert rows[0]['orientation_deg'] is None
    assert summary['dominant_direction_deg'] is None
    assert summary['direction_coherence'] == 0


@pytest.mark.parametrize('offset', [(0, 32760), (32760, 0)])
@pytest.mark.parametrize('shape', ['square', 'rectangle', 'diagonal'])
@pytest.mark.parametrize('chunk_pixels', [17, 65536])
def test_component_shape_and_axial_direction_are_translation_invariant(offset, shape, chunk_pixels):
    # Integer-pixel translations must not change pixel-center covariance.
    # 1e-12 allows float64 eigensolver roundoff; it is below the existing
    # 1e-8 anisotropy threshold and far below the R-16 (~3e-8) artifact.
    pattern = np.ones((3, 3), bool) if shape == 'square' else np.ones((3, 7), bool)
    if shape == 'diagonal':
        pattern = np.eye(7, dtype=bool)
        pattern |= np.eye(7, k=1, dtype=bool)
    y, x = offset
    near = np.zeros((pattern.shape[0] + 4, pattern.shape[1] + 4), bool)
    near[2:2 + pattern.shape[0], 2:2 + pattern.shape[1]] = pattern
    far = np.zeros((y + near.shape[0], x + near.shape[1]), bool)
    far[y:y + near.shape[0], x:x + near.shape[1]] = near
    base_summary, base_rows = mask_metrics(near, chunk_pixels=chunk_pixels)
    far_summary, far_rows = mask_metrics(far, chunk_pixels=chunk_pixels)
    base, shifted = base_rows[0], far_rows[0]
    assert shifted['area_pixels'] == base['area_pixels'] == int(pattern.sum())
    for field in ('major_axis_px', 'minor_axis_px', 'anisotropy', 'eccentricity',
                  'perimeter_grid_px', 'aspect_ratio'):
        assert shifted[field] == pytest.approx(base[field], rel=1e-12, abs=1e-12)
    assert shifted['centroid_x_px'] == pytest.approx(base['centroid_x_px'] + x, abs=1e-11)
    assert shifted['centroid_y_px'] == pytest.approx(base['centroid_y_px'] + y, abs=1e-11)
    if shape == 'square':
        assert shifted['orientation_deg'] is base['orientation_deg'] is None
        assert far_summary['dominant_direction_deg'] is None
        assert far_summary['direction_coherence'] == 0
    else:
        assert shifted['orientation_deg'] == pytest.approx(base['orientation_deg'], abs=1e-10)
        assert far_summary['dominant_direction_deg'] == pytest.approx(
            base_summary['dominant_direction_deg'], abs=1e-10)
        assert far_summary['direction_coherence'] == pytest.approx(1, abs=1e-12)


def test_isotropic_components_do_not_dilute_anisotropic_direction():
    mask = np.zeros((24, 18), bool)
    mask[3:6, 2:5] = True
    mask[10:20, 12:15] = True
    summary, rows = mask_metrics(mask)
    assert len(rows) == 2
    assert summary['dominant_direction_deg'] == pytest.approx(90, abs=1e-12)
    assert summary['direction_coherence'] == pytest.approx(1, abs=1e-14)
