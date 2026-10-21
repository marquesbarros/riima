"""Bounded, on-demand geometric and sampled optical-attribute CSV exports.

Connected regions are mask components, not inferred mineral grains. Angles use
image coordinates: 0 degrees points right and 90 degrees points down.
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

import cv2
import numpy as np

from . import core
from .exports import export_operation, staged_directory
from .viewer import workspace_group_label


def mask_metrics(mask, *, min_area=9, max_regions=2000, chunk_pixels=65536):
    """Exact total area; bounded component geometry from pixel-center moments."""
    mask = np.asarray(mask, dtype=bool)
    if mask.ndim != 2 or min_area < 1 or max_regions < 1 or chunk_pixels < 1:
        raise ValueError('Máscara ou limites de métricas inválidos.')
    count, labels, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), connectivity=8)
    eligible = np.flatnonzero(stats[1:, cv2.CC_STAT_AREA] >= min_area) + 1
    chosen = eligible[np.argsort(-stats[eligible, cv2.CC_STAT_AREA], kind='stable')[:max_regions]]
    lookup = np.zeros(count, np.int32)
    lookup[chosen] = np.arange(1, len(chosen) + 1)
    labels = lookup[labels].ravel()
    n, width = len(chosen) + 1, mask.shape[1]
    origins = np.zeros((n, 2), np.float64)
    origins[1:] = stats[chosen, :2]
    sums = np.zeros((6, n), np.float64)
    perimeter = np.zeros(n, np.float64)
    for start in range(0, labels.size, chunk_pixels):
        idx = np.arange(start, min(start + chunk_pixels, labels.size))
        idx = idx[labels[idx] > 0]
        if not len(idx):
            continue
        ids = labels[idx]
        x, y = (idx % width).astype(np.float64), (idx // width).astype(np.float64)
        # Component-local moments avoid cancellation of large global offsets.
        # Pixel-center covariance and its existing isotropy thresholds are unchanged.
        lx, ly = x - origins[ids, 0], y - origins[ids, 1]
        for j, weight in enumerate((np.ones(len(idx)), lx, ly, lx * lx, ly * ly, lx * ly)):
            sums[j] += np.bincount(ids, weights=weight, minlength=n)
        for shift, edge in ((-1, x == 0), (1, x == width - 1),
                            (-width, y == 0), (width, y == mask.shape[0] - 1)):
            neighbour = labels[np.clip(idx + shift, 0, labels.size - 1)]
            boundary = edge | (neighbour != ids)
            perimeter += np.bincount(ids[boundary], minlength=n)
    rows = []
    for local, component in enumerate(chosen, 1):
        area, sx, sy, sxx, syy, sxy = sums[:, local]
        cx, cy = sx / area, sy / area
        cov = np.array([[sxx / area - cx * cx, sxy / area - cx * cy],
                        [sxy / area - cx * cy, syy / area - cy * cy]])
        values, vectors = np.linalg.eigh(cov)
        minor, major = np.maximum(values, 0)
        anisotropy = float((major - minor) / (major + minor)) if major + minor > 1e-12 else 0.
        angle = float(np.degrees(np.arctan2(vectors[1, 1], vectors[0, 1])) % 180) if anisotropy > 1e-8 else None
        left, top, w, h, _ = stats[component]
        p = float(perimeter[local])
        rows.append(dict(component_id=int(component), area_pixels=int(area), centroid_x_px=float(cx + left),
                         centroid_y_px=float(cy + top), major_axis_px=float(4 * np.sqrt(major)),
                         minor_axis_px=float(4 * np.sqrt(minor)),
                         aspect_ratio=float(np.sqrt(major / minor)) if minor > 1e-12 else None,
                         eccentricity=float(np.sqrt(max(0., 1 - minor / major))) if major > 1e-12 else 0.,
                         anisotropy=anisotropy, orientation_deg=angle, perimeter_grid_px=p,
                         compactness_grid=float(4 * np.pi * area / (p * p)) if p else None,
                         equivalent_diameter_px=float(np.sqrt(4 * area / np.pi)),
                         bbox_x=int(left), bbox_y=int(top), bbox_width=int(w), bbox_height=int(h),
                         touches_image_border=bool(left == 0 or top == 0 or left + w == mask.shape[1]
                                                   or top + h == mask.shape[0])))
    directed = [row for row in rows if row['orientation_deg'] is not None]
    weights = np.array([row['area_pixels'] * row['anisotropy'] for row in directed])
    angles = np.radians([2 * row['orientation_deg'] for row in directed])
    vector = np.sum(weights * np.exp(1j * angles))
    coherence = float(np.clip(abs(vector) / weights.sum(), 0, 1)) if weights.sum() else 0.
    areas = np.array([row['area_pixels'] for row in rows])
    summary = dict(pixels=int(mask.sum()), components_total=count - 1,
                   components_below_min_area=int(count - 1 - len(eligible)),
                   components_measured=len(rows), components_omitted=int(len(eligible) - len(rows)),
                   component_area_mean_px=float(areas.mean()) if len(areas) else None,
                   component_area_median_px=float(np.median(areas)) if len(areas) else None,
                   component_area_std_px=float(areas.std()) if len(areas) else None,
                   component_area_q25_px=float(np.percentile(areas, 25)) if len(areas) else None,
                   component_area_q75_px=float(np.percentile(areas, 75)) if len(areas) else None,
                   mean_eccentricity=float(np.mean([r['eccentricity'] for r in rows])) if rows else None,
                   mean_anisotropy=float(np.mean([r['anisotropy'] for r in rows])) if rows else None,
                   dominant_direction_deg=float(np.degrees(np.angle(vector)) / 2 % 180) if coherence > 1e-8 else None,
                   direction_coherence=coherence)
    return summary, rows


def _write_csv(path, rows, columns, *, append=False):
    with path.open('a' if append else 'w', newline='', encoding='utf-8' if append else 'utf-8-sig') as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        if not append:
            writer.writeheader()
        writer.writerows(rows)


@export_operation
def export_metrics(prepared, directory=None, *, sample_per_group=2000, min_area=9, max_regions=2000):
    """Export the applied state without fitting classifiers or changing image data."""
    state = prepared['review']
    if state.has_pending:
        raise ValueError('Atualize a classificação antes de exportar as métricas.')
    if sample_per_group < 1:
        raise ValueError('A amostra das métricas deve ser positiva.')
    destination = Path(directory or Path(prepared['output']) / 'metrics')
    path = staged_directory(destination)
    path.mkdir(parents=True, exist_ok=True)
    summary_rows, attribute_rows = [], []
    identity_columns = ['image_id', 'image', 'group', 'group_name']
    region_columns = identity_columns + ['component_id', 'area_pixels', 'centroid_x_px', 'centroid_y_px',
        'major_axis_px', 'minor_axis_px', 'aspect_ratio', 'eccentricity', 'anisotropy', 'orientation_deg',
        'perimeter_grid_px', 'compactness_grid', 'equivalent_diameter_px', 'bbox_x', 'bbox_y',
        'bbox_width', 'bbox_height', 'touches_image_border']
    _write_csv(path / 'regions.csv', [], region_columns)
    pooled = {}
    image_names = state.items[0]['optical']['image_names']
    unique_names = [f'Imagem {i + 1} ({name})' if image_names.count(name) > 1 else name
                    for i, name in enumerate(image_names)]
    feature_names = ([channel for channel in ('L*', 'a*', 'b*', 'T')] if state.mode == 'independent' else
                     [f'{name}: {channel}' for name in unique_names for channel in ('L*', 'a*', 'b*', 'T')])
    # Limit the total sample as well as every group's share, independently of image size.
    per_group = min(int(sample_per_group), max(1, 10000 // len(state.class_ids)))
    rng = np.random.default_rng(state.config.seed)
    for image_id, item in enumerate(state.items):
        valid = item['display_valid']
        image_samples, image_groups = [], []
        for c in state.class_ids:
            mask = valid & (item['labels'] == c)
            summary, regions = mask_metrics(mask, min_area=min_area, max_regions=max_regions)
            identity = dict(image_id=image_id, image=item['name'], group=c,
                            group_name=workspace_group_label(prepared, c))
            summary_rows.append(dict(**identity, **summary, valid_pixels=int(valid.sum()),
                                     area_percent=100 * summary['pixels'] / max(1, int(valid.sum())),
                                     approved=c in state.completed, background=c in state.background_groups))
            # Stream detailed rows so additional images never accumulate a large
            # in-memory region catalogue. Each geometry batch is bounded above.
            _write_csv(path / 'regions.csv', (dict(**identity, **row) for row in regions), region_columns, append=True)
            indices = np.flatnonzero(mask)
            if len(indices):
                sample = rng.choice(indices, min(len(indices), per_group), replace=False)
                image_samples.append(sample)
                image_groups.append(np.full(len(sample), c, np.int32))
        if not image_samples:
            continue
        indices, groups = np.concatenate(image_samples), np.concatenate(image_groups)
        if item['sources'] is not None:
            X = core.collect_features(item['sources'], indices).astype(np.float64)
        else:
            # Resume needs at most the current image's Lab/T planes at one time;
            # retain only sampled attributes instead of materializing a 4N cube.
            images = item['optical']['images'] if state.mode == 'registered' else [item['rgb']]
            X = np.empty((len(indices), len(feature_names)), np.float64)
            for j, plane in enumerate(core.iter_optical_feature_planes(images, state.config.feature_blur_sigma)):
                X[:, j] = plane.ravel()[indices]
        for c in np.unique(groups):
            values = X[groups == c]
            n = len(values)
            mean, std = values.mean(axis=0), values.std(axis=0)
            q25, median, q75 = np.percentile(values, [25, 50, 75], axis=0)
            total, square = values.sum(axis=0), (values * values).sum(axis=0)
            old_n, old_total, old_square = pooled.get(int(c), (0, np.zeros(X.shape[1]), np.zeros(X.shape[1])))
            pooled[int(c)] = (old_n + n, old_total + total, old_square + square)
            for j, feature in enumerate(feature_names):
                attribute_rows.append(dict(image_id=image_id, image=item['name'], group=int(c),
                                           group_name=workspace_group_label(prepared, int(c)), feature=feature,
                                           sampled_pixels=n, mean=float(mean[j]), std=float(std[j]),
                                           q25=float(q25[j]), median=float(median[j]), q75=float(q75[j])))
    ranks = []
    if pooled:
        total_n = sum(n for n, _, _ in pooled.values())
        grand = sum(total for _, total, _ in pooled.values()) / total_n
        between = sum(n * (total / n - grand) ** 2 for n, total, _ in pooled.values())
        within = sum(np.maximum(square - total * total / n, 0) for n, total, square in pooled.values())
        denominator = between + within
        scores = np.divide(between, denominator, out=np.zeros_like(between), where=denominator > 1e-12)
        for j in np.argsort(-scores, kind='stable'):
            ranks.append(dict(rank=len(ranks) + 1, feature=feature_names[j], separation_eta2=float(np.clip(scores[j], 0, 1)),
                              sampled_pixels=total_n, groups_sampled=len(pooled)))
    summary_columns = identity_columns + ['pixels', 'valid_pixels', 'area_percent', 'approved', 'background',
        'components_total', 'components_below_min_area', 'components_measured', 'components_omitted',
        'component_area_mean_px', 'component_area_median_px', 'component_area_std_px',
        'component_area_q25_px', 'component_area_q75_px', 'mean_eccentricity', 'mean_anisotropy',
        'dominant_direction_deg', 'direction_coherence']
    _write_csv(path / 'group_summary.csv', summary_rows, summary_columns)
    _write_csv(path / 'attribute_statistics.csv', attribute_rows,
               identity_columns + ['feature', 'sampled_pixels', 'mean', 'std', 'q25', 'median', 'q75'])
    _write_csv(path / 'attribute_separation.csv', ranks, ['rank', 'feature', 'separation_eta2', 'sampled_pixels', 'groups_sampled'])
    info = dict(version=1, units='pixels', connectivity=8, min_component_area_px=min_area,
                max_measured_regions_per_group_image=max_regions, sample_per_group=sample_per_group,
                effective_sample_per_group=per_group, max_sample_per_image=10000,
                orientation='0 degrees right; 90 down; axial range [0,180); null for isotropic regions.',
                ellipse_axes='4 * sqrt(eigenvalues of pixel-center covariance).',
                anisotropy='(lambda_max - lambda_min)/(lambda_max + lambda_min), geometric only.',
                dominant_direction='Axial mean weighted by area * anisotropy, separately per image/group.',
                perimeter='Exposed pixel faces on the 4-neighbour grid, including holes.',
                attribute_separation='Sampled between-group sum of squares / total sum of squares (eta squared). Descriptive, not model importance or mineral identification.',
                scope='Applied masks. Components are connected regions, not validated grains. Counts and areas include all pixels; component shape summaries use measured regions only.',
                sources=['https://docs.opencv.org/4.x/d3/dc0/group__imgproc__shape.html',
                         'https://scikit-image.org/docs/stable/api/skimage.measure.html',
                         'https://itl.nist.gov/div898/handbook/prc/section4/prc431.htm'])
    (path / 'metrics_info.json').write_text(json.dumps(info, indent=2, ensure_ascii=False, allow_nan=False), encoding='utf-8')
    return destination
