"""Descriptive calculations over RIIMA CSVs, with NumPy and stable group IDs."""
from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np


def standardize(values):
    """Population z-scores; missing observations stay missing, constants do not rank."""
    values = np.asarray(values, float)
    finite = np.isfinite(values)
    counts = finite.sum(axis=0)
    means = np.divide(np.where(finite, values, 0).sum(axis=0), counts,
                      out=np.zeros(values.shape[1]), where=counts > 0)
    centered = np.where(finite, values - means, 0)
    std = np.sqrt(np.divide((centered ** 2).sum(axis=0), counts,
                            out=np.zeros_like(means), where=counts > 0))
    varying = (counts > 1) & (std > 1e-12)
    z = centered / np.where(varying, std, 1)
    z[~finite] = np.nan
    return z, varying


def standardize_imputed(values):
    """The reference similarity profile: fill means first, then standardize all rows."""
    values = np.asarray(values, float)
    finite = np.isfinite(values)
    counts = finite.sum(axis=0)
    means = np.divide(np.where(finite, values, 0).sum(axis=0), counts,
                      out=np.zeros(values.shape[1]), where=counts > 0)
    filled = np.where(finite, values, means)
    z, varying = standardize(filled)
    return z[:, varying]


def similarity_matrix(z):
    """Gaussian affinity with median pair distance; missing z-scores use mean zero."""
    z = np.nan_to_num(np.asarray(z, float), nan=0.)
    distances = np.sqrt(np.maximum(((z[:, None] - z[None, :]) ** 2).sum(axis=2), 0))
    pairs = distances[np.triu_indices(len(z), 1)]
    if not np.any(pairs > 1e-12):
        return np.ones((len(z), len(z)))
    sigma = float(np.median(pairs))
    if sigma <= 1e-12:
        sigma = float(np.median(pairs[pairs > 1e-12]))
    return np.exp(-distances ** 2 / (2 * sigma ** 2))


def ward_linkage(z):
    """Ward distances sqrt(2 * nA * nB / (nA+nB)) * centroid distance."""
    z = np.nan_to_num(np.asarray(z, float), nan=0.)
    n = len(z)
    if n < 2:
        return np.empty((0, 4)), list(range(n))
    centers = {i: row.copy() for i, row in enumerate(z)}
    sizes = {i: 1 for i in range(n)}
    leaves = {i: [i] for i in range(n)}
    rows = []
    for step in range(n - 1):
        ids = sorted(centers)
        vectors = np.array([centers[i] for i in ids])
        counts = np.array([sizes[i] for i in ids], float)
        squared = ((vectors[:, None] - vectors[None, :]) ** 2).sum(axis=2)
        squared *= 2 * counts[:, None] * counts[None, :] / (counts[:, None] + counts[None, :])
        squared[np.tril_indices(len(ids))] = np.inf
        a, b = np.unravel_index(np.argmin(squared), squared.shape)
        left, right = ids[a], ids[b]
        size = sizes[left] + sizes[right]
        rows.append([left, right, np.sqrt(squared[a, b]), size])
        merged = n + step
        centers[merged] = (sizes[left] * centers[left] + sizes[right] * centers[right]) / size
        sizes[merged] = size
        leaves[merged] = leaves[left] + leaves[right]
        del centers[left], centers[right]
    return np.asarray(rows), leaves[2 * n - 2]


def gaussian_density(values, xs):
    """Univariate Gaussian KDE with Scott bandwidth; bounded grid work in chunks."""
    values = np.asarray(values, float)
    values = values[np.isfinite(values)]
    if len(values) < 2:
        return None
    bandwidth = float(values.std(ddof=1) * len(values) ** (-1 / 5))
    if bandwidth <= 1e-12:
        return None
    result = np.zeros(len(xs), float)
    for start in range(0, len(values), 512):
        distances = (np.asarray(xs)[:, None] - values[None, start:start + 512]) / bandwidth
        result += np.exp(-.5 * distances ** 2).sum(axis=1)
    return result / (len(values) * bandwidth * np.sqrt(2 * np.pi))


def _csv(path):
    with Path(path).open(encoding='utf-8-sig', newline='') as handle:
        return list(csv.DictReader(handle))


def number(value):
    try:
        result = float(value)
        return result if np.isfinite(result) else np.nan
    except (TypeError, ValueError):
        return np.nan


def load_metrics(directory):
    """Read current CSVs. Pooled legacy fields keep each ID distinct from its name."""
    folder = Path(directory)
    summaries = _csv(folder / 'group_summary.csv')
    catalog = _csv(folder / 'regions.csv')
    attributes = _csv(folder / 'attribute_statistics.csv')
    ranks = _csv(folder / 'attribute_separation.csv')
    info = json.loads((folder / 'metrics_info.json').read_text(encoding='utf-8'))
    ids = sorted({int(r['group']) for r in summaries})
    images = {int(r['image_id']): dict(name=r['image'], valid=int(r['valid_pixels'])) for r in summaries}
    valid_pixels = sum(image['valid'] for image in images.values())
    groups, regions = [], {}
    region_fields = ('area_pixels', 'major_axis_px', 'minor_axis_px', 'aspect_ratio',
                     'eccentricity', 'anisotropy', 'orientation_deg', 'compactness_grid')
    grouped_regions = {c: [] for c in ids}
    for row in catalog:
        grouped_regions[int(row['group'])].append(row)
    grouped_summary = {c: [] for c in ids}
    for row in summaries:
        grouped_summary[int(row['group'])].append(row)
    for c in ids:
        entries = grouped_summary[c]
        regions[c] = {key: np.array([number(row[key]) for row in grouped_regions[c]]) for key in region_fields}
        values = regions[c]
        area = values['area_pixels']
        valid_angles = np.isfinite(values['orientation_deg'])
        weights = np.where(valid_angles, area * values['anisotropy'], 0.)
        angles = np.deg2rad(np.nan_to_num(values['orientation_deg']) * 2)
        vector = np.sum(weights * np.exp(1j * angles))
        coherence = float(np.clip(abs(vector) / weights.sum(), 0, 1)) if weights.sum() else 0.
        direction = float(np.degrees(np.angle(vector)) / 2 % 180) if coherence > 1e-8 else np.nan
        pixels = sum(int(r['pixels']) for r in entries)
        groups.append(dict(id=c, name=entries[0]['group_name'], pixels=pixels,
                           area_percent=100 * pixels / max(1, valid_pixels),
                           components_total=sum(int(r['components_total']) for r in entries),
                           components_measured=len(area),
                           components_omitted=sum(int(r['components_omitted']) for r in entries),
                           components_below_min_area=sum(int(r['components_below_min_area']) for r in entries),
                           component_area_median_px=float(np.median(area)) if len(area) else np.nan,
                           component_area_q75_px=float(np.percentile(area, 75)) if len(area) else np.nan,
                           mean_eccentricity=float(values['eccentricity'].mean()) if len(area) else np.nan,
                           mean_anisotropy=float(values['anisotropy'].mean()) if len(area) else np.nan,
                           dominant_direction_deg=direction, direction_coherence=coherence,
                           approved=all(r['approved'] == 'True' for r in entries),
                           background=all(r['background'] == 'True' for r in entries)))
    features = list(dict.fromkeys(row['feature'] for row in attributes))
    index = {c: i for i, c in enumerate(ids)}
    feature_index = {f: j for j, f in enumerate(features)}
    totals, counts = np.zeros((len(ids), len(features))), np.zeros((len(ids), len(features)))
    for row in attributes:
        i, j = index[int(row['group'])], feature_index[row['feature']]
        n, mean = number(row['sampled_pixels']), number(row['mean'])
        if np.isfinite(mean) and n > 0:
            totals[i, j] += n * mean
            counts[i, j] += n
    attribute_means = np.divide(totals, counts, out=np.full_like(totals, np.nan), where=counts > 0)
    keys = ['area_percent', 'components_measured', 'component_area_median_px', 'component_area_q75_px',
            'mean_eccentricity', 'mean_anisotropy', 'direction_coherence']
    labels = ['Área (%)', 'Regiões medidas', 'Área mediana (px)', 'Área Q75 (px)',
              'Excentricidade média', 'Anisotropia média', 'Coerência direcional']
    directions = np.deg2rad([g['dominant_direction_deg'] * 2 for g in groups])
    profile = np.column_stack((np.array([[g[k] for k in keys] for g in groups]).reshape(len(groups), len(keys)),
                               attribute_means, np.cos(directions), np.sin(directions)))
    columns = labels + features + ['Direção (cos 2θ)', 'Direção (sin 2θ)']
    z, varying = standardize(profile)
    # Empty/background groups remain in area tables, but never determine relative profiles.
    active = [i for i, g in enumerate(groups) if g['pixels'] > 0 and not g['background']]
    if len(active) != len(groups):
        z[:] = np.nan
        active_z, varying = standardize(profile[active])
        z[active] = active_z
    # The model imputes raw means BEFORE StandardScaler, unlike descriptor z-scores.
    active_z = standardize_imputed(profile[active])
    similarity = similarity_matrix(active_z)
    linkage, order = ward_linkage(active_z)
    descriptors = {}
    for i, g in enumerate(groups):
        eligible = np.flatnonzero(varying & np.isfinite(z[i]))
        best = eligible[np.argsort(-abs(z[i, eligible]), kind='stable')[:5]]
        descriptors[g['id']] = [(columns[j], float(z[i, j])) for j in best]
    return dict(groups=groups, regions=regions, features=features, columns=columns, profile=profile,
                z=z, varying=varying, descriptors=descriptors, active=active,
                similarity=similarity, linkage=linkage, order=order, ranks=ranks,
                info=info, valid_pixels=valid_pixels, images=images)
