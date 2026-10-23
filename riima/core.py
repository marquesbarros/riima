#!/usr/bin/env python3
"""RIIMA: multi-image affinity mapping with expert-controlled interpretation.

Registered images -> canonical CIELAB and local chromatic heterogeneity (4N)
-> robust median/IQR normalization -> Euclidean K-means -> immutable clusters
-> manual interpreted clusters and families. A cluster is not a mineral.

Dependencies: numpy, opencv-python, pillow. No GUI dependency.
"""
from __future__ import annotations

import csv
import json
import os
from dataclasses import dataclass, asdict
from copy import deepcopy
import warnings
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw

from . import __version__
from .exports import export_operation, staged_directory, staged_file

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

PALETTE = np.array([
    [78, 121, 167], [242, 142, 43], [225, 87, 89], [118, 183, 178],
    [89, 161, 79], [237, 201, 72], [176, 122, 161], [255, 157, 167],
    [156, 117, 95], [186, 176, 172], [86, 180, 233], [213, 94, 0],
], dtype=np.uint8)


@dataclass
class Config:
    """Optical-stage parameters (registration through pixel-level clustering)."""
    clusters: int = 8
    sample_pixels: int = 80_000
    kmeans_attempts: int = 3
    feature_blur_sigma: float = 0.8
    spatial_window: int = 7
    edge_percentile: float = 68.0
    majority_threshold: float = 0.56
    confidence_threshold: float = 0.16
    island_max_area: int = 64
    island_confidence_threshold: float = 0.18
    island_max_aspect: float = 3.5
    chunk_pixels: int = 220_000
    seed: int = 13
    # Minimum fraction of the reference image that the common overlap of all active
    # images must cover. This is a software safety threshold (see
    # ``check_common_overlap``), not a value derived from petrographic validation.
    min_overlap_fraction: float = 0.30
    min_overlap_fraction_warn: float = 0.60


# ---------------------------------------------------------------------------
# 1. Loading images
# ---------------------------------------------------------------------------

def read_rgb(path: str | os.PathLike) -> np.ndarray:
    """Load RGB uint8, including Unicode paths unsupported by OpenCV on Windows."""
    encoded = np.fromfile(path, dtype=np.uint8)
    img = cv2.imdecode(encoded, cv2.IMREAD_COLOR) if encoded.size else None
    if img is None:
        raise FileNotFoundError(f"Could not read image: {path}")
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def load_images(paths: list[str | os.PathLike]) -> list[np.ndarray]:
    """Load N images as RGB uint8, in the given order."""
    return [read_rgb(p) for p in paths]


# ---------------------------------------------------------------------------
# 2. Registration (always direct-to-reference)
# ---------------------------------------------------------------------------

def structural_edge(rgb: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0
    sx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    sy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    return cv2.GaussianBlur(cv2.magnitude(sx, sy), (0, 0), 0.8)


def _edge_ncc(a: np.ndarray, b: np.ndarray, mask: np.ndarray | None = None) -> float:
    if mask is None:
        mask = np.ones(a.shape, dtype=bool)
    x = a[mask].astype(np.float64)
    y = b[mask].astype(np.float64)
    if x.size == 0:
        return 0.0
    x -= x.mean()
    y -= y.mean()
    denom = float(np.sqrt(np.sum(x * x) * np.sum(y * y)))
    return float(np.sum(x * y) / denom) if denom > 1e-12 else 0.0


def _warp_overlap(shape: tuple[int, int], matrix: np.ndarray) -> np.ndarray:
    """Pixels whose full linear-interpolation support lies in the source image."""
    h, w = shape
    # Match the image warp's actual OpenCV interpolation weights (including their
    # subpixel quantization). Nearest-neighbour support admits pixels partly made
    # from reflected border values. Float coverage avoids integer rounding hiding
    # a small out-of-image contribution; unity means every nonzero weight is real.
    src = np.ones((h, w), dtype=np.float32)
    warped = cv2.warpAffine(
        src, matrix.astype(np.float32), (w, h),
        flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP,
        borderMode=cv2.BORDER_CONSTANT, borderValue=0,
    )
    return warped == 1.0


def register_pair(reference: np.ndarray, moving: np.ndarray):
    """Register one image to a reference with translation + optional tiny-rotation refinement.

    Phase correlation on structural edges supplies a robust initial translation. A small
    Euclidean ECC refinement is accepted only when it measurably improves edge agreement
    and remains within conservative geometric limits. The returned overlap mask marks
    pixels that have real support in both images after warping.
    Compare candidates with identity on the same supported pixels. Without a
    structural improvement, retain identity and explicit unverified fallback metadata.
    """
    if reference.shape[:2] != moving.shape[:2]:
        raise ValueError("Images must have the same pixel dimensions to register.")

    h, w = reference.shape[:2]
    ref_edge = structural_edge(reference)
    mov_edge = structural_edge(moving)
    margin = max(4, int(round(0.06 * min(h, w))))
    ys = slice(margin, h - margin)
    xs = slice(margin, w - margin)
    ref_crop = ref_edge[ys, xs]
    mov_crop = mov_edge[ys, xs]

    center_mask = np.zeros((h, w), dtype=bool)
    center_mask[ys, xs] = True
    score_before = _edge_ncc(ref_edge, mov_edge, center_mask)

    def identity_fallback(reason, phase_response=0.0):
        # No geometric displacement can be inferred from absent/noninformative
        # structure. Retain coordinates explicitly; this does not verify alignment.
        info = {
            "method": "identity_fallback", "matrix": np.eye(2, 3, dtype=np.float32),
            "translation": (0.0, 0.0), "angle_deg": 0.0,
            "phase_response": float(phase_response) if np.isfinite(phase_response) else None,
            "ecc_score": None, "score_before": float(score_before), "score_after": float(score_before),
            "fallback_reason": reason,
            "registration_warning": "No supported structural improvement was found; identity was retained. "
                                    "Alignment is unverified and requires visual inspection.",
        }
        return moving.copy(), info, np.ones((h, w), dtype=bool)

    if ref_crop.size == 0 or min(ref_crop.shape) < 2:
        return identity_fallback("insufficient_interior")
    if np.ptp(ref_crop) <= 1e-8 or np.ptp(mov_crop) <= 1e-8:
        return identity_fallback("no_structural_signal")

    scale = 0.5 if min(ref_crop.shape) >= 700 else 1.0
    if scale != 1.0:
        a = cv2.resize(ref_crop, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
        b = cv2.resize(mov_crop, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    else:
        a, b = ref_crop, mov_crop

    (dx_s, dy_s), phase_response = cv2.phaseCorrelate(a, b)
    if not np.all(np.isfinite([dx_s, dy_s, phase_response])) or phase_response <= 0.0:
        return identity_fallback("uninformative_phase_correlation", phase_response)
    dx, dy = dx_s / scale, dy_s / scale
    phase_matrix = np.float32([[1.0, 0.0, dx], [0.0, 1.0, dy]])

    def score_matrix(matrix, common=None):
        aligned_edge = cv2.warpAffine(
            mov_edge, matrix.astype(np.float32), (w, h),
            flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP,
            borderMode=cv2.BORDER_REFLECT,
        )
        overlap = _warp_overlap((h, w), matrix)
        mask = center_mask & overlap
        if common is not None:
            mask &= common
        return (_edge_ncc(ref_edge, aligned_edge, mask), overlap,
                _edge_ncc(ref_edge, mov_edge, mask))

    best_matrix = phase_matrix
    best_score, best_overlap, best_before = score_matrix(phase_matrix)
    improves_identity = best_score > best_before + 1e-8
    method = "phase_translation"
    ecc_score = None

    # ECC is run only at reduced resolution and only as a constrained Euclidean refinement.
    # This captures the small scanner rotations seen in real pairs without opening the door
    # to scale/shear distortions that would be hard to justify petrographically.
    warp_small = np.float32([[1.0, 0.0, dx_s], [0.0, 1.0, dy_s]])
    try:
        ecc_score, warp_crop_s = cv2.findTransformECC(
            a, b, warp_small, cv2.MOTION_EUCLIDEAN,
            (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 80, 1e-5),
            None, 3,
        )
        A = warp_crop_s[:, :2].astype(np.float64)
        t_crop = warp_crop_s[:, 2].astype(np.float64) / scale
        offset = np.array([margin, margin], dtype=np.float64)
        t_full = t_crop + offset - A @ offset
        ecc_matrix = np.column_stack([A, t_full]).astype(np.float32)
        angle_deg = float(np.degrees(np.arctan2(A[1, 0], A[0, 0])))
        translation_mag = float(np.linalg.norm(t_full))
        ecc_alignment_score, ecc_overlap, ecc_before = score_matrix(ecc_matrix)
        plausible = abs(angle_deg) <= 0.75 and translation_mag <= 0.08 * min(h, w)
        better = False
        if plausible and ecc_alignment_score > ecc_before + 1e-8:
            if improves_identity:
                # Neither candidate gains an advantage merely by discarding a
                # different part of the field during the comparison.
                current_common, _, _ = score_matrix(best_matrix, ecc_overlap)
                refined_common, _, _ = score_matrix(ecc_matrix, best_overlap)
                better = refined_common >= current_common + 0.005
            else:
                better = ecc_alignment_score >= ecc_before + 0.005
        if better:
            best_matrix = ecc_matrix
            best_score = ecc_alignment_score
            best_overlap = ecc_overlap
            best_before = ecc_before
            improves_identity = True
            method = "ecc_euclidean"
    except cv2.error:
        pass

    if not improves_identity:
        return identity_fallback("no_improvement", phase_response)

    aligned = cv2.warpAffine(
        moving, best_matrix, (w, h),
        flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP,
        borderMode=cv2.BORDER_REFLECT,
    )
    angle_deg = float(np.degrees(np.arctan2(best_matrix[1, 0], best_matrix[0, 0])))
    info = {
        "method": method,
        "matrix": best_matrix,
        "translation": (float(best_matrix[0, 2]), float(best_matrix[1, 2])),
        "angle_deg": angle_deg,
        "phase_response": float(phase_response),
        "ecc_score": None if ecc_score is None else float(ecc_score),
        "score_before": float(best_before),
        "score_after": float(best_score),
    }
    return aligned, info, best_overlap.astype(bool)


class InsufficientOverlapError(ValueError):
    """Raised when the common overlap across active images is too small to analyze."""


def check_common_overlap(common_overlap: np.ndarray, cfg: Config) -> str:
    """Guard against registration leaving too little common area to analyze.

    This is a **software safety threshold**, not a value derived from petrographic
    validation: no evidence establishes a scientifically "correct" minimum overlap, so
    two conservative, explicitly named, documented fractions are used instead of an
    unexplained magic number:

    - below ``cfg.min_overlap_fraction`` (default 30% of the reference image), the
      analysis is refused outright: at that point too little of the slide would be
      shared across every active image for a joint result to be scientifically
      meaningful;
    - between that and ``cfg.min_overlap_fraction_warn`` (default 60%), the analysis
      proceeds, but the caller should surface a warning (the GUI/CLI do).

    Returns "" when the overlap is healthy, "warn" when it is low but usable, and
    raises :class:`InsufficientOverlapError` when it is below the hard floor.
    """
    fraction = float(common_overlap.mean())
    if fraction < cfg.min_overlap_fraction:
        raise InsufficientOverlapError(
            f"Common overlap across active images is only {100 * fraction:.1f}% of the "
            f"reference image, below the minimum of {100 * cfg.min_overlap_fraction:.0f}% "
            "required for a scientifically meaningful joint analysis. Check registration "
            "quality, image order, or choose a different reference."
        )
    if fraction < cfg.min_overlap_fraction_warn:
        return "warn"
    return ""


def register_images(image_paths: list[str], reference_index: int = 0) -> dict:
    """Register every active image directly to one chosen reference (never sequentially).

    Each non-reference image is aligned with an independent call to ``register_pair``,
    so a weak alignment on one image cannot propagate its error into another. The
    reference image itself receives an all-True overlap mask and identity metadata.
    """
    if not image_paths:
        raise ValueError("At least one image is required.")
    if not (0 <= reference_index < len(image_paths)):
        raise ValueError(f"reference_index outside range 0..{len(image_paths) - 1}")

    reference_rgb = read_rgb(image_paths[reference_index])
    images = []
    overlaps = []
    for i, path in enumerate(image_paths):
        name = Path(path).name
        if i == reference_index:
            overlap = np.ones(reference_rgb.shape[:2], dtype=bool)
            images.append({
                "name": name,
                "path": path,
                "rgb": reference_rgb,
                "overlap": overlap,
                "registration": {
                    "reference_index": reference_index,
                    "is_reference": True,
                    "method": "reference",
                    "matrix": np.eye(2, 3, dtype=np.float32),
                    "translation": (0.0, 0.0),
                    "shift_dx_dy": (0.0, 0.0),
                    "angle_deg": 0.0,
                    "phase_response": 1.0,
                    "ecc_score": None,
                    "score_before": 1.0,
                    "score_after": 1.0,
                },
            })
            overlaps.append(overlap)
            continue

        moving_rgb = read_rgb(path)
        aligned_rgb, info, overlap = register_pair(reference_rgb, moving_rgb)
        images.append({
            "name": name,
            "path": path,
            "rgb": aligned_rgb,
            "overlap": overlap,
            "registration": {
                "reference_index": reference_index,
                "is_reference": False,
                **info,
                "shift_dx_dy": info["translation"],
            },
        })
        overlaps.append(overlap)

    common_overlap = np.logical_and.reduce(overlaps)
    return {
        "reference_index": reference_index,
        "reference": reference_rgb,
        "images": images,
        "common_overlap": common_overlap,
    }


# ---------------------------------------------------------------------------
# 3. Analysis mask
# ---------------------------------------------------------------------------

def _background_connected_to_border(dark: np.ndarray) -> np.ndarray:
    """Dark connected components that touch the image border are background.

    A dark region that does *not* reach the border (an interior mineral, however
    irregularly shaped) is never flooded, regardless of whether some single bright
    contour happens to enclose it: this is what lets an oddly-shaped or
    edge-adjacent dark specimen region survive without needing to sit inside one
    dominant bright blob.
    """
    n, labels = cv2.connectedComponents(dark.astype(np.uint8), connectivity=8)
    if n <= 1:
        return np.zeros_like(dark, dtype=bool)
    border_labels = set(labels[0, :].tolist()) | set(labels[-1, :].tolist())
    border_labels |= set(labels[:, 0].tolist()) | set(labels[:, -1].tolist())
    border_labels.discard(0)  # label 0 is "signal" (non-dark), not a background component
    if not border_labels:
        return np.zeros_like(dark, dtype=bool)
    return np.isin(labels, list(border_labels))


def specimen_area_mask(image: np.ndarray) -> np.ndarray:
    """Estimate the useful thin-section footprint, preferring false inclusion over
    false exclusion of real specimen material.

    Background is identified directly (dark *and* connected to the image border)
    instead of inferred as "everything outside the single largest bright contour".
    A brightness-only threshold can merge an edge-adjacent dark grain with the
    background. Two criteria distinguish specimen evidence from border background:

    1. specimen *evidence* is "bright OR distinctly saturated": a mineral that is
       dark under this particular illumination usually still has some hue/chroma
       that a neutral scanner background lacks, so saturation alone can rescue it;
    2. background is only what is dark *and* reachable from the image border by a
       path of dark pixels: an interior dark region is never excluded merely for
       failing to sit inside one single dominant bright contour.

    A dark, low-saturation region that is genuinely part of the background frame is
    excluded.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    saturation = hsv[..., 1]
    h, w = gray.shape
    sigma = max(1.5, 0.0035 * min(h, w))
    smooth = cv2.GaussianBlur(gray, (0, 0), sigma, borderType=cv2.BORDER_REFLECT)
    threshold = float(np.clip(np.percentile(smooth, 2.0) + 6.0, 8.0, 20.0))
    signal = (smooth > threshold) | (saturation > 40)
    k = max(5, int(round(0.012 * min(h, w))) | 1)
    k = min(k, 31)
    signal_closed = cv2.morphologyEx(
        signal.astype(np.uint8) * 255, cv2.MORPH_CLOSE, np.ones((k, k), np.uint8), iterations=1
    ) > 0
    background = _background_connected_to_border(~signal_closed)
    specimen = ~background
    if specimen.mean() < 0.40:
        # The heuristic implausibly rejected most of the image; do not trust it.
        return np.ones((h, w), dtype=bool)
    specimen = cv2.erode(specimen.astype(np.uint8), np.ones((3, 3), np.uint8), iterations=1)
    return specimen.astype(bool)


def red_annotation_mask(rgb: np.ndarray) -> np.ndarray:
    """Flag likely scale-bar/manual-annotation pixels without removing red/pink minerals.

    A saturated-red pixel is only treated as an annotation when its connected component
    is also shape-consistent with an annotation: thin/elongated (a drawn line or a text
    stroke) or touching the image border (where scale bars and stamps are typically
    placed). A red/pink mineral grain of ordinary (non-elongated) shape in the interior
    is left alone. Color alone does not distinguish minerals from annotations.
    """
    r = rgb[..., 0].astype(np.int16)
    g = rgb[..., 1].astype(np.int16)
    b = rgb[..., 2].astype(np.int16)
    color_hit = (r > 175) & (g < 105) & (b < 105) & (r - g > 80)
    h, w = color_hit.shape
    if not np.any(color_hit):
        return np.zeros((h, w), dtype=bool)
    n, labels, stats, _ = cv2.connectedComponentsWithStats(color_hit.astype(np.uint8), connectivity=8)
    border_margin = max(2, int(round(0.02 * min(h, w))))
    keep = np.zeros((h, w), dtype=bool)
    for cid in range(1, n):
        x, y, cw, ch, area = stats[cid]
        if area <= 0:
            continue
        aspect = max(cw, ch) / max(1, min(cw, ch))
        touches_border = (
            x <= border_margin or y <= border_margin
            or (x + cw) >= w - border_margin or (y + ch) >= h - border_margin
        )
        elongated = aspect >= 4.0
        if elongated or touches_border:
            keep[labels == cid] = True
    return cv2.dilate(keep.astype(np.uint8), np.ones((5, 5), np.uint8), iterations=1).astype(bool)


def analysis_valid_mask(
    ppl: np.ndarray, cpol: np.ndarray, overlap: np.ndarray, *, mask_mode: str = "auto"
) -> np.ndarray:
    """Two-image convenience wrapper; delegates to :func:`build_analysis_mask`."""
    return build_analysis_mask([ppl, cpol], overlap, mask_mode=mask_mode)


def analysis_mask_components(
    images: list[np.ndarray], common_overlap: np.ndarray, *, mask_mode: str = "auto"
) -> dict:
    """Decompose the analysis mask into its hard vs. heuristic parts, for diagnostics
    and for the ``mask_mode`` user override.

    ``common_overlap`` (the registered geometric field shared by every active image) is
    a **hard constraint**: it is never weakened by ``mask_mode``. ``specimen``/
    ``annotations`` are **heuristic** exclusions that can be wrong on unusual material or
    lighting; ``mask_mode="overlap_only"`` lets the user bypass them entirely (falling
    back to the geometric overlap alone) when the automatic heuristic looks wrong for a
    given slide, rather than requiring a code change.

    The specimen footprint is the **union** of every active image's own specimen mask
    across all image counts. A pixel that is valid specimen in at least one
    active modality must not be excluded merely because another modality happens to
    render that same mineral very dark there; intersecting independently-thresholded
    per-image masks compounds each one's false negatives instead of using the positive
    evidence any single image provides. Annotations are still the union of every
    image's own annotation mask, since a scale bar or stamp is equally invalid
    regardless of which image it was drawn on.
    """
    if mask_mode not in ("auto", "overlap_only"):
        raise ValueError(f"Unknown mask_mode {mask_mode!r}; expected 'auto' or 'overlap_only'")
    if mask_mode == "overlap_only":
        specimen = np.ones_like(common_overlap)
        annotations = np.zeros_like(common_overlap)
    else:
        specimen = np.logical_or.reduce([specimen_area_mask(img) for img in images])
        annotations = np.logical_or.reduce([red_annotation_mask(img) for img in images])
    valid = specimen & common_overlap & ~annotations
    return {"overlap": common_overlap, "specimen": specimen, "annotations": annotations, "valid": valid}


def build_analysis_mask(
    images: list[np.ndarray], common_overlap: np.ndarray, *, mask_mode: str = "auto"
) -> np.ndarray:
    """Generalize the valid-analysis mask to N images. See :func:`analysis_mask_components`
    for the hard-vs-heuristic breakdown and the ``mask_mode`` override."""
    return analysis_mask_components(images, common_overlap, mask_mode=mask_mode)["valid"]


# ---------------------------------------------------------------------------
# 4. Multimodal optical features
# ---------------------------------------------------------------------------

def edge_strength_multi(images: list[np.ndarray]) -> np.ndarray:
    """Maximum gradient across N images with only one persistent float image.

    Updating the maximum in place avoids allocating one extra full-size array per
    additional modality, keeping memory roughly linear in image count.
    """
    e = None
    for rgb in images:
        g = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0
        g = cv2.GaussianBlur(g, (0, 0), 0.8)
        sx = cv2.Sobel(g, cv2.CV_32F, 1, 0, ksize=3)
        sy = cv2.Sobel(g, cv2.CV_32F, 0, 1, ksize=3)
        mag = cv2.magnitude(sx, sy)
        if e is None:
            e = mag
        else:
            np.maximum(e, mag, out=e)
    assert e is not None
    hi = np.percentile(e, 99.5)
    if hi > 0:
        np.divide(e, hi, out=e)
        np.clip(e, 0.0, 1.0, out=e)
    return e.astype(np.float32, copy=False)


def edge_strength(ppl: np.ndarray, cpol: np.ndarray) -> np.ndarray:
    """Two-image wrapper around ``edge_strength_multi``."""
    return edge_strength_multi([ppl, cpol])


# Single documented window (px) for the local chromatic-heterogeneity measure T.
# Kept as one named constant, never a different value per image or per image count.
CHROMATIC_HETEROGENEITY_WINDOW = 11


def lab_star_channels(rgb: np.ndarray, blur_sigma: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Canonical-unit CIELAB channels from an RGB uint8 image.

    OpenCV's 8-bit ``COLOR_RGB2LAB`` packs L into [0, 255] and a*/b* into [0, 255] with a
    128 offset, not the canonical CIELAB ranges (L* in [0, 100]; a*, b* roughly in
    [-128, 127]). The rescaling below is an exact affine per-channel transform:
    ``L* = 100 * L8 / 255``, ``a* = a8 - 128``, ``b* = b8 - 128``. This changes the
    reported units only, not the clustering result: a per-feature affine rescaling is
    absorbed by the robust normalization in :func:`standardized_sample`.
    """
    color = cv2.GaussianBlur(rgb, (0, 0), blur_sigma, borderType=cv2.BORDER_REFLECT) if blur_sigma > 0 else rgb
    lab8 = cv2.cvtColor(color, cv2.COLOR_RGB2LAB).astype(np.float32)
    L = lab8[..., 0] * (100.0 / 255.0)
    a = lab8[..., 1] - 128.0
    b = lab8[..., 2] - 128.0
    return L, a, b


def _local_variance(channel: np.ndarray, window: int) -> np.ndarray:
    x = channel.astype(np.float32)
    mean = cv2.boxFilter(x, cv2.CV_32F, (window, window), normalize=True, borderType=cv2.BORDER_REFLECT)
    sq = cv2.boxFilter(x * x, cv2.CV_32F, (window, window), normalize=True, borderType=cv2.BORDER_REFLECT)
    return np.maximum(sq - mean * mean, 0.0).astype(np.float32)


def local_chromatic_heterogeneity(
    L: np.ndarray, a: np.ndarray, b: np.ndarray, window: int = CHROMATIC_HETEROGENEITY_WINDOW
) -> np.ndarray:
    """T(p) = sqrt(var(L*) + var(a*) + var(b*)) in a fixed local window.

    A compact measure of local chromatic heterogeneity, not a full texture descriptor: it
    does not capture orientation, periodicity, or spatial arrangement, only how much the
    CIELAB signal varies within the window around each pixel. The same formula and the
    same window are used for every image, with no exception for any modality.
    """
    return np.sqrt(
        _local_variance(L, window) + _local_variance(a, window) + _local_variance(b, window)
    ).astype(np.float32)


def iter_optical_feature_planes(images: list[np.ndarray], blur_sigma: float):
    """Yield the symmetric 4N optical feature planes ``[L*_1, a*_1, b*_1, T_1, ...,
    L*_N, a*_N, b*_N, T_N]``, one plane at a time, in image order.

    Every active image contributes identically; none is privileged, and a reference
    image (used only for geometric registration elsewhere) plays no role here. This
    generator is the single definition of RIIMA's optical feature space.
    """
    for img in images:
        L, a, b = lab_star_channels(img, blur_sigma)
        t = local_chromatic_heterogeneity(L, a, b)
        yield L
        yield a
        yield b
        yield t


def extract_optical_features(images: list[np.ndarray], blur_sigma: float) -> list[np.ndarray]:
    """Materialize the symmetric 4N optical feature space as a list of full-image planes.

    ``x(p) = [L*_1, a*_1, b*_1, T_1, ..., L*_N, a*_N, b*_N, T_N]``, so ``dim(x) = 4N``.
    No image is privileged; the chosen reference index is used only for registration
    (:func:`register_images`) and never affects this feature vector.
    """
    return list(iter_optical_feature_planes(images, blur_sigma))


def collect_features(sources: list[np.ndarray], flat_idx: np.ndarray) -> np.ndarray:
    X = np.empty((len(flat_idx), len(sources)), dtype=np.float32)
    for j, src in enumerate(sources):
        X[:, j] = src.reshape(-1)[flat_idx].astype(np.float32)
    return X


def nearest_two_centers(X: np.ndarray, centers: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Nearest and second-nearest center *squared* distances, without an N x K distance matrix."""
    n = len(X)
    best = np.full(n, np.inf, dtype=np.float32)
    second = np.full(n, np.inf, dtype=np.float32)
    labels = np.zeros(n, dtype=np.uint8)
    for k, center in enumerate(centers):
        d = np.sum((X - center) ** 2, axis=1, dtype=np.float32)
        better = d < best
        second = np.where(better, best, np.minimum(second, d)).astype(np.float32, copy=False)
        best[better] = d[better]
        labels[better] = k
    return labels, best, second


def compute_optical_confidence(d1: np.ndarray, d2: np.ndarray) -> np.ndarray:
    """Confidence from nearest/second-nearest squared distances: 1 - d1/d2, clipped to [0, 1].

    A pixel exactly on its assigned cluster center relative to the next-closest one scores
    1.0; a pixel equidistant between two clusters (an optical class boundary or an
    ambiguous/transitional response) scores close to 0.0. Low confidence is a statement
    about optical ambiguity (a class transition or internal heterogeneity), not a
    statement about mineral certainty.

    The existing epsilon convention yields 1.0 for d1 == d2 == 0. A warning
    identifies this degenerate tie; that raw value is not evidence of separation.
    """
    if np.any((d1 == 0) & (d2 == 0)):
        warnings.warn(
            'A confiança contém empate degenerado 0/0 entre centros coincidentes; '
            'o valor bruto regularizado não demonstra separação óptica.',
            RuntimeWarning, stacklevel=2,
        )
    return np.clip(1.0 - d1 / (d2 + 1e-8), 0.0, 1.0).astype(np.float32)


# ---------------------------------------------------------------------------
# 5. Optical clustering + confidence
# ---------------------------------------------------------------------------

def standardized_sample(sources: list[np.ndarray], valid: np.ndarray, n: int, seed: int):
    """Sample valid pixels and apply robust (median/IQR) normalization per feature.

    ``z_j = (x_j - median(x_j)) / IQR(x_j)``, with median and IQR computed only over the
    sampled valid pixels. Robust statistics are used instead of mean/standard deviation
    because a small number of extreme pixels (bright specimen edges, residual annotation
    fragments) can otherwise dominate the scale estimate; median/IQR are unaffected by
    up to 25% outliers on either side. A near-zero IQR (a feature that is nearly constant
    in the sample) falls back to a scale of 1.0, matching the analogous mean/std fallback
    used elsewhere in this module, so that feature does not blow up to an arbitrary scale.
    """
    rng = np.random.default_rng(seed)
    valid_idx = np.flatnonzero(valid.reshape(-1))
    if len(valid_idx) == 0:
        raise ValueError("No valid pixels remain after masking annotations.")
    n = min(n, len(valid_idx))
    idx = rng.choice(valid_idx, size=n, replace=False)
    X = collect_features(sources, idx)
    q25, median, q75 = np.percentile(X, [25.0, 50.0, 75.0], axis=0)
    iqr = q75 - q25
    iqr[iqr < 1e-6] = 1.0
    X = (X - median) / iqr
    return X.astype(np.float32), idx, median.astype(np.float32), iqr.astype(np.float32)


def _cluster_diagnostic_issues(labels: np.ndarray, centers: np.ndarray) -> list[str]:
    """Flag degeneracy without changing the fitted arrays or raw score conventions."""
    issues = []
    # Direct float64 differences avoid cancellation in the Gram distance formula.
    # Reuse DB's existing 1e-12 centroid-distance cutoff, not a new fit tolerance.
    points = np.asarray(centers, dtype=np.float64)
    if any(np.any(np.linalg.norm(points[c + 1:] - points[c], axis=1) <= 1e-12)
           for c in range(len(points) - 1)):
        issues.append('Centros ópticos coincidentes: confiança e índices brutos '
                      'não demonstram separação entre todos os grupos.')
    if len(np.unique(labels)) < len(centers):
        issues.append('Há grupos ópticos vazios: a separação solicitada não foi realizada.')
    return issues


def fit_kmeans(X: np.ndarray, k: int, attempts: int, seed: int):
    cv2.setRNGSeed(seed)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 120, 1e-3)
    _compact, labels, centers = cv2.kmeans(X, k, None, criteria, attempts, cv2.KMEANS_PP_CENTERS)
    for issue in _cluster_diagnostic_issues(labels, centers):
        warnings.warn(issue, RuntimeWarning, stacklevel=2)
    return labels.reshape(-1).astype(np.uint8), centers.astype(np.float32)


def predict_chunked(sources, center, scale, centers, chunk):
    """Predict labels/confidence while keeping memory O(chunk x features), not O(chunk x clusters).

    ``center``/``scale`` are the median/IQR returned by :func:`standardized_sample`, applied
    identically here so every pixel is classified in the same normalized space the k-means
    centers were fitted in.
    """
    total = sources[0].size
    labels = np.empty(total, dtype=np.uint8)
    confidence = np.empty(total, dtype=np.float32)

    for start in range(0, total, chunk):
        end = min(start + chunk, total)
        idx = np.arange(start, end, dtype=np.int64)
        X = collect_features(sources, idx)
        X = (X - center) / scale
        first, d1, d2 = nearest_two_centers(X, centers)
        labels[start:end] = first
        confidence[start:end] = compute_optical_confidence(d1, d2)
    return labels, confidence


def cluster_optical_affinity(sources: list[np.ndarray], valid: np.ndarray, cfg: Config):
    """Fit and apply k-means over the robustly-normalized 4N optical feature space.

    Orchestrates sampling, fitting, and chunked prediction. Unit weights (implicit:
    Euclidean distance in the normalized space, no per-feature multiplier) are used
    throughout; RIIMA 1.0 does not apply empirically-tuned feature weights. Returns a
    dict with flat ``labels``/``confidence`` (same shape as any source plane), the
    fitted ``centers``, and the sampling artifacts (``center``, ``scale``, ``sample_idx``,
    ``sample_X``, ``sample_labels``) needed downstream (e.g. representative patches, the
    on-demand cluster-count diagnostics).
    """
    sample_X, sample_idx, center, scale = standardized_sample(sources, valid, cfg.sample_pixels, cfg.seed)
    sample_labels, centers = fit_kmeans(sample_X, cfg.clusters, cfg.kmeans_attempts, cfg.seed)
    flat_labels, flat_conf = predict_chunked(sources, center, scale, centers, cfg.chunk_pixels)
    return {
        "labels": flat_labels,
        "confidence": flat_conf,
        "centers": centers,
        "center": center,
        "scale": scale,
        "sample_idx": sample_idx,
        "sample_X": sample_X,
        "sample_labels": sample_labels,
    }


# ---------------------------------------------------------------------------
# 6. Spatial regularization
# ---------------------------------------------------------------------------

def confidence_regularize(labels, confidence, edges, k, cfg: Config, valid):
    """Only ambiguous pixels are allowed to follow a strong local majority.

    This is deliberately conservative: confident optical assignments remain untouched, and
    strong structural boundaries block smoothing.
    """
    win = max(3, int(cfg.spatial_window) | 1)
    support = cv2.blur(valid.astype(np.float32), (win, win), borderType=cv2.BORDER_REFLECT)
    best_score = np.zeros(labels.shape, dtype=np.float32)
    majority = labels.copy()
    for c in range(k):
        votes = cv2.blur(((labels == c) & valid).astype(np.float32), (win, win), borderType=cv2.BORDER_REFLECT)
        score = np.divide(votes, support, out=np.zeros_like(votes), where=support > 0)
        better = score > best_score
        majority[better] = c
        best_score[better] = score[better]

    edge_cut = np.percentile(edges[valid], cfg.edge_percentile) if np.any(valid) else 1.0
    ambiguous = confidence < cfg.confidence_threshold
    can_change = valid & ambiguous & (edges < edge_cut) & (best_score >= cfg.majority_threshold)
    out = labels.copy()
    out[can_change] = majority[can_change]
    return out


def cleanup_uncertain_islands(labels, confidence, edges, valid, k, cfg: Config):
    """Remove only tiny, low-confidence, compact islands.

    Conservative by design: elongated components (possible lamellae) and confident optical
    domains are preserved. Processing is restricted to each component bounding box, keeping
    runtime close to linear in image size.
    """
    out = labels.copy()
    edge_cut = np.percentile(edges[valid], cfg.edge_percentile) if np.any(valid) else 1.0
    kernel = np.ones((3, 3), np.uint8)
    H, W = labels.shape

    for c in range(k):
        binary = ((out == c) & valid).astype(np.uint8)
        n, cc, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
        for cid in range(1, n):
            area = int(stats[cid, cv2.CC_STAT_AREA])
            if area > cfg.island_max_area:
                continue
            x = int(stats[cid, cv2.CC_STAT_LEFT])
            y = int(stats[cid, cv2.CC_STAT_TOP])
            w = int(stats[cid, cv2.CC_STAT_WIDTH])
            h = int(stats[cid, cv2.CC_STAT_HEIGHT])
            aspect = max(w, h) / max(1, min(w, h))
            if aspect > cfg.island_max_aspect:
                continue

            # Work only in a one-pixel padded bounding box.
            x0, x1 = max(0, x - 1), min(W, x + w + 1)
            y0, y1 = max(0, y - 1), min(H, y + h + 1)
            cc_roi = cc[y0:y1, x0:x1]
            comp = (cc_roi == cid)
            if not np.any(comp):
                continue
            conf_roi = confidence[y0:y1, x0:x1]
            edge_roi = edges[y0:y1, x0:x1]
            if float(np.median(conf_roi[comp])) >= cfg.island_confidence_threshold:
                continue
            if float(np.median(edge_roi[comp])) >= edge_cut:
                continue

            valid_roi = valid[y0:y1, x0:x1]
            out_roi = out[y0:y1, x0:x1]
            ring = cv2.dilate(comp.astype(np.uint8), kernel, iterations=1).astype(bool) & ~comp & valid_roi
            neigh = out_roi[ring]
            if neigh.size == 0:
                continue
            counts = np.bincount(neigh, minlength=k)
            counts[c] = 0
            target = int(np.argmax(counts))
            if counts[target] == 0 or counts[target] < 0.60 * counts.sum():
                continue
            out_roi[comp] = target
    return out


def cluster_palette(k: int) -> np.ndarray:
    if k <= len(PALETTE):
        return PALETTE[:k].copy()
    return np.tile(PALETTE, (int(np.ceil(k / len(PALETTE))), 1))[:k]


# ---------------------------------------------------------------------------
# 7. Interpreted optical merge (raw clusters always preserved)
# ---------------------------------------------------------------------------

def identity_groups(count: int) -> list[set[int]]:
    return [{i} for i in range(int(count))]


def _validate_partition(groups: list[set[int]], count: int) -> list[set[int]]:
    normalized = [set(int(v) for v in group) for group in groups if group]
    flat = [v for group in normalized for v in group]
    if sorted(flat) != list(range(int(count))):
        raise ValueError(f"Groups must partition IDs 0..{int(count)-1}")
    normalized.sort(key=lambda g: min(g))
    return normalized


def merge_selected_groups(groups: list[set[int]], selected_indices: list[int] | tuple[int, ...]) -> list[set[int]]:
    """Return a new partition with selected current groups united; input is never mutated."""
    selected = sorted(set(int(i) for i in selected_indices))
    if len(selected) < 2:
        raise ValueError("Select at least two groups to merge")
    if selected[0] < 0 or selected[-1] >= len(groups):
        raise IndexError("Selected group index outside range")
    union = set().union(*(groups[i] for i in selected))
    out = [set(group) for i, group in enumerate(groups) if i not in selected]
    out.append(union)
    out.sort(key=lambda g: min(g))
    return out


def apply_optical_interpretation(optical: dict, groups: list[set[int]] | None = None) -> dict:
    """Consolidate raw optical clusters for interpretation while leaving raw arrays untouched."""
    raw_count = int(optical["clusters"])
    groups = _validate_partition(groups or identity_groups(raw_count), raw_count)
    raw_labels = np.asarray(optical["labels"])
    lut = np.empty(raw_count, dtype=np.uint8)
    for new_id, members in enumerate(groups):
        for member in members:
            lut[int(member)] = int(new_id)
    labels = lut[raw_labels]
    palette_raw = np.asarray(optical["palette"], dtype=np.uint8)
    palette = np.asarray([palette_raw[min(group)] for group in groups], dtype=np.uint8)
    valid = np.asarray(optical["valid"], dtype=bool)
    confidence = np.asarray(optical["confidence"], dtype=np.float32)
    denom = max(int(valid.sum()), 1)
    stats = []
    for gid, members in enumerate(groups):
        mask = valid & (labels == gid)
        pixels = int(mask.sum())
        stats.append({
            "cluster": gid,
            "pixels": pixels,
            "area": 100.0 * pixels / denom,
            "confidence": float(confidence[mask].mean()) if pixels else 0.0,
            "source_clusters": tuple(sorted(members)),
        })
    return {
        **optical,
        "labels": labels,
        "palette": palette,
        "stats": stats,
        "clusters": len(groups),
        "groups": [set(g) for g in groups],
        "raw_clusters": raw_count,
    }


def parse_merge_spec(spec: str, k: int):
    """Parse '0+4+7;2+5+6'. Unmentioned clusters remain as singleton groups."""
    used = set()
    groups = []
    if spec.strip():
        for block in spec.split(";"):
            block = block.strip()
            if not block:
                continue
            vals = [int(x.strip()) for x in block.replace(",", "+").split("+") if x.strip()]
            if len(vals) < 2:
                raise ValueError(f"Merge group must contain at least two clusters: {block!r}")
            if any(v < 0 or v >= k for v in vals):
                raise ValueError(f"Cluster outside valid range 0..{k-1}: {block!r}")
            if len(vals) != len(set(vals)) or used.intersection(vals):
                raise ValueError("A cluster may appear only once in the merge specification.")
            used.update(vals)
            groups.append(set(vals))
    groups.extend({c} for c in range(k) if c not in used)
    groups.sort(key=lambda g: min(g))
    return groups


def apply_groups(labels: np.ndarray, groups):
    merged = np.empty_like(labels, dtype=np.uint8)
    for new_id, members in enumerate(groups):
        mask = np.zeros(labels.shape, dtype=bool)
        for c in members:
            mask |= labels == c
        merged[mask] = new_id
    return merged


class Interpretation:
    """Reversible manual metadata over immutable raw cluster labels.

    Families contain disjoint sets of raw IDs. Selecting an interpreted cluster
    always acts on all of its members. No operation refits the scientific model.
    """

    def __init__(self, optical: dict):
        self.raw = dict(optical)
        self.raw['labels'] = np.array(optical['labels'], copy=True)
        self.raw['labels'].setflags(write=False)
        self.groups = identity_groups(optical['clusters'])
        self.names: dict[int, str] = {}
        self.colors: dict[int, tuple] = {}
        self.families: list[dict] = []
        self._history: list[dict] = []

    def to_dict(self) -> dict:
        return dict(version=1, groups=[sorted(g) for g in self.groups],
                    names={str(k): v for k, v in self.names.items()},
                    colors={str(k): list(v) for k, v in self.colors.items()},
                    families=deepcopy(self.families))

    def _restore(self, payload: dict):
        incoming = [set(g) for g in payload['groups']]
        groups = _validate_partition(incoming, self.raw['clusters'])
        if incoming != groups:
            raise ValueError('Cluster group order must follow the smallest raw ID in each group')
        names = {int(k): str(v) for k, v in payload.get('names', {}).items()}
        colors = {int(k): self._color(v) for k, v in payload.get('colors', {}).items()}
        if any(i < 0 or i >= len(groups) for i in names.keys() | colors.keys()):
            raise ValueError('Interpretation metadata references an unknown cluster')
        families = deepcopy(payload.get('families', []))
        used = set()
        for family in families:
            members = set(family['clusters'])
            if not members <= set(range(self.raw['clusters'])) or used & members:
                raise ValueError('Family members must be valid, disjoint raw cluster IDs')
            if any(g & members and not g <= members for g in groups):
                raise ValueError('A family cannot split an interpreted cluster')
            family['clusters'] = sorted(members)
            family['name'] = str(family.get('name', ''))
            family['color'] = list(self._color(family['color']))
            used.update(members)
        self.groups, self.names, self.colors, self.families = groups, names, colors, families

    def _remember(self):
        self._history.append(self.to_dict())

    def undo(self):
        if self._history:
            self._restore(self._history.pop())

    @staticmethod
    def _color(color):
        values = tuple(int(v) for v in color)
        if len(values) != 3 or any(v < 0 or v > 255 for v in values):
            raise ValueError('Color must contain three RGB values in 0..255')
        return values

    @staticmethod
    def _selection(indices, count, minimum=1):
        selected = sorted(set(int(i) for i in indices))
        if len(selected) < minimum or any(i < 0 or i >= count for i in selected):
            raise ValueError('Selecione itens válidos em quantidade suficiente.')
        return selected

    def _members(self, indices):
        selected = self._selection(indices, len(self.groups))
        return set().union(*(self.groups[i] for i in selected))

    def merge_clusters(self, indices):
        selected = self._selection(indices, len(self.groups), 2)
        members = self._members(selected)
        owners = [f for f in self.families if members & set(f['clusters'])]
        if len(owners) > 1:
            raise ValueError('Os clusters pertencem a famílias diferentes. Una as famílias ou remova a associação antes.')
        new_groups = merge_selected_groups(self.groups, selected)
        # Retain the first selected group's identity and every unaffected identity.
        sources = [next(i for i, g in enumerate(self.groups) if min(new) in g) for new in new_groups]
        self._remember()
        self.names = {j: self.names[i] for j, i in enumerate(sources) if i in self.names}
        self.colors = {j: self.colors[i] for j, i in enumerate(sources) if i in self.colors}
        self.groups = new_groups
        if owners:
            owners[0]['clusters'] = sorted(set(owners[0]['clusters']) | members)

    def rename_cluster(self, index, name):
        self._selection([index], len(self.groups))
        self._remember()
        if str(name).strip():
            self.names[index] = str(name).strip()
        else:
            self.names.pop(index, None)

    def recolor_cluster(self, index, color):
        self._selection([index], len(self.groups))
        color = self._color(color)
        self._remember()
        self.colors[index] = color

    def reset_clusters(self):
        self._remember()
        self.groups = identity_groups(self.raw['clusters'])
        self.names, self.colors = {}, {}

    def reset_families(self):
        self._remember()
        self.families = []

    def reset(self):
        self._remember()
        self.groups = identity_groups(self.raw['clusters'])
        self.names, self.colors, self.families = {}, {}, []

    def create_family(self, indices, name='', color=None):
        members = self._members(indices)
        color = self._color(cluster_palette(len(self.families) + 1)[-1] if color is None else color)
        self._remember()
        for family in self.families:
            family['clusters'] = sorted(set(family['clusters']) - members)
        self.families.append(dict(clusters=sorted(members), name=str(name).strip(), color=list(color)))
        return len(self.families) - 1

    def add_to_family(self, index, indices):
        self._selection([index], len(self.families))
        members = self._members(indices)
        self._remember()
        for i, family in enumerate(self.families):
            current = set(family['clusters'])
            family['clusters'] = sorted(current | members if i == index else current - members)

    def remove_from_family(self, index, indices):
        self._selection([index], len(self.families))
        members = self._members(indices)
        self._remember()
        self.families[index]['clusters'] = sorted(set(self.families[index]['clusters']) - members)

    def merge_families(self, indices):
        selected = self._selection(indices, len(self.families), 2)
        self._remember()
        target = self.families[selected[0]]
        target['clusters'] = sorted(set().union(*(set(self.families[i]['clusters']) for i in selected)))
        self.families = [f for i, f in enumerate(self.families) if i not in selected[1:]]

    def delete_family(self, index):
        self._selection([index], len(self.families))
        self._remember()
        del self.families[index]

    def rename_family(self, index, name):
        self._selection([index], len(self.families))
        self._remember()
        self.families[index]['name'] = str(name).strip()

    def recolor_family(self, index, color):
        self._selection([index], len(self.families))
        color = self._color(color)
        self._remember()
        self.families[index]['color'] = list(color)

    def interpreted(self):
        result = apply_optical_interpretation(self.raw, self.groups)
        for i, color in self.colors.items():
            result['palette'][i] = color
        for row in result['stats']:
            i = row['cluster']
            row['name'] = self.names.get(i, '')
            row['color'] = result['palette'][i].tolist()
        return result

    def family_result(self):
        lut = np.zeros(self.raw['clusters'], dtype=np.int32)
        for i, family in enumerate(self.families):
            lut[family['clusters']] = i + 1
        labels = lut[self.raw['labels']]
        labels[~self.raw['valid']] = 0
        total = max(int(self.raw['valid'].sum()), 1)
        counts = np.bincount(labels[self.raw['valid']], minlength=len(self.families) + 1)
        stats = [dict(family=i + 1, source_clusters=list(f['clusters']), pixels=int(counts[i + 1]),
                      area=100.0 * counts[i + 1] / total, name=f['name'], color=list(f['color']))
                 for i, f in enumerate(self.families)]
        return dict(family_map=labels, family_count=len(self.families), families=deepcopy(self.families),
                    stats=stats, palette=np.asarray([f['color'] for f in self.families], dtype=np.uint8).reshape(-1, 3))


def load_interpretation(optical: dict, directory: str | Path) -> Interpretation:
    """Load current metadata or safely migrate pre-1.0 cluster interpretation.

    Older regional products are ignored; they are never converted into cluster
    families because there is no unambiguous cluster-level correspondence.
    """
    directory = Path(directory)
    state = Interpretation(optical)
    saved_labels = directory / 'clusters.npy'
    legacy_labels = directory / 'optical_labels_raw.npz'
    if saved_labels.exists():
        same = np.array_equal(np.load(saved_labels, allow_pickle=False), optical['labels'])
    elif legacy_labels.exists():
        with np.load(legacy_labels, allow_pickle=False) as arrays:
            same = np.array_equal(arrays['labels'], optical['labels']) and np.array_equal(arrays['valid'], optical['valid'])
    else:
        same = True
    if not same:
        raise ValueError('A interpretação salva pertence a outra solução de clusters.')
    current = directory / 'interpretation.json'
    if current.exists():
        state._restore(json.loads(current.read_text(encoding='utf-8')))
        return state
    payload = state.to_dict()
    old_groups = directory / 'optical_merge.json'
    if old_groups.exists():
        payload['groups'] = json.loads(old_groups.read_text(encoding='utf-8'))['groups']
    metadata = directory / 'display_metadata.json'
    meta = json.loads(metadata.read_text(encoding='utf-8')) if metadata.exists() else {}
    try:
        state._restore(payload)
    except (ValueError, KeyError, TypeError):
        warnings.warn('A interpretação antiga de clusters não pôde ser convertida; labels brutos preservados.', UserWarning)
    # Recover each valid display entry independently, so a stale index cannot
    # discard the safely convertible partition or unrelated names and colors.
    lost_metadata = False
    for key, attribute in [('optical_group_names', 'names'), ('optical_group_colors', 'colors')]:
        for index, value in meta.get(key, {}).items():
            try:
                index = int(index)
                if not 0 <= index < len(state.groups):
                    raise ValueError('stale index')
                getattr(state, attribute)[index] = state._color(value) if attribute == 'colors' else str(value)
            except (TypeError, ValueError):
                lost_metadata = True
    if lost_metadata:
        warnings.warn('Alguns nomes ou cores antigos não puderam ser convertidos; associações válidas preservadas.', UserWarning)
    if (directory / 'family_merge.json').exists() or meta.get('family_group_names') or meta.get('family_group_colors'):
        warnings.warn('Famílias antigas não têm correspondência segura com clusters; recrie as associações manualmente.', UserWarning)
    return state


def boundary_mask(labels: np.ndarray, valid: np.ndarray | None = None) -> np.ndarray:
    """Adjacent cluster/family label transitions inside the valid analysis area."""
    valid = np.ones(labels.shape, dtype=bool) if valid is None else np.asarray(valid, dtype=bool)
    b = np.zeros(labels.shape, dtype=bool)
    horizontal = (labels[:, 1:] != labels[:, :-1]) & valid[:, 1:] & valid[:, :-1]
    vertical = (labels[1:] != labels[:-1]) & valid[1:] & valid[:-1]
    b[:, 1:] |= horizontal
    b[:, :-1] |= horizontal
    b[1:] |= vertical
    b[:-1] |= vertical
    return b


def render_boundaries(base: np.ndarray, labels: np.ndarray, color=(255, 0, 0), thickness: int = 2, *, valid=None) -> np.ndarray:
    b = boundary_mask(labels, valid)
    if thickness > 1:
        b = cv2.dilate(b.astype(np.uint8), np.ones((thickness, thickness), np.uint8), iterations=1).astype(bool)
    if valid is not None:
        b &= valid
    out = base.copy(); out[b] = np.array(color, dtype=np.uint8)
    return out


@export_operation
def _save_png_fast(array: np.ndarray, path: str | os.PathLike, mode: str | None = None) -> None:
    """Save lossless PNG with a speed-oriented compression level."""
    if not Path(path).parent.is_dir():
        raise FileNotFoundError(f'Diretório de exportação inexistente: {Path(path).parent}')
    image = Image.fromarray(array, mode=mode) if mode is not None else Image.fromarray(array)
    image.save(staged_file(path), compress_level=3, optimize=False)


def render_optical_clusters(labels: np.ndarray, valid: np.ndarray, palette: np.ndarray) -> np.ndarray:
    rgb = palette[labels].copy()
    rgb[~valid] = [30, 30, 30]
    return rgb


def render_optical_confidence(confidence: np.ndarray, valid: np.ndarray) -> np.ndarray:
    gray = np.clip(confidence * 255.0, 0, 255).astype(np.uint8)
    gray[~valid] = 0
    return gray


def optical_statistics(labels, confidence, valid, k: int):
    """Collect cluster area and mean confidence without filesystem access."""
    denom = int(valid.sum())
    rows = []
    for c in range(k):
        m = (labels == c) & valid
        n = int(m.sum())
        area = (100.0 * n / denom) if denom else 0.0
        conf = float(confidence[m].mean()) if n else 0.0
        rows.append({"cluster": c, "pixels": n, "area": area, "confidence": conf})
    return rows


@export_operation
def save_optical_statistics(labels, confidence, valid, out_dir: Path, k: int, filename: str = "cluster_statistics.csv"):
    rows = optical_statistics(labels, confidence, valid, k)
    with open(staged_file(out_dir / filename), "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["cluster", "pixels", "area_percent_valid_image", "mean_confidence"])
        for row in rows:
            w.writerow([row['cluster'], row['pixels'], row['area'], row['confidence']])
    return rows


def cluster_similarity(centers: np.ndarray):
    """Distances between optical cluster centers in the standardized feature space."""
    pairs = []
    for a in range(len(centers)):
        for b in range(a + 1, len(centers)):
            d = float(np.linalg.norm(centers[a] - centers[b]) / np.sqrt(centers.shape[1]))
            pairs.append((a, b, d))
    pairs.sort(key=lambda x: x[2])
    nearest = {}
    for c in range(len(centers)):
        choices = [(d, b if a == c else a) for a, b, d in pairs if a == c or b == c]
        choices.sort()
        nearest[c] = choices[0][1] if choices else None
    return pairs, nearest


@export_operation
def save_registration_csv(registered_images: list[dict], out_dir: Path, filename: str = "registration.csv") -> None:
    with open(staged_file(out_dir / filename), "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["index", "name", "is_reference", "dx", "dy", "angle_deg", "score_before", "score_after",
                    "method", "fallback_reason", "registration_warning"])
        for i, im in enumerate(registered_images):
            r = im["registration"]
            w.writerow([
                i, im["name"], r["is_reference"], r["shift_dx_dy"][0], r["shift_dx_dy"][1],
                r["angle_deg"], r["score_before"], r["score_after"],
                r.get("method", ""), r.get("fallback_reason", ""), r.get("registration_warning", ""),
            ])


def find_representatives(sample_idx, final_labels_flat, confidence_flat, sample_X, centers, per_cluster):
    """Representative patches favor high-confidence interior pixels, then proximity to center.

    Diagnostics-only: not part of the standard output set.
    """
    reps = {}
    sample_final = final_labels_flat[sample_idx]
    sample_conf = confidence_flat[sample_idx]
    for c in range(len(centers)):
        loc = np.flatnonzero((sample_final == c) & (sample_conf >= np.quantile(sample_conf[sample_final == c], 0.55) if np.any(sample_final == c) else False))
        if len(loc) == 0:
            loc = np.flatnonzero(sample_final == c)
        if len(loc) == 0:
            reps[c] = []
            continue
        d = np.sum((sample_X[loc] - centers[c]) ** 2, axis=1)
        take = loc[np.argsort(d)[:per_cluster]]
        reps[c] = sample_idx[take].tolist()
    return reps


def crop_patch(rgb, y, x, size):
    r = size // 2
    y0, y1 = max(0, y - r), min(rgb.shape[0], y + r)
    x0, x1 = max(0, x - r), min(rgb.shape[1], x + r)
    patch = rgb[y0:y1, x0:x1]
    return cv2.copyMakeBorder(patch, 0, size - patch.shape[0], 0, size - patch.shape[1], cv2.BORDER_REFLECT)


@export_operation
def make_contact_sheet(ppl, cpol, reps, out_path: Path, palette, patch, stats=None, nearest=None):
    """Diagnostics-only representative-patch grid; never auto-saved as a standard output."""
    from .visual_style import DOCUMENT, pillow_font
    k = len(reps)
    cols = max((len(v) for v in reps.values()), default=1)
    pair_w, header_h = patch * 2 + 6, 34
    row_h = patch + header_h + 10
    sheet = Image.new("RGB", (cols * pair_w + 20, k * row_h + 20), DOCUMENT['surface'])
    draw = ImageDraw.Draw(sheet)
    font = pillow_font()
    H, W = ppl.shape[:2]
    for c in range(k):
        y0 = 10 + c * row_h
        color = tuple(int(x) for x in palette[c])
        draw.rectangle((10, y0, 32, y0 + 22), fill=color, outline=DOCUMENT['axis'])
        extra = ""
        if stats is not None:
            st = stats[c]
            extra += f"   {st['area']:.1f}%   conf {st['confidence']:.2f}"
        if nearest is not None and nearest.get(c) is not None:
            extra += f"   nearest: {nearest[c]}"
        draw.text((42, y0 + 1), f"Cluster {c}   (image 0 | image 1){extra}", fill=DOCUMENT['text'], font=font)
        for j, flat in enumerate(reps[c]):
            y, x = divmod(flat, W)
            a, b = crop_patch(ppl, y, x, patch), crop_patch(cpol, y, x, patch)
            pair = np.concatenate([a, np.full((patch, 6, 3), 255, np.uint8), b], axis=1)
            sheet.paste(Image.fromarray(pair), (10 + j * pair_w, y0 + header_h))
    sheet.save(staged_file(out_path), compress_level=3, optimize=False)


# ---------------------------------------------------------------------------
# 11. Top-level orchestration
# ---------------------------------------------------------------------------

@export_operation
def run_optical_analysis(
    image_paths: list[str], output: str, cfg: Config, reference_index: int = 0,
    *, diagnostics: bool = False, mask_mode: str = "auto", publish_outputs: bool = True,
) -> dict:
    """Register and jointly cluster N co-registered images; optionally write outputs.

    Registration is always direct-to-reference (see ``register_images``); the reference
    image has no role beyond registration. Every image, including the reference,
    contributes the same four features [L*, a*, b*, T] to clustering (see
    ``extract_optical_features``), so the feature dimension grows linearly as 4N.

    ``mask_mode``: ``"auto"`` (default) uses the specimen/annotation heuristics in
    :func:`analysis_mask_components`; ``"overlap_only"`` bypasses them and uses just the
    hard geometric registration overlap, for the (rare) case where the heuristic looks
    wrong on a particular slide. Never silently invented: see ``docs/METHOD.md``.

    ``publish_outputs=False`` computes the same in-memory result without creating
    the output directory or writing files. Diagnostic exports are also suppressed
    in this mode, even when ``diagnostics=True``.
    """
    if not image_paths:
        raise ValueError("At least one active image is required for analysis.")
    if publish_outputs:
        out_dir = staged_directory(output)
        out_dir.mkdir(parents=True, exist_ok=True)

    registered = register_images(image_paths, reference_index)
    images = [im["rgb"] for im in registered["images"]]
    reference_rgb = images[reference_index]
    overlap_status = check_common_overlap(registered["common_overlap"], cfg)
    mask_parts = analysis_mask_components(images, registered["common_overlap"], mask_mode=mask_mode)
    valid = mask_parts["valid"]
    if not np.any(valid):
        raise ValueError("Registration left no common valid analysis area.")
    if publish_outputs:
        Image.fromarray((valid.astype(np.uint8) * 255), mode="L").save(out_dir / "analysis_mask.png")
        if diagnostics:
            Image.fromarray((mask_parts["overlap"].astype(np.uint8) * 255), mode="L").save(out_dir / "mask_overlap.png")
            Image.fromarray((mask_parts["specimen"].astype(np.uint8) * 255), mode="L").save(out_dir / "mask_specimen_support.png")
            Image.fromarray((mask_parts["annotations"].astype(np.uint8) * 255), mode="L").save(out_dir / "mask_annotations.png")

    sources = extract_optical_features(images, cfg.feature_blur_sigma)
    is_pair = len(images) == 2
    fit = cluster_optical_affinity(sources, valid, cfg)
    labels = fit["labels"].reshape(reference_rgb.shape[:2])
    confidence = fit["confidence"].reshape(reference_rgb.shape[:2])
    centers = fit["centers"]

    edges = edge_strength_multi(images)
    labels = confidence_regularize(labels, confidence, edges, cfg.clusters, cfg, valid)
    labels = cleanup_uncertain_islands(labels, confidence, edges, valid, cfg.clusters, cfg)
    labels = labels.copy()
    labels.setflags(write=False)
    flat_labels = labels.reshape(-1)
    counts = np.bincount(labels[valid], minlength=cfg.clusters)
    feature_means = np.column_stack([
        np.divide(np.bincount(labels[valid], weights=plane[valid], minlength=cfg.clusters),
                  counts, out=np.zeros(cfg.clusters, dtype=float), where=counts > 0)
        for plane in sources
    ])

    palette = cluster_palette(cfg.clusters)
    if publish_outputs:
        _save_png_fast(render_optical_clusters(labels, valid, palette), out_dir / "clusters.png")
        _save_png_fast(render_optical_confidence(confidence, valid), out_dir / "optical_confidence.png", mode="L")
        stats = save_optical_statistics(labels, confidence, valid, out_dir, cfg.clusters)
        np.save(out_dir / "clusters.npy", labels)
    else:
        stats = optical_statistics(labels, confidence, valid, cfg.clusters)
    similarity_pairs, nearest = cluster_similarity(centers)

    if publish_outputs and diagnostics and is_pair:
        other_rgb = images[1] if reference_index == 0 else images[0]
        reps = find_representatives(fit["sample_idx"], flat_labels, fit["confidence"], fit["sample_X"], centers, 6)
        make_contact_sheet(reference_rgb, other_rgb, reps, out_dir / "diagnostic_representative_patches.png", palette, 96, stats=stats, nearest=nearest)

    image_names = [im["name"] for im in registered["images"]]
    if publish_outputs:
        save_registration_csv(registered["images"], out_dir)

    return {
        "output": str(Path(output)),
        "reference_index": reference_index,
        "registered_images": registered["images"],
        "image_names": image_names,
        "overlap": registered["common_overlap"],
        "overlap_status": overlap_status,
        "mask_mode": mask_mode,
        "clusters": cfg.clusters,
        "reference": reference_rgb,
        "images": images,
        "labels": labels,
        "confidence": confidence,
        "valid": valid,
        "palette": palette,
        "stats": stats,
        "similarity_pairs": similarity_pairs,
        "nearest": nearest,
        # Cached standardized k-means sample, reused (never recomputed) by the
        # optional on-demand cluster-count diagnostics (evaluate_cluster_counts) so
        # evaluating a k range never re-registers, re-extracts features, or touches the
        # full-resolution image.
        "sample_X": fit["sample_X"],
        # Normalization used by the fit and all full-resolution predictions.
        "centers": centers,
        "feature_means": feature_means,
        "center": fit["center"],
        "scale": fit["scale"],
        "feature_blur_sigma": cfg.feature_blur_sigma,
    }


@export_operation
def save_analysis_settings(cfg: Config, image_names: list[str], reference_index: int,
                           out_dir: Path, *, mask_mode: str = 'auto') -> None:
    payload = dict(riima_version=__version__, images=image_names, reference_index=reference_index,
                   mask_mode=mask_mode, optical=asdict(cfg), feature_space=dict(
                       feature_space='LabT_4N', feature_dimension=4 * len(image_names),
                       chromatic_heterogeneity_window=CHROMATIC_HETEROGENEITY_WINDOW,
                       normalization='median_iqr', distance='euclidean', feature_weights='unit'))
    staged_file(Path(out_dir) / 'analysis_settings.json').write_text(json.dumps(payload, indent=2), encoding='utf-8')


@export_operation
def save_session_outputs(prepared: dict, interpretation: Interpretation | None = None) -> None:
    """Export consistent raw clusters, manual interpretation and semantic families."""
    state = interpretation if interpretation is not None else prepared['interpretation']
    raw = state.raw
    interpreted, families = state.interpreted(), state.family_result()
    out = staged_directory(prepared['output'])
    out.mkdir(parents=True, exist_ok=True)
    np.save(out / 'clusters.npy', raw['labels'])
    np.save(out / 'interpreted_clusters.npy', interpreted['labels'])
    np.save(out / 'families.npy', families['family_map'])
    for filename, result in [('clusters.png', raw), ('interpreted_clusters.png', interpreted)]:
        _save_png_fast(render_optical_clusters(result['labels'], raw['valid'], result['palette']), out / filename)
    lut = np.vstack([np.array([[30, 30, 30]], dtype=np.uint8), families['palette']])
    _save_png_fast(lut[families['family_map']], out / 'families.png')
    _save_png_fast(raw['valid'].astype(np.uint8) * 255, out / 'analysis_mask.png')
    _save_png_fast(render_optical_confidence(raw['confidence'], raw['valid']), out / 'optical_confidence.png')
    (out / 'interpretation.json').write_text(json.dumps(state.to_dict(), indent=2, ensure_ascii=False), encoding='utf-8')
    with (out / 'cluster_statistics.csv').open('w', newline='', encoding='utf-8') as handle:
        writer = csv.writer(handle)
        writer.writerow(['cluster', 'interpreted_cluster', 'pixels', 'area_percent_valid_image',
                         'mean_confidence', 'name', 'color_rgb', 'centroid_normalized_4N', 'feature_mean_4N'])
        total = max(int(raw['valid'].sum()), 1)
        for cluster in range(raw['clusters']):
            mask = raw['valid'] & (raw['labels'] == cluster)
            count = int(mask.sum())
            group = next(i for i, g in enumerate(state.groups) if cluster in g)
            centroid = np.asarray(raw['centers'][cluster]).tolist() if 'centers' in raw else []
            mean = np.asarray(raw['feature_means'][cluster]).tolist() if 'feature_means' in raw else []
            writer.writerow([cluster, group, count, 100.0 * count / total,
                             float(raw['confidence'][mask].mean()) if count else 0.0,
                             state.names.get(group, ''), json.dumps(interpreted['palette'][group].tolist()),
                             json.dumps(centroid), json.dumps(mean)])
    with (out / 'family_statistics.csv').open('w', newline='', encoding='utf-8') as handle:
        writer = csv.writer(handle)
        writer.writerow(['family', 'clusters', 'pixels', 'area_percent_valid_image', 'name', 'color_rgb'])
        for row in families['stats']:
            writer.writerow([row['family'], json.dumps(row['source_clusters']), row['pixels'],
                             row['area'], row['name'], json.dumps(row['color'])])
    save_analysis_settings(prepared['config'], raw['image_names'], raw['reference_index'], out,
                           mask_mode=raw.get('mask_mode', 'auto'))


@export_operation
def prepare_analysis(image_paths: list[str], output: str, reference_index: int = 0,
                     optical_cfg: Config | None = None, *, diagnostics: bool = False,
                     mask_mode: str = 'auto', stage_callback=None, publish_outputs: bool = True) -> dict:
    """Compute clusters once; optionally publish outputs, including diagnostics.

    With ``publish_outputs=False``, preparation and interpretation remain entirely
    in memory; the output path is retained for later explicit exports.
    """
    cfg = optical_cfg or Config()
    if stage_callback:
        stage_callback('Registrando e analisando imagens…')
    optical = run_optical_analysis(image_paths, output, cfg, reference_index=reference_index,
                                   diagnostics=diagnostics, mask_mode=mask_mode,
                                   publish_outputs=publish_outputs)
    if stage_callback:
        stage_callback('Preparando interpretação de clusters…')
    return dict(output=str(output), optical=optical, config=cfg, interpretation=Interpretation(optical))


@export_operation
def run_analysis(image_paths: list[str], output: str, optical_cfg: Config | None = None,
                 reference_index: int = 0, *, diagnostics: bool = False,
                 mask_mode: str = 'auto', stage_callback=None) -> dict:
    prepared = prepare_analysis(image_paths, output, reference_index, optical_cfg,
                                diagnostics=diagnostics, mask_mode=mask_mode, stage_callback=stage_callback)
    save_session_outputs(prepared)
    return prepared


def active_reference_index(active_flags: list[bool], reference_index: int) -> int:
    """Rebase a reference index from the full picker list to its position among active images."""
    return sum(1 for a in active_flags[:reference_index] if a)


def normalize_image_selection(
    paths: list[str], active_flags: list[bool], reference_index: int
) -> tuple[list[str], int]:
    """Filter to active images and rebase the reference index to the filtered list.

    Raises if the reference is out of range, inactive, or if no image is active.
    """
    if not (0 <= reference_index < len(paths)):
        raise ValueError(f"Índice de referência fora do intervalo 0..{len(paths) - 1}.")
    if not active_flags[reference_index]:
        raise ValueError("A imagem de referência precisa estar marcada como ativa (Usar).")
    if not any(active_flags):
        raise ValueError("Marque ao menos uma imagem como ativa (Usar).")
    selected = [p for p, a in zip(paths, active_flags) if a]
    return selected, active_reference_index(active_flags, reference_index)


def registration_quality_label(score_before: float, score_after: float) -> str:
    """A conservative, purely advisory registration-quality label for the GUI.

    This never affects the pipeline; it only helps the user notice a pair that likely
    needs a different reference image or manual review. Real PPL/cPol pairs often have
    modest absolute edge-NCC scores (different modalities look very different), so the
    floor is deliberately low; a registration is flagged mainly when it barely improved
    edge agreement at all.
    """
    improvement = float(score_after) - float(score_before)
    if float(score_after) < 0.12 or improvement < 0.01:
        return "registro fraco"
    return "registro OK"


# ---------------------------------------------------------------------------
# 12. Optional, on-demand diagnostics (decision support, not automatic truth)
# ---------------------------------------------------------------------------
#
# Everything in this section is opt-in: nothing here runs during a normal analysis, and
# none of it feeds back into the scientific pipeline. It exists to help a user judge
# whether a chosen optical-cluster count is reasonable, never to declare a single
# "correct" K. Statistical cluster separation is not mineralogical truth.

def _pairwise_sq_dist(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Dense squared-Euclidean distance matrix; only ever called on small sub-samples."""
    a2 = np.sum(a * a, axis=1)[:, None]
    b2 = np.sum(b * b, axis=1)[None, :]
    return np.maximum(a2 + b2 - 2.0 * (a @ b.T), 0.0)


def _sampled_silhouette(X: np.ndarray, labels: np.ndarray, k: int, *, sample_size: int, seed: int) -> float:
    """Silhouette coefficient (Rousseeuw, 1987) on a bounded random sub-sample.

    A full silhouette needs an N x N distance matrix; for a k-means sample of tens of
    thousands of points can exceed available memory. A fixed-size sub-sample keeps the cost
    predictable regardless of how large ``X`` is.
    """
    n = len(X)
    if k < 2 or n < 3:
        return float("nan")
    m = min(sample_size, n)
    rng = np.random.default_rng(seed)
    idx = rng.choice(n, size=m, replace=False)
    Xs, Ls = X[idx], labels[idx]
    present = np.unique(Ls)
    if len(present) < 2:
        return float("nan")
    dist = np.sqrt(_pairwise_sq_dist(Xs, Xs))
    scores = np.zeros(m, dtype=np.float64)
    for i in range(m):
        same = (Ls == Ls[i])
        same[i] = False
        if not np.any(same):
            scores[i] = 0.0
            continue
        a_i = dist[i, same].mean()
        b_i = np.inf
        for c in present:
            if c == Ls[i]:
                continue
            other = Ls == c
            if np.any(other):
                b_i = min(b_i, dist[i, other].mean())
        scores[i] = 0.0 if b_i == np.inf else (b_i - a_i) / max(a_i, b_i, 1e-12)
    return float(scores.mean())


def _calinski_harabasz(X: np.ndarray, labels: np.ndarray, centers: np.ndarray, inertia: float) -> float:
    """Calinski-Harabasz variance-ratio criterion (Calinski & Harabasz, 1974)."""
    n, k = len(X), len(centers)
    if k < 2 or k >= n:
        return float("nan")
    grand_mean = X.mean(axis=0)
    between = 0.0
    for c in range(k):
        n_c = int(np.sum(labels == c))
        if n_c:
            between += n_c * float(np.sum((centers[c] - grand_mean) ** 2))
    if inertia <= 1e-12:
        return float("nan")
    return (between / max(k - 1, 1)) / (inertia / max(n - k, 1))


def _davies_bouldin(X: np.ndarray, labels: np.ndarray, centers: np.ndarray) -> float:
    """Euclidean Davies-Bouldin with mean centroid distances; lower is more separated.

    Preserve the raw zero-distance exclusion convention. Degenerate fits emit a
    warning: even a returned zero must not be interpreted as maximum separation.
    """
    k = len(centers)
    if k < 2:
        return float("nan")
    if _cluster_diagnostic_issues(labels, centers):
        warnings.warn('Davies-Bouldin em solução degenerada: o índice bruto não '
                      'demonstra separação máxima.', RuntimeWarning, stacklevel=2)
    spread = np.zeros(k, dtype=np.float64)
    for c in range(k):
        pts = X[labels == c]
        if len(pts):
            spread[c] = float(np.mean(np.sqrt(np.sum((pts - centers[c]) ** 2, axis=1))))
    center_dist = np.sqrt(_pairwise_sq_dist(centers, centers))
    worst = np.zeros(k, dtype=np.float64)
    for c in range(k):
        ratios = [
            (spread[c] + spread[c2]) / center_dist[c, c2]
            for c2 in range(k) if c2 != c and center_dist[c, c2] > 1e-12
        ]
        worst[c] = max(ratios) if ratios else 0.0
    return float(worst.mean())


def evaluate_cluster_counts(
    sample_X: np.ndarray, k_values: list[int], *, attempts: int = 3, seed: int = 13,
    silhouette_sample_size: int = 1500,
) -> dict:
    """Evaluate a small range of candidate optical-cluster counts on the SAME cached,
    standardized k-means sample already used by the real analysis (never the
    full-resolution image, and never re-registered or re-extracted).

    Returns ``{"k_values": [...], "inertia": [...], "silhouette": [...],
    "calinski_harabasz": [...], "davies_bouldin": [...]}``, one entry per requested k, in
    the same order as ``k_values``. This is decision support only: it never selects or
    reports a single "optimal" K, and re-fitting a k for these metrics never mutates or
    replaces the analysis actually shown in the viewer; the caller decides whether to set
    ``Config.clusters`` to a candidate and explicitly re-run the analysis.

    ``diagnostic_valid`` and ``diagnostic_warnings`` contain one flag/message list
    per K. A false flag denotes coincident centers or empty fitted groups; raw
    indices retain their numerical conventions and must not be ranked/displayed as
    evidence of separation for that K. Pass this mask to recommend_cluster_count.
    """
    X = np.asarray(sample_X, dtype=np.float32)
    out = {"k_values": list(k_values), "inertia": [], "silhouette": [], "calinski_harabasz": [], "davies_bouldin": []}
    out["diagnostic_valid"], out["diagnostic_warnings"] = [], []
    for k in k_values:
        labels, centers = fit_kmeans(X, int(k), attempts, seed)
        issues = _cluster_diagnostic_issues(labels, centers)
        out["diagnostic_valid"].append(not issues)
        out["diagnostic_warnings"].append(issues)
        d1 = np.sum((X - centers[labels]) ** 2, axis=1)
        inertia = float(d1.sum())
        out["inertia"].append(inertia)
        out["silhouette"].append(_sampled_silhouette(X, labels, int(k), sample_size=silhouette_sample_size, seed=seed))
        out["calinski_harabasz"].append(_calinski_harabasz(X, labels, centers, inertia))
        out["davies_bouldin"].append(_davies_bouldin(X, labels, centers))
    return out


def recommend_cluster_count(k_values: list[int], silhouette: list[float], *,
                            diagnostic_valid: list[bool] | None = None) -> int | None:
    """K_sugerido = argmax_K S(K) over candidates already scored by
    :func:`evaluate_cluster_counts`, ties broken toward the smaller K.

    This is a suggestion, not a decision: a statistically better-separated K is not
    necessarily the mineralogically correct one. Returns ``None`` when every candidate's
    silhouette is undefined (NaN), e.g. when no K in the range yielded at least two
    populated clusters.

    An optional ``diagnostic_valid`` mask excludes flagged degenerate fits without
    changing their raw silhouette values or the existing scoring convention.
    """
    if diagnostic_valid is None:
        diagnostic_valid = [True] * len(k_values)
    elif len(diagnostic_valid) != len(k_values) or len(silhouette) != len(k_values):
        raise ValueError('diagnostic_valid deve ter uma entrada por K e silhueta.')
    candidates = [
        (int(k), float(s)) for k, s, valid in zip(k_values, silhouette, diagnostic_valid)
        if valid and s is not None and not np.isnan(s)
    ]
    if not candidates:
        return None
    best_score = max(s for _, s in candidates)
    return min(k for k, s in candidates if s == best_score)


def resolve_cli_images(
    ppl: str | None, cpol: str | None, images: list[str] | None, reference_index: int
) -> tuple[list[str], int]:
    """Resolve CLI image sources: repeated ``--image`` takes priority; else the legacy pair."""
    if images:
        return list(images), reference_index
    if ppl and cpol:
        return [ppl, cpol], reference_index
    raise ValueError("Provide --image (repeatable) or both --ppl and --cpol.")
