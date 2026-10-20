import sys
from pathlib import Path

import cv2
import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def _synthetic_texture(h=180, w=240):
    img = np.full((h, w, 3), 225, dtype=np.uint8)
    cv2.circle(img, (60, 70), 28, (80, 120, 160), -1)
    cv2.rectangle(img, (120, 35), (190, 95), (170, 90, 60), -1)
    cv2.line(img, (20, 150), (220, 115), (40, 60, 80), 5)
    cv2.putText(img, "A", (145, 155), cv2.FONT_HERSHEY_SIMPLEX, 1.4, (90, 160, 70), 3, cv2.LINE_AA)
    return img


def synthetic_registration_image(h=180, w=240):
    """Alias used by multimodal registration tests; same texture generator."""
    return _synthetic_texture(h, w)


def synthetic_color_texture_image(h=140, w=200):
    """A distinct texture generator for feature-source tests (kept separate from
    the registration fixture so both concerns can evolve independently)."""
    img = np.full((h, w, 3), 200, dtype=np.uint8)
    cv2.circle(img, (70, 60), 24, (60, 140, 90), -1)
    cv2.rectangle(img, (110, 20), (170, 80), (150, 70, 130), -1)
    cv2.line(img, (10, 120), (190, 90), (30, 90, 150), 4)
    return img


def warp_test_image(img, dx=0.0, dy=0.0, angle=0.0):
    """Apply a small Euclidean warp (rotation about center + translation)."""
    h, w = img.shape[:2]
    center = (w / 2.0, h / 2.0)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    M[:, 2] += np.array([dx, dy], dtype=np.float64)
    return cv2.warpAffine(
        img, M.astype(np.float32), (w, h),
        flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0,
    )


def write_rgb_images(tmp_path, images):
    paths = []
    for i, img in enumerate(images):
        p = tmp_path / f"im{i}.png"
        cv2.imwrite(str(p), cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
        paths.append(str(p))
    return paths


def make_small_realistic_pair():
    """A small PPL/cPol-like pair with a slight registration offset, for fast pipeline tests."""
    ppl = _synthetic_texture(120, 160)
    cpol = warp_test_image(ppl, dx=2.0, dy=-1.0, angle=0.0)
    return ppl, cpol


def test_specimen_mask_excludes_dark_frame_but_keeps_enclosed_dark_grain():
    from riima.core import specimen_area_mask

    cpol = np.full((120, 160, 3), 120, dtype=np.uint8)
    cpol[:10, :] = 2
    cpol[-10:, :] = 2
    cpol[:, :12] = 2
    cpol[:, -12:] = 2
    # Dark grain inside the specimen must remain valid.
    cv2.circle(cpol, (80, 60), 18, (2, 2, 2), -1)

    mask = specimen_area_mask(cpol)
    assert mask.dtype == bool
    assert not mask[2, 2]
    assert not mask[115, 80]
    assert mask[60, 80]
    assert mask[60, 40]


def test_specimen_mask_keeps_dark_but_colored_mineral_touching_the_edge():
    """A dark mineral that touches (not encloses inside a
    single bright contour) the specimen's outer edge must not be silently dropped just
    because a naive brightness-only threshold merges it with the background frame."""
    from riima.core import specimen_area_mask

    img = np.full((140, 180, 3), (150, 150, 150), dtype=np.uint8)
    # Neutral, desaturated dark scanner frame.
    img[:14, :] = (4, 4, 4)
    img[-14:, :] = (4, 4, 4)
    img[:, :14] = (4, 4, 4)
    img[:, -14:] = (4, 4, 4)
    # A dark-but-colored mineral grain touching the specimen's inner edge (adjacent to,
    # and, under a brightness-only threshold, contiguous with, the frame, so it is not
    # "enclosed" by a single bright contour around it). Dark enough to fall below the
    # brightness threshold on its own, but strongly saturated (unlike the neutral frame).
    img[40:70, 14:34] = (18, 3, 3)

    mask = specimen_area_mask(img)
    assert mask.dtype == bool
    assert not mask[5, 5]  # scanner frame excluded
    assert mask[55, 20]  # the dark-but-colored mineral must remain valid
    assert mask[70, 90]  # ordinary bright interior remains valid


def test_build_analysis_mask_rescues_pixel_dark_in_one_modality_but_bright_in_another():
    """Intersecting independent per-image specimen masks would
    drop a pixel that is clearly valid specimen in at least one active modality merely
    because other modalities happen to render that mineral very dark (and touching the
    image border, so a single-image contour heuristic treats it as background) there."""
    from riima.core import build_analysis_mask

    h, w = 100, 140
    region = (slice(30, 60), slice(0, 20))  # touches the left border in b/c
    img_a = np.full((h, w, 3), (150, 150, 150), dtype=np.uint8)  # bright (valid) in modality A
    img_b = np.full((h, w, 3), (150, 150, 150), dtype=np.uint8)
    img_c = np.full((h, w, 3), (150, 150, 150), dtype=np.uint8)
    img_b[region] = (5, 5, 5)  # ...but rendered very dark (and desaturated) in modality B...
    img_c[region] = (5, 5, 5)  # ...and modality C.
    overlap = np.ones((h, w), dtype=bool)

    valid = build_analysis_mask([img_a, img_b, img_c], overlap)
    assert valid[45, 10], "a mineral visible in any active modality must not be excluded"


def test_red_annotation_mask_spares_interior_mineral_but_flags_scale_bar_and_stamp():
    """A red/pink mineral-like patch must not be treated the
    same as a scale bar or a bordered annotation stamp just because it passes a raw RGB
    threshold: shape (elongation) and border proximity distinguish them."""
    from riima.core import red_annotation_mask

    img = np.full((160, 220, 3), (150, 150, 150), dtype=np.uint8)
    # An ordinary, roughly equant red/pink mineral-like patch, away from the border.
    cv2.circle(img, (110, 80), 16, (200, 40, 40), -1)
    # A thin, elongated scale-bar-like line.
    img[150:153, 20:170] = (220, 20, 20)
    # A small red stamp/label touching the image border (not elongated, but border-adjacent).
    img[0:10, 0:10] = (210, 30, 30)

    mask = red_annotation_mask(img)
    assert not mask[80, 110], "an equant interior red mineral patch must not be masked as an annotation"
    assert mask[151, 90], "a thin elongated scale-bar-like line must still be masked"
    assert mask[4, 4], "a red annotation touching the image border must still be masked"


def test_registration_returns_overlap_mask_and_improves_edge_alignment():
    from riima.core import register_pair, structural_edge

    ref = _synthetic_texture()
    h, w = ref.shape[:2]
    center = (w / 2.0, h / 2.0)
    # Create the moving image with a small realistic scanner misalignment.
    M = cv2.getRotationMatrix2D(center, 0.35, 1.0)
    M[:, 2] += np.array([5.0, -3.0], dtype=np.float32)
    moving = cv2.warpAffine(ref, M, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)

    aligned, info, overlap = register_pair(ref, moving)
    before = np.mean(np.abs(structural_edge(ref) - structural_edge(moving)))
    after = np.mean(np.abs(structural_edge(ref)[overlap] - structural_edge(aligned)[overlap]))

    assert overlap.shape == ref.shape[:2]
    assert overlap.dtype == bool
    assert 0 < overlap.sum() < overlap.size
    assert after < before * 0.75
    assert info["method"] in {"phase_translation", "ecc_euclidean"}
    assert "score_before" in info and "score_after" in info
    assert info["score_after"] >= info["score_before"]


def test_analysis_valid_mask_combines_specimen_overlap_and_annotations():
    from riima.core import analysis_valid_mask

    ppl = np.full((80, 120, 3), 230, dtype=np.uint8)
    cpol = np.full((80, 120, 3), 90, dtype=np.uint8)
    cpol[:8, :] = 0
    cpol[-8:, :] = 0
    cpol[:, :8] = 0
    cpol[:, -8:] = 0
    # Red annotation inside specimen.
    ppl[35:40, 50:70] = [255, 0, 0]
    overlap = np.ones((80, 120), dtype=bool)
    overlap[:, :3] = False

    valid = analysis_valid_mask(ppl, cpol, overlap)
    assert not valid[2, 2]
    assert not valid[37, 60]
    assert not valid[40, 1]
    assert valid[40, 90]


def test_run_saves_analysis_mask_and_registration_metadata(tmp_path):
    from riima.core import Config, run_optical_analysis

    ref = _synthetic_texture(120, 160)
    moving = ref.copy()
    moving[:6, :] = 0
    moving[-6:, :] = 0
    # Keep dimensions identical; small translation exercises overlap handling.
    M = np.float32([[1, 0, 3], [0, 1, -2]])
    moving = cv2.warpAffine(moving, M, (160, 120), borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    ppl_path = tmp_path / "ppl.png"
    cpol_path = tmp_path / "cpol.png"
    cv2.imwrite(str(ppl_path), cv2.cvtColor(ref, cv2.COLOR_RGB2BGR))
    cv2.imwrite(str(cpol_path), cv2.cvtColor(moving, cv2.COLOR_RGB2BGR))

    out = tmp_path / "out"
    result = run_optical_analysis(
        [str(ppl_path), str(cpol_path)], str(out), Config(clusters=4, sample_pixels=1200), reference_index=0
    )

    assert (out / "analysis_mask.png").exists()
    registration_csv = (out / "registration.csv").read_text(encoding="utf-8")
    assert "angle_deg" in registration_csv  # structured registration metadata, no free-text run_info.txt
    method = result["registered_images"][1]["registration"]["method"]
    assert method in {"phase_translation", "ecc_euclidean"}
    assert 0.0 <= float(result["valid"].mean()) <= 1.0


def test_optical_run_is_deterministic_for_same_inputs_and_config(tmp_path):
    from riima.core import Config, run_optical_analysis

    ppl = _synthetic_texture(120, 160)
    cpol = ppl.copy()
    ppl_path = tmp_path / "ppl_repeat.png"
    cpol_path = tmp_path / "cpol_repeat.png"
    cv2.imwrite(str(ppl_path), cv2.cvtColor(ppl, cv2.COLOR_RGB2BGR))
    cv2.imwrite(str(cpol_path), cv2.cvtColor(cpol, cv2.COLOR_RGB2BGR))

    cfg = Config(clusters=4, sample_pixels=1400, seed=13)
    a = run_optical_analysis([str(ppl_path), str(cpol_path)], str(tmp_path / "a"), cfg, reference_index=0)
    b = run_optical_analysis([str(ppl_path), str(cpol_path)], str(tmp_path / "b"), cfg, reference_index=0)

    assert np.array_equal(a["valid"], b["valid"])
    assert np.array_equal(a["labels"], b["labels"])
    assert np.array_equal(a["confidence"], b["confidence"])


def test_streaming_two_nearest_matches_dense_reference():
    from riima.core import nearest_two_centers
    rng = np.random.default_rng(123)
    X = rng.normal(size=(257, 8)).astype(np.float32)
    centers = rng.normal(size=(7, 8)).astype(np.float32)
    labels, d1, d2 = nearest_two_centers(X, centers)

    dense = np.sum((X[:, None, :] - centers[None, :, :]) ** 2, axis=2, dtype=np.float32)
    order = np.argsort(dense, axis=1)
    ref_labels = order[:, 0].astype(np.uint8)
    ref_d1 = np.take_along_axis(dense, order[:, :1], axis=1)[:, 0]
    ref_d2 = np.take_along_axis(dense, order[:, 1:2], axis=1)[:, 0]

    assert np.array_equal(labels, ref_labels)
    assert np.allclose(d1, ref_d1, rtol=0, atol=1e-6)
    assert np.allclose(d2, ref_d2, rtol=0, atol=1e-6)


def test_register_images_registers_every_image_directly_to_reference(tmp_path):
    from riima import core as riima_core
    ref = synthetic_registration_image()
    m1 = warp_test_image(ref, dx=4.0, dy=-3.0, angle=0.10)
    m2 = warp_test_image(ref, dx=-7.0, dy=2.0, angle=-0.15)
    paths = write_rgb_images(tmp_path, [ref, m1, m2])

    result = riima_core.register_images(paths, reference_index=0)

    assert result["reference_index"] == 0
    assert len(result["images"]) == 3
    assert result["images"][0]["registration"]["is_reference"] is True
    assert result["images"][1]["registration"]["reference_index"] == 0
    assert result["images"][2]["registration"]["reference_index"] == 0
    assert result["common_overlap"].shape == ref.shape[:2]
    assert result["common_overlap"].dtype == bool


def test_register_images_common_overlap_is_intersection(tmp_path):
    from riima import core as riima_core
    ref = synthetic_registration_image()
    shifted = warp_test_image(ref, dx=18.0, dy=0.0, angle=0.0)
    paths = write_rgb_images(tmp_path, [ref, shifted])
    result = riima_core.register_images(paths, reference_index=0)
    expected = result["images"][0]["overlap"] & result["images"][1]["overlap"]
    assert np.array_equal(result["common_overlap"], expected)


def test_two_image_registration_matches_existing_register_pair(tmp_path):
    from riima import core as riima_core
    ref = synthetic_registration_image()
    mov = warp_test_image(ref, dx=5.0, dy=-2.0, angle=0.12)
    paths = write_rgb_images(tmp_path, [ref, mov])
    old_rgb, old_meta, old_overlap = riima_core.register_pair(ref, mov)
    new = riima_core.register_images(paths, reference_index=0)
    assert np.array_equal(new["images"][1]["rgb"], old_rgb)
    assert np.array_equal(new["images"][1]["overlap"], old_overlap)


def test_one_image_features_are_lab_star_plus_chromatic_heterogeneity():
    """The optical feature space is exactly [L*, a*, b*, T] per image, dim = 4N."""
    from riima import core as riima_core
    img = synthetic_color_texture_image()
    sources = riima_core.extract_optical_features([img], 0.8)
    assert len(sources) == 4
    assert all(s.shape == img.shape[:2] for s in sources)


def test_n_image_features_are_exactly_4n_and_symmetric():
    """dim(x) = 4N for N = 1, 2, 3, and every image contributes identically: two images
    with swapped positions produce the same four feature planes each, just reordered
    (no image is privileged as a "reference" in the feature space itself)."""
    from riima import core as riima_core
    a = synthetic_color_texture_image()
    b = np.roll(a, 2, axis=1)

    for n, imgs in ((1, [a]), (2, [a, b]), (3, [a, b, a])):
        sources = riima_core.extract_optical_features(imgs, 0.8)
        assert len(sources) == 4 * n

    forward = riima_core.extract_optical_features([a, b], 0.8)
    swapped = riima_core.extract_optical_features([b, a], 0.8)
    # Image a's own [L*,a*,b*,T] block is identical whether it is passed first or second.
    for x, y in zip(forward[0:4], swapped[4:8]):
        assert np.array_equal(x, y)
    for x, y in zip(forward[4:8], swapped[0:4]):
        assert np.array_equal(x, y)


def test_lab_star_channels_use_canonical_cielab_units():
    """L* in [0, 100]; a*/b* centered on 0 (roughly [-128, 127]), via the exact affine
    rescaling of OpenCV's 8-bit Lab output, not raw 0..255 codes."""
    from riima.core import lab_star_channels

    img = synthetic_color_texture_image()
    L, a, b = lab_star_channels(img, blur_sigma=0.0)
    assert L.min() >= 0.0 and L.max() <= 100.0 + 1e-3
    assert a.min() >= -128.0 - 1e-3 and a.max() <= 127.0 + 1e-3
    assert b.min() >= -128.0 - 1e-3 and b.max() <= 127.0 + 1e-3

    import cv2
    lab8 = cv2.cvtColor(img, cv2.COLOR_RGB2LAB).astype(np.float32)
    assert np.allclose(L, lab8[..., 0] * (100.0 / 255.0), atol=1e-3)
    assert np.allclose(a, lab8[..., 1] - 128.0, atol=1e-3)
    assert np.allclose(b, lab8[..., 2] - 128.0, atol=1e-3)


def test_local_chromatic_heterogeneity_matches_manual_variance_formula():
    """T(p) = sqrt(var(L*) + var(a*) + var(b*)) in a small window, checked against a
    direct (non-boxFilter) windowed-variance computation on a tiny synthetic array."""
    from riima.core import local_chromatic_heterogeneity

    rng = np.random.default_rng(3)
    h, w = 15, 15
    L = rng.normal(50.0, 5.0, size=(h, w)).astype(np.float32)
    a = rng.normal(0.0, 3.0, size=(h, w)).astype(np.float32)
    b = rng.normal(0.0, 3.0, size=(h, w)).astype(np.float32)
    window = 5

    got = local_chromatic_heterogeneity(L, a, b, window=window)

    r = window // 2
    y, x = 7, 7  # a pixel far enough from the border for a full window
    def local_var(channel):
        patch = channel[y - r:y + r + 1, x - r:x + r + 1]
        return float(np.var(patch))
    expected = np.sqrt(local_var(L) + local_var(a) + local_var(b))
    assert got[y, x] == pytest.approx(expected, abs=1e-3)


def test_standardized_sample_uses_robust_median_iqr_not_mean_std():
    """A single extreme outlier pixel must not distort the normalization scale the way
    it would under mean/std standardization: this is the whole point of switching to
    robust (median/IQR) normalization for RIIMA 1.0."""
    from riima.core import standardized_sample

    rng = np.random.default_rng(9)
    base = rng.normal(0.0, 1.0, size=(200, 200)).astype(np.float32)
    sources = [base.copy()]
    valid = np.ones((200, 200), dtype=bool)
    # A single wildly extreme pixel: negligible effect on the median/IQR, but would
    # inflate a mean/std-based scale substantially.
    sources[0][0, 0] = 10_000.0

    X, idx, center, scale = standardized_sample(sources, valid, n=39_999, seed=1)
    assert center[0] == pytest.approx(float(np.median(base)), abs=0.05)
    q25, q75 = np.percentile(base, [25.0, 75.0])
    assert scale[0] == pytest.approx(float(q75 - q25), abs=0.05)


def test_standardized_sample_and_predict_chunked_are_reproducible():
    from riima.core import standardized_sample, predict_chunked, fit_kmeans

    rng = np.random.default_rng(5)
    sources = [rng.normal(size=(20, 24)).astype(np.float32) for _ in range(8)]  # 2-image 4N case
    valid = np.ones((20, 24), dtype=bool)

    sample_a, idx_a, center_a, scale_a = standardized_sample(sources, valid, 200, seed=7)
    sample_b, idx_b, center_b, scale_b = standardized_sample(sources, valid, 200, seed=7)
    assert np.array_equal(sample_a, sample_b)
    assert np.array_equal(idx_a, idx_b)

    labels_a, centers_a = fit_kmeans(sample_a, 3, 2, seed=7)
    conf_a = predict_chunked(sources, center_a, scale_a, centers_a, 100)
    conf_b = predict_chunked(sources, center_b, scale_b, centers_a, 100)
    assert np.array_equal(conf_a[0], conf_b[0])
    assert np.allclose(conf_a[1], conf_b[1], atol=0, rtol=0)


def test_run_multimodal_saves_registration_csv_and_run_info(tmp_path):
    from riima import core as riima_core
    ppl, cpol = make_small_realistic_pair()
    p1, p2 = write_rgb_images(tmp_path, [ppl, cpol])
    cfg = riima_core.Config(clusters=4, sample_pixels=2000)
    out = tmp_path / "out"
    result = riima_core.run_optical_analysis([p1, p2], str(out), cfg, reference_index=0)
    assert (out / "registration.csv").exists()
    rows = (out / "registration.csv").read_text(encoding="utf-8").strip().splitlines()
    assert len(rows) == 1 + 2  # header + 2 images
    assert result["reference_index"] == 0
    assert result["image_names"] == [Path(p1).name, Path(p2).name]
    assert result["sample_X"].shape[1] == 8  # dim(x) = 4N for N=2 images


def test_run_multimodal_supports_a_single_image():
    from riima import core as riima_core
    ppl, _ = make_small_realistic_pair()
    import tempfile
    with tempfile.TemporaryDirectory() as td:
        tmp_path = Path(td)
        [p1] = write_rgb_images(tmp_path, [ppl])
        cfg = riima_core.Config(clusters=4, sample_pixels=1500)
        result = riima_core.run_optical_analysis([p1], str(tmp_path / "out"), cfg, reference_index=0)
        assert result["labels"].shape == ppl.shape[:2]
        assert result["valid"].any()


def test_resolve_cli_images_supports_old_pair_and_new_multi_image_forms():
    from riima.core import resolve_cli_images

    # Old pair form remains the default.
    paths, ref = resolve_cli_images(ppl="a.png", cpol="b.png", images=None, reference_index=0)
    assert paths == ["a.png", "b.png"]
    assert ref == 0

    # New repeated --image form.
    paths, ref = resolve_cli_images(ppl=None, cpol=None, images=["x.png", "y.png", "z.png"], reference_index=1)
    assert paths == ["x.png", "y.png", "z.png"]
    assert ref == 1

    # A single image is also valid.
    paths, ref = resolve_cli_images(ppl=None, cpol=None, images=["only.png"], reference_index=0)
    assert paths == ["only.png"]
    assert ref == 0


def test_feature_dimension_is_exactly_4n_for_several_image_counts():
    """Feature dimension is 4N, independent of the reference image."""
    from riima import core as riima_core
    img = synthetic_color_texture_image()
    for n in (1, 2, 3, 4):
        sources = riima_core.extract_optical_features([img] * n, 0.8)
        assert len(sources) == 4 * n


# test_uint8_overlay_blend_matches_reference_and_preserves_invalid moved to
# test_riima_viewer.py: blend_overlay_uint8 is a compositing utility, not core science.


def test_check_common_overlap_raises_below_the_hard_floor():
    from riima.core import Config, InsufficientOverlapError, check_common_overlap

    cfg = Config()
    tiny_overlap = np.zeros((10, 10), dtype=bool)
    tiny_overlap[:1, :1] = True  # 1% common overlap, well below the 30% floor
    with pytest.raises(InsufficientOverlapError):
        check_common_overlap(tiny_overlap, cfg)


def test_check_common_overlap_warns_between_floor_and_healthy_threshold():
    from riima.core import Config, check_common_overlap

    cfg = Config()
    overlap = np.zeros((10, 10), dtype=bool)
    overlap[:5, :] = True  # 50% common overlap: above the 30% floor, below the 60% warn line
    assert check_common_overlap(overlap, cfg) == "warn"


def test_check_common_overlap_is_healthy_at_full_coverage():
    from riima.core import Config, check_common_overlap

    cfg = Config()
    overlap = np.ones((10, 10), dtype=bool)
    assert check_common_overlap(overlap, cfg) == ""


_LEGACY_NAME_FRAGMENTS = ("petro_similarity", "grain_experimental", "PetroSimilarity", "_v04", "V5")


def test_run_analysis_output_directory_has_no_legacy_naming(tmp_path):
    """Standard outputs use RIIMA names and exclude display-only overlay files."""
    from riima import core as riima_core
    ref = _synthetic_texture(120, 160)
    moving = ref.copy()
    ppl_path = tmp_path / "ppl.png"
    cpol_path = tmp_path / "cpol.png"
    cv2.imwrite(str(ppl_path), cv2.cvtColor(ref, cv2.COLOR_RGB2BGR))
    cv2.imwrite(str(cpol_path), cv2.cvtColor(moving, cv2.COLOR_RGB2BGR))

    out = tmp_path / "out"
    riima_core.run_analysis(
        [str(ppl_path), str(cpol_path)], str(out),
        optical_cfg=riima_core.Config(clusters=4, sample_pixels=1200),
    )

    entries = list(out.iterdir())
    assert entries, "run_analysis produced no output files"
    for entry in entries:
        assert entry.is_file(), f"unexpected subdirectory in a single flat output dir: {entry}"
        for fragment in _LEGACY_NAME_FRAGMENTS:
            assert fragment not in entry.name, f"legacy name fragment {fragment!r} in filename {entry.name}"
        assert "overlay" not in entry.name, f"auto-generated overlay file should not exist: {entry.name}"
        if entry.suffix in {".csv", ".json", ".txt"}:
            text = entry.read_text(encoding="utf-8")
            for fragment in _LEGACY_NAME_FRAGMENTS:
                assert fragment not in text, f"legacy name fragment {fragment!r} found inside {entry.name}"


def test_insufficient_overlap_error_is_raised_by_the_full_pipeline(tmp_path):
    """A real displacement below the configured overlap floor refuses analysis."""
    from riima.core import Config, InsufficientOverlapError, run_optical_analysis

    ref = _synthetic_texture(200, 260)
    moving = warp_test_image(ref, dx=20., dy=-15.)
    ppl_path = tmp_path / "ppl.png"
    cpol_path = tmp_path / "cpol.png"
    cv2.imwrite(str(ppl_path), cv2.cvtColor(ref, cv2.COLOR_RGB2BGR))
    cv2.imwrite(str(cpol_path), cv2.cvtColor(moving, cv2.COLOR_RGB2BGR))

    with pytest.raises(InsufficientOverlapError):
        run_optical_analysis(
            [str(ppl_path), str(cpol_path)], str(tmp_path / "out"),
            Config(clusters=4, sample_pixels=1200, min_overlap_fraction=.95)
        )


def _two_well_separated_blobs(seed=0, n_per_blob=300, dim=4, spread=0.3, gap=12.0):
    rng = np.random.default_rng(seed)
    a = rng.normal(loc=0.0, scale=spread, size=(n_per_blob, dim)).astype(np.float32)
    b = rng.normal(loc=gap, scale=spread, size=(n_per_blob, dim)).astype(np.float32)
    return np.concatenate([a, b], axis=0)


def test_evaluate_cluster_counts_prefers_the_true_k_on_two_separated_blobs():
    """Known synthetic case: two well-separated, tight Gaussian blobs. k=2 must score
    clearly better (higher silhouette/Calinski-Harabasz, lower Davies-Bouldin, and a
    sharp inertia drop) than k=1 is invalid for silhouette/CH/DB) or an over-split k=6."""
    from riima.core import evaluate_cluster_counts

    X = _two_well_separated_blobs()
    result = evaluate_cluster_counts(X, [2, 3, 6], attempts=5, seed=13)
    assert result["k_values"] == [2, 3, 6]

    sil = dict(zip(result["k_values"], result["silhouette"]))
    ch = dict(zip(result["k_values"], result["calinski_harabasz"]))
    db = dict(zip(result["k_values"], result["davies_bouldin"]))
    inertia = dict(zip(result["k_values"], result["inertia"]))

    assert sil[2] > 0.9, "two tight, well-separated blobs should have near-ideal silhouette at k=2"
    assert sil[2] > sil[6]
    assert ch[2] > ch[6]
    assert db[2] < db[6]
    # Inertia must never increase with k for the same underlying k-means objective.
    assert inertia[2] >= inertia[3] >= inertia[6]


def test_evaluate_cluster_counts_returns_nan_metrics_gracefully_at_k_equal_1():
    """k=1 has no between-cluster structure; silhouette/CH/DB are mathematically
    undefined there and must come back as NaN rather than a fabricated number."""
    from riima.core import evaluate_cluster_counts
    import math

    X = _two_well_separated_blobs()
    result = evaluate_cluster_counts(X, [1], attempts=3, seed=13)
    assert math.isnan(result["silhouette"][0])
    assert math.isnan(result["calinski_harabasz"][0])
    assert math.isnan(result["davies_bouldin"][0])
    assert result["inertia"][0] > 0  # inertia alone is always well-defined, even at k=1


def test_recommend_cluster_count_picks_the_argmax_silhouette():
    from riima.core import evaluate_cluster_counts, recommend_cluster_count

    X = _two_well_separated_blobs()
    result = evaluate_cluster_counts(X, [2, 3, 6], attempts=5, seed=13)
    assert recommend_cluster_count(result["k_values"], result["silhouette"]) == 2


def test_recommend_cluster_count_breaks_ties_toward_the_smaller_k():
    from riima.core import recommend_cluster_count

    assert recommend_cluster_count([2, 3, 4], [0.5, 0.5, 0.1]) == 2
    assert recommend_cluster_count([4, 3, 2], [0.1, 0.5, 0.5]) == 2


def test_recommend_cluster_count_returns_none_when_every_silhouette_is_nan():
    from riima.core import recommend_cluster_count

    assert recommend_cluster_count([1], [float("nan")]) is None
    assert recommend_cluster_count([], []) is None


def test_sampled_silhouette_matches_dense_reference_on_a_small_case():
    """Cross-check the sampled silhouette implementation against a direct, textbook
    (dense, unsampled) computation of Rousseeuw's (1987) formula on a small case small
    enough to compute exactly by hand-equivalent brute force."""
    from riima.core import _sampled_silhouette

    rng = np.random.default_rng(7)
    a = rng.normal(0.0, 0.2, size=(15, 2)).astype(np.float32)
    b = rng.normal(6.0, 0.2, size=(15, 2)).astype(np.float32)
    X = np.concatenate([a, b], axis=0)
    labels = np.array([0] * 15 + [1] * 15, dtype=np.int32)

    got = _sampled_silhouette(X, labels, k=2, sample_size=30, seed=1)  # sample_size >= n: exact, not sampled

    dist = np.sqrt(((X[:, None, :] - X[None, :, :]) ** 2).sum(axis=2))
    scores = []
    for i in range(len(X)):
        same = (labels == labels[i]).copy(); same[i] = False
        a_i = dist[i, same].mean()
        other = labels != labels[i]
        b_i = dist[i, other].mean()
        scores.append((b_i - a_i) / max(a_i, b_i))
    expected = float(np.mean(scores))
    assert got == pytest.approx(expected, abs=1e-5)


def test_feature_plane_iterator_matches_materialized_features():
    """iter_optical_feature_planes is the single source of truth for the 4N feature
    space; extract_optical_features just materializes it into a list. Locks that they
    never drift apart (e.g. if one is memoized/cached and the other is not)."""
    from riima.core import iter_optical_feature_planes, extract_optical_features
    rng = np.random.default_rng(23)
    imgs = [rng.integers(0, 256, size=(30, 36, 3), dtype=np.uint8) for _ in range(3)]

    ref_sources = extract_optical_features(imgs, 0.8)
    got = list(iter_optical_feature_planes(imgs, 0.8))
    assert len(got) == len(ref_sources) == 12  # 4 * 3 images
    for a, b in zip(got, ref_sources):
        assert np.array_equal(a, b)

    single_ref = extract_optical_features([imgs[0]], 0.8)
    single_got = list(iter_optical_feature_planes([imgs[0]], 0.8))
    assert len(single_got) == len(single_ref) == 4
    for a, b in zip(single_got, single_ref):
        assert np.array_equal(a, b)


def test_prepare_analysis_two_image_path_is_stable_across_repeated_calls(tmp_path):
    """Repeated two-image analyses produce reproducible integer outputs."""
    from riima import core as ge
    ppl = np.full((30, 40, 3), 210, np.uint8)
    ppl[5:15, 5:15] = [80, 120, 160]
    cpol = np.full((30, 40, 3), 90, np.uint8)
    cpol[5:15, 5:15] = [40, 60, 80]
    ppl_path = tmp_path / "ppl.png"
    cpol_path = tmp_path / "cpol.png"
    import cv2
    cv2.imwrite(str(ppl_path), cv2.cvtColor(ppl, cv2.COLOR_RGB2BGR))
    cv2.imwrite(str(cpol_path), cv2.cvtColor(cpol, cv2.COLOR_RGB2BGR))

    from riima.core import Config
    cfg = Config(clusters=4, sample_pixels=500)
    first = ge.prepare_analysis(
        [str(ppl_path), str(cpol_path)], str(tmp_path / "first"), reference_index=0, optical_cfg=cfg
    )
    second = ge.prepare_analysis(
        [str(ppl_path), str(cpol_path)], str(tmp_path / "second"), reference_index=0, optical_cfg=cfg
    )
    assert np.array_equal(first["optical"]["labels"], second["optical"]["labels"])


@pytest.mark.parametrize('valid_labels, expected', [([0, 0, 0], 0), ([1, 0, 1], 1)])
def test_regularization_votes_only_over_valid_neighbours(valid_labels, expected):
    from riima import core
    labels = np.ones((9, 9), np.uint8)
    valid = np.zeros(labels.shape, bool)
    valid[4, 3:6] = True
    labels[valid] = valid_labels
    confidence = np.ones(labels.shape, np.float32)
    confidence[4, 4] = 0
    edges = np.ones(labels.shape, np.float32)
    edges[4, 4] = 0
    cfg = core.Config(spatial_window=3)
    results = []
    for excluded_label in (0, 1):
        current = labels.copy()
        current[~valid] = excluded_label
        result = core.confidence_regularize(current, confidence, edges, 2, cfg, valid)
        results.append(int(result[4, 4]))
        np.testing.assert_array_equal(result[~valid], current[~valid])
    assert results == [expected, expected]


@pytest.mark.parametrize('textured_reference', [False, True])
def test_registration_without_structural_signal_preserves_field_with_explicit_fallback(textured_reference):
    from riima import core
    moving = np.full((100, 100, 3), 120, np.uint8)
    reference = moving.copy()
    if textured_reference:
        cv2.rectangle(reference, (15, 20), (70, 75), (20, 50, 90), -1)
    aligned, info, overlap = core.register_pair(reference, moving)
    assert info['translation'] == (0., 0.)
    assert overlap.all()
    np.testing.assert_array_equal(aligned, moving)
    assert info['method'] == 'identity_fallback'
    assert info['fallback_reason'] == 'no_structural_signal'
    assert info['registration_warning']


@pytest.mark.parametrize('shift, response', [((12., 7.), 0.), ((12., 7.), float('nan')),
                                             ((float('nan'), 7.), .9)])
def test_registration_rejects_uninformative_phase_estimate(monkeypatch, shift, response):
    from riima import core
    rgb = synthetic_registration_image()
    # Simulate the dependency's noninformative output on a textured input.
    monkeypatch.setattr(core.cv2, 'phaseCorrelate', lambda *args: (shift, response))
    aligned, info, overlap = core.register_pair(rgb, rgb.copy())
    assert info['translation'] == (0., 0.)
    assert overlap.all()
    np.testing.assert_array_equal(aligned, rgb)
    assert info['method'] == 'identity_fallback'
    assert info['fallback_reason'] == 'uninformative_phase_correlation'
    assert info['registration_warning']


def test_registration_csv_preserves_fallback_method_reason_and_warning(tmp_path):
    import csv
    from riima import core
    from PIL import Image

    paths = []
    for i in range(2):
        path = tmp_path / f'constant_{i}.png'
        Image.fromarray(np.full((100, 100, 3), 120, np.uint8)).save(path)
        paths.append(str(path))
    registered = core.register_images(paths)
    core.save_registration_csv(registered['images'], tmp_path)
    rows = list(csv.DictReader((tmp_path / 'registration.csv').open(encoding='utf-8')))
    assert rows[1]['method'] == 'identity_fallback'
    assert rows[1]['fallback_reason'] == 'no_structural_signal'
    assert rows[1]['registration_warning']
    assert registered['common_overlap'].all()


@pytest.mark.parametrize('pattern', ['half', 'periodic'])
def test_registration_preserves_identity_when_ambiguous_shift_does_not_improve(pattern):
    from riima import core
    rgb = np.full((32, 32, 3), 50, np.uint8)
    if pattern == 'half':
        rgb[16:] = 200
    else:
        rgb[8:16] = rgb[24:] = 200
    aligned, info, overlap = core.register_pair(rgb, rgb.copy())
    assert info['translation'] == (0., 0.)
    assert overlap.all()
    np.testing.assert_array_equal(aligned, rgb)
    assert info['method'] == 'identity_fallback'
    assert info['fallback_reason'] == 'no_improvement'
    assert info['registration_warning']
    assert info['score_before'] == pytest.approx(1.)
    assert info['score_after'] == pytest.approx(1.)


def test_davies_bouldin_uses_mean_euclidean_distance_to_centroid():
    from riima import core
    X = np.array([[0], [0], [6], [10], [10], [16]], np.float32)
    labels = np.array([0, 0, 0, 1, 1, 1], np.int32)
    centers = np.array([[2], [12]], np.float32)
    # Each spread is (2 + 2 + 4) / 3, centroid separation is 10.
    assert core._davies_bouldin(X, labels, centers) == pytest.approx(8 / 15)


def test_read_rgb_supports_unicode_paths_on_windows(tmp_path):
    from riima import core
    from PIL import Image

    directory = tmp_path / 'diretório_á_ç'
    directory.mkdir()
    path = directory / 'lâmina_á_ç.png'
    rgb = np.arange(24 * 32 * 3, dtype=np.uint16).reshape(24, 32, 3).astype(np.uint8)
    Image.fromarray(rgb).save(path)
    np.testing.assert_array_equal(core.read_rgb(path), rgb)


@pytest.mark.parametrize('payload', [None, b'', b'not an image'])
def test_read_rgb_rejects_missing_or_invalid_file(tmp_path, payload):
    from riima import core
    path = tmp_path / 'invalid.png'
    if payload is not None:
        path.write_bytes(payload)
    with pytest.raises(FileNotFoundError):
        core.read_rgb(path)


def test_png_export_supports_unicode_path_and_reports_write_failure(tmp_path):
    from riima import core
    from PIL import Image

    directory = tmp_path / 'diretório_á_ç'
    directory.mkdir()
    rgb = np.arange(24 * 32 * 3, dtype=np.uint16).reshape(24, 32, 3).astype(np.uint8)
    path = directory / 'saída_á_ç.png'
    core._save_png_fast(rgb, path)
    np.testing.assert_array_equal(np.asarray(Image.open(path)), rgb)
    with pytest.raises(OSError):
        core._save_png_fast(rgb, directory / 'missing_parent' / 'failed.png')
