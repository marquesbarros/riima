import dataclasses
import sys
from pathlib import Path

import numpy as np


from riima import viewer
def test_workspace_mask_uses_current_group_union_and_excludes_rejected_pixels(tmp_path):
    from test_riima_review import prepared_dataset
    prepared = prepared_dataset(tmp_path, count=1, clusters=3)
    state = prepared['review']
    prepared['interpretation'].merge_clusters([0, 1])
    state.sync_groups(prepared['interpretation'].groups)
    state.annotate(0, 0, [212], 1)
    state.annotate(0, 0, [195], -1)
    state.refine_pending()
    raw = state.items[0]['initial'].copy()
    mask = viewer.workspace_mask(prepared, [0])
    np.testing.assert_array_equal(mask, state.items[0]['valid'] & (state.items[0]['labels'] == 0))
    assert not mask.flat[195]
    rgb = viewer.render_workspace(prepared, 'Máscara selecionada', selected_groups=[0])
    np.testing.assert_array_equal(rgb[..., 0], mask.astype(np.uint8) * 255)
    assert np.array_equal(rgb[..., 0], rgb[..., 1])
    np.testing.assert_array_equal(state.items[0]['initial'], raw)


def test_workspace_sources_remain_fixed_as_groups_are_organized_and_refined(tmp_path):
    from test_riima_review import prepared_dataset
    prepared = prepared_dataset(tmp_path)
    expected = ['original', 'classification', 'selection_mask']
    assert [entry['id'] for entry in viewer.workspace_sources(prepared, [0])] == expected
    prepared['interpretation'].merge_clusters([0, 1])
    prepared['review'].sync_groups(prepared['interpretation'].groups)
    prepared['review'].approve(0)
    assert [entry['id'] for entry in viewer.workspace_sources(prepared, [0])] == expected
    prepared['review'].selected_image = 1
    for entry in viewer.workspace_sources(prepared, [0]):
        assert entry['render']().shape == prepared['review'].items[1]['rgb'].shape


def test_pending_examples_render_on_mask_and_hide_after_update_and_resume(tmp_path):
    from test_riima_review import prepared_dataset
    from riima import review
    prepared = prepared_dataset(tmp_path)
    state = prepared['review']
    state.annotate(0, 0, [195], -1)
    kwargs = dict(selected_groups=[0], annotation_group=0)
    rgb = viewer.render_workspace(prepared, 'Máscara selecionada', **kwargs)
    np.testing.assert_array_equal(rgb.reshape(-1, 3)[195], [255, 70, 70])
    state.refine_pending()
    path = tmp_path / 'hidden-examples.riima'
    review.save_project(prepared, path)
    prepared = review.load_project(path)
    assert prepared['review'].items[0]['annotations'][0].flat[195] == -1
    rgb = viewer.render_workspace(prepared, 'Máscara selecionada', **kwargs)
    np.testing.assert_array_equal(rgb.reshape(-1, 3)[195], [0, 0, 0])
    shown = viewer.render_workspace(prepared, 'Máscara selecionada', show_applied=True, **kwargs)
    np.testing.assert_array_equal(shown.reshape(-1, 3)[195], [255, 70, 70])
    prepared['review'].annotate(0, 0, [195], 1)
    rgb = viewer.render_workspace(prepared, 'Máscara selecionada', **kwargs)
    np.testing.assert_array_equal(rgb.reshape(-1, 3)[195], [0, 255, 100])


def test_incremental_brush_pixels_match_full_view_in_every_display(tmp_path):
    from test_riima_review import prepared_dataset
    prepared = prepared_dataset(tmp_path)
    state = prepared['review']
    state.annotate(0, 0, [195], -1)
    state.refine_pending()
    state.annotate(0, 0, [195], 0)
    state.annotate(0, 0, [212], 1)
    state.exclude(0, [196], True)
    indices = np.array([195, 196, 212, 213])
    for display in ('Imagem', 'Sobreposição', 'Classificação', 'Máscara selecionada'):
        for show_applied in (False, True):
            kwargs = dict(selected_groups=[0], annotation_group=0, opacity=.37, show_applied=show_applied)
            full = viewer.render_workspace(prepared, display, **kwargs)
            partial = viewer.workspace_pixels(prepared, display, indices=indices, **kwargs)
            np.testing.assert_array_equal(full.reshape(-1, 3)[indices], partial)


def test_review_overlay_selected_mask_annotations_and_initial_unchanged():
    rgb = np.full((3, 4, 3), 100, np.uint8)
    review = dict(labels=np.array([[0, 0, 1, -1]] * 3), valid=np.ones((3, 4), bool),
                  locked=np.full((3, 4), -1), palette=np.array([[200, 0, 0], [0, 200, 0]], np.uint8),
                  annotations={0: np.array([[1, 0, -1, 0], [0, 0, 0, 0], [0, 0, 0, 0]], np.int8)})
    result = viewer.render_review_overlay(rgb, review, 0, 0.5)
    np.testing.assert_array_equal(result[1, 0], [150, 50, 50])
    np.testing.assert_array_equal(result[1, 2], [100, 100, 100])
    np.testing.assert_array_equal(result[0, 0], [0, 255, 100])
    np.testing.assert_array_equal(result[0, 2], [255, 70, 70])
    np.testing.assert_array_equal(rgb, np.full_like(rgb, 100))


def test_brush_indices_follow_segment_and_clip_to_image():
    from riima import viewer
    indices = viewer.brush_indices((4, 6), (-2, 2), (4, 2), 1)
    assert set(range(12, 17)) <= set(indices)
    assert np.all(indices >= 0)
    assert np.all(indices < 24)


def test_interpreted_singleton_legend_keeps_raw_cluster_identity():
    from riima import core
    from riima import viewer
    raw = dict(labels=np.array([[0, 1, 2]], dtype=np.uint8), valid=np.ones((1, 3), bool),
               confidence=np.ones((1, 3)), clusters=3, palette=core.cluster_palette(3))
    state = core.Interpretation(raw)
    state.merge_clusters([0, 1])
    rows = viewer.build_preview_legend('Clusters interpretados', {'optical': raw, 'interpretation': state})
    assert rows[1]['label'] == 'Cluster 2'
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def _session_fixture():
    optical_labels = np.array([
        [0, 0, 1, 1],
        [0, 0, 1, 1],
        [2, 2, 1, 1],
        [2, 2, 1, 1],
    ], dtype=np.uint8)
    valid = np.ones((4, 4), dtype=bool)
    ppl = np.full((4, 4, 3), 210, dtype=np.uint8)
    cpol = np.full((4, 4, 3), 90, dtype=np.uint8)
    cpol[:, 2:] = 140
    optical = {
        "ppl": ppl,
        "cpol": cpol,
        "labels": optical_labels,
        "confidence": np.array([
            [0.9, 0.8, 0.7, 0.6],
            [0.9, 0.8, 0.7, 0.6],
            [0.5, 0.4, 0.7, 0.6],
            [0.5, 0.4, 0.7, 0.6],
        ], dtype=np.float32),
        "valid": valid,
        "palette": np.array([[10,20,30],[40,50,60],[70,80,90]], dtype=np.uint8),
        "stats": [
            {"cluster": 0, "pixels": 4, "area": 25.0, "confidence": 0.85},
            {"cluster": 1, "pixels": 8, "area": 50.0, "confidence": 0.65},
            {"cluster": 2, "pixels": 4, "area": 25.0, "confidence": 0.45},
        ],
        "clusters": 3,
        "registration_method": "phase_translation",
        "registration_angle_deg": 0.0,
        "valid_area_percent": 100.0,
    }
    from riima.core import Interpretation
    optical["reference"] = optical.pop("ppl")
    optical["images"] = [optical["reference"], optical.pop("cpol")]
    interpretation = Interpretation(optical)
    interpretation.create_family([0, 2], "Feldspato")
    interpretation.create_family([1], "Matriz")
    return {"optical": optical, "interpretation": interpretation, "output": "/tmp/out"}


def test_view_modes_use_plain_language_and_include_core_products():
    from riima.viewer import VIEW_MODES
    # Registered source images are layers (see preview_sources/compose_layers), not
    # fixed "PPL"/"cPol"/"PPL | cPol" modes: those are superseded by the layer manager
    # and the two-source comparator.
    assert VIEW_MODES == (
        "Clusters brutos", "Clusters interpretados", "Famílias", "Confiança óptica", "Máscara",
        "Contornos de clusters", "Contornos de famílias",
    )


def test_render_preview_mode_returns_same_canvas_shape_for_all_modes():
    from riima.viewer import VIEW_MODES, render_preview_mode
    prepared = _session_fixture()
    for mode in VIEW_MODES:
        rgb = render_preview_mode(mode, prepared)
        assert rgb.shape == prepared["optical"]["reference"].shape, mode
        assert rgb.dtype == np.uint8, mode


def test_render_preview_mode_uses_reference_image_when_ppl_cpol_are_absent():
    from riima.viewer import render_preview_mode
    prepared = _fake_prepared_with_registered_images(["A", "B", "C"])
    for mode in ("Clusters brutos", "Clusters interpretados", "Confiança óptica", "Famílias", "Máscara"):
        rgb = render_preview_mode(mode, prepared)
        assert rgb.shape == prepared["optical"]["images"][0].shape, mode
        assert rgb.dtype == np.uint8, mode


def test_optical_and_family_legends_are_human_readable():
    from riima.viewer import build_preview_legend
    prepared = _session_fixture()
    optical_rows = build_preview_legend("Clusters interpretados", prepared)
    assert optical_rows[0]["label"] == "Cluster 0"
    assert "25,0%" in optical_rows[0]["detail"]
    assert "confiança" in optical_rows[0]["detail"].lower()

    family_rows = build_preview_legend("Famílias", prepared)
    assert family_rows[0]["label"] == "Feldspato"
    assert "clusters 0+2" in family_rows[0]["detail"]
    assert "50,0%" in family_rows[0]["detail"]


def test_confidence_label_is_coarse_and_interpretable():
    from riima.viewer import confidence_label
    assert confidence_label(0.2) == "baixa"
    assert confidence_label(0.55) == "média"
    assert confidence_label(0.85) == "alta"


def test_preview_title_always_names_the_current_product():
    from riima.viewer import preview_title
    prepared = _session_fixture()
    assert preview_title("Clusters brutos", prepared) == "Clusters brutos: 3 grupos"
    assert preview_title("Clusters interpretados", prepared) == "Clusters interpretados: 3 grupos"
    assert preview_title("Famílias", prepared) == "Famílias: 2 famílias"


def test_preview_fit_size_preserves_aspect_ratio_and_viewport():
    from riima.viewer import fit_preview_size
    w, h = fit_preview_size((1476, 2048), (800, 600))
    assert w <= 800 and h <= 600
    assert abs((w / h) - (2048 / 1476)) < 0.01
    # Same source geometry must get the same rendered size regardless of view mode/legend.
    assert (w, h) == fit_preview_size((1476, 2048), (800, 600))


def test_merge_selected_groups_is_reversible_data_model():
    from riima.core import identity_groups, merge_selected_groups
    groups = identity_groups(4)
    merged = merge_selected_groups(groups, [0, 2])
    assert merged == [{0, 2}, {1}, {3}]
    # helper must not mutate the original state, so GUI undo can simply restore it
    assert groups == [{0}, {1}, {2}, {3}]


def test_optical_interpretation_merges_raw_clusters_without_touching_raw_labels():
    from riima.core import apply_optical_interpretation
    prepared = _session_fixture()
    optical = prepared["optical"]
    raw = optical["labels"].copy()
    interpreted = apply_optical_interpretation(optical, [{0, 2}, {1}])
    assert interpreted["clusters"] == 2
    assert set(np.unique(interpreted["labels"])) == {0, 1}
    # raw 0 and 2 become one interpreted class; raw 1 remains separate
    assert np.all(interpreted["labels"][raw == 0] == 0)
    assert np.all(interpreted["labels"][raw == 2] == 0)
    assert np.all(interpreted["labels"][raw == 1] == 1)
    assert np.array_equal(optical["labels"], raw)
    assert interpreted["groups"] == [{0, 2}, {1}]


def test_fit_width_and_100_percent_scales_are_predictable():
    from riima import viewer as ge
    image_shape = (1000, 2000)
    viewport = (1000, 800)
    assert ge.fit_scale(image_shape, viewport) == pytest.approx(0.5)
    assert ge.width_scale(image_shape, viewport) == pytest.approx(0.5)
    assert ge.ViewportState.for_100_percent(image_shape).zoom == pytest.approx(1.0)


def test_switching_layers_does_not_change_viewport_state():
    from riima import viewer as ge
    prepared = _session_fixture()
    state = ge.ViewportState(zoom=2.5, center_x=640, center_y=430, mode="manual")
    before = dataclasses.asdict(state)
    _ = ge.render_preview_mode("Clusters interpretados", prepared)
    assert dataclasses.asdict(state) == before


def test_zoom_about_cursor_preserves_image_point_under_cursor():
    from riima import viewer as ge
    state = ge.ViewportState.fit((1000, 2000), (1000, 800))
    p_before = ge.canvas_to_image((700, 400), state, (1000, 2000), (1000, 800))
    state2 = ge.zoom_about_point(state, 1.5, (700, 400), (1000, 2000), (1000, 800))
    p_after = ge.canvas_to_image((700, 400), state2, (1000, 2000), (1000, 800))
    assert p_after == pytest.approx(p_before, abs=1e-6)


def test_pan_view_moves_center_opposite_the_drag_direction_after_reveal():
    from riima import viewer as ge
    state = ge.ViewportState(zoom=1.0, center_x=1000, center_y=500, mode="manual")
    image_shape = (1000, 2000)
    viewport = (400, 300)
    # Dragging the mouse right by 40px should reveal content to the left: center moves left.
    moved = ge.pan_view(state, 40, 0, image_shape, viewport)
    assert moved.center_x < state.center_x
    assert moved.center_y == pytest.approx(state.center_y)


def test_viewport_state_survives_mode_switch_without_recompute():
    from riima import viewer as ge
    state = ge.ViewportState(zoom=3.0, center_x=10, center_y=20, mode="manual")
    # A no-op geometry query must not mutate or reset the passed-in state.
    _ = ge.viewport_visible_box(state, (200, 300), (150, 150))
    assert state.zoom == 3.0 and state.center_x == 10 and state.center_y == 20


def _fake_prepared_with_registered_images(names):
    n = len(names)
    h, w = 6, 8
    images = []
    for i in range(n):
        img = np.full((h, w, 3), 40 * (i + 1), dtype=np.uint8)
        images.append(img)
    labels = np.zeros((h, w), dtype=np.uint8)
    labels[:, w // 2:] = 1
    valid = np.ones((h, w), dtype=bool)
    optical = {
        "images": images,
        "registered_images": [{"name": name} for name in names],
        "reference": images[0],
        "labels": labels,
        "confidence": np.full((h, w), 0.6, dtype=np.float32),
        "valid": valid,
        "palette": np.array([[10, 20, 30], [40, 50, 60]], dtype=np.uint8),
        "stats": [
            {"cluster": 0, "pixels": 24, "area": 50.0, "confidence": 0.6, "source_clusters": (0,)},
            {"cluster": 1, "pixels": 24, "area": 50.0, "confidence": 0.6, "source_clusters": (1,)},
        ],
        "clusters": 2,
    }
    from riima.core import Interpretation
    return {"optical": optical, "interpretation": Interpretation(optical), "output": "/tmp/out"}


def test_preview_sources_include_every_registered_image_and_core_products():
    from riima.viewer import preview_sources
    prepared = _fake_prepared_with_registered_images(["A", "B", "C"])
    entries = preview_sources(prepared)
    ids = [x["id"] for x in entries]
    assert ids[:3] == ["image:0", "image:1", "image:2"]
    assert {"raw_clusters", "clusters", "optical_confidence", "families", "mask", "cluster_boundaries", "family_boundaries"} <= set(ids)
    for entry in entries:
        rgb = entry["render"]()
        assert rgb.shape == prepared["optical"]["images"][0].shape
        assert rgb.dtype == np.uint8


def test_compose_layers_blends_by_opacity_and_skips_hidden_layers():
    from riima.viewer import compose_layers
    entries = [
        {"id": "a", "render": lambda: np.full((4, 4, 3), 200, np.uint8)},
        {"id": "b", "render": lambda: np.full((4, 4, 3), 0, np.uint8)},
    ]
    order = ["a", "b"]

    only_a = compose_layers(entries, order, {"a": {"visible": True, "opacity": 1.0}, "b": {"visible": False, "opacity": 1.0}})
    assert np.array_equal(only_a, np.full((4, 4, 3), 200, np.uint8))

    half_b_over_a = compose_layers(entries, order, {"a": {"visible": True, "opacity": 1.0}, "b": {"visible": True, "opacity": 0.5}})
    assert np.array_equal(half_b_over_a, np.full((4, 4, 3), 100, np.uint8))

    zero_opacity_b_is_skipped = compose_layers(entries, order, {"a": {"visible": True, "opacity": 1.0}, "b": {"visible": True, "opacity": 0.0}})
    assert np.array_equal(zero_opacity_b_is_skipped, np.full((4, 4, 3), 200, np.uint8))


def test_compose_layers_returns_neutral_canvas_when_nothing_visible():
    from riima.viewer import compose_layers
    entries = [{"id": "a", "render": lambda: np.full((3, 3, 3), 200, np.uint8)}]
    out = compose_layers(entries, ["a"], {"a": {"visible": False, "opacity": 1.0}})
    assert out.shape == (3, 3, 3)
    assert np.all(out == 30)


def test_compose_split_obeys_fraction_without_resizing():
    from riima.viewer import compose_split
    left = np.zeros((20, 100, 3), np.uint8)
    right = np.full((20, 100, 3), 255, np.uint8)
    out = compose_split(left, right, 0.3)
    assert out.shape == left.shape
    assert np.all(out[:, :30] == 0)
    assert np.all(out[:, 30:] == 255)


def test_reference_index_is_rebased_after_inactive_images_are_filtered():
    from riima.core import normalize_image_selection
    paths = ["a", "b", "c"]
    active = [False, True, True]
    selected, ref = normalize_image_selection(paths, active, 2)
    assert selected == ["b", "c"]
    assert ref == 1


def test_inactive_reference_is_rejected():
    from riima.core import normalize_image_selection
    with pytest.raises(ValueError, match="referência"):
        normalize_image_selection(["a", "b"], [True, False], 1)


def test_active_reference_index_counts_active_images_before_it():
    from riima.core import active_reference_index
    assert active_reference_index([True, False, True, True], 3) == 2
    assert active_reference_index([True, True, True], 0) == 0


def test_normalize_image_selection_requires_at_least_one_active_image():
    from riima.core import normalize_image_selection
    with pytest.raises(ValueError, match="[Aa]tiv"):
        normalize_image_selection(["a", "b"], [False, False], 0)


def test_registration_quality_label_flags_weak_alignment():
    from riima.core import registration_quality_label
    # Real values observed on the two validation slides: both are accepted as OK.
    assert registration_quality_label(0.3235, 0.4772) == "registro OK"
    assert registration_quality_label(0.0665, 0.2067) == "registro OK"
    assert registration_quality_label(0.10, 0.11) == "registro fraco"
    assert registration_quality_label(0.05, 0.05) == "registro fraco"


def test_legends_name_consolidated_optical_and_family_sources():
    from riima.viewer import build_preview_legend
    prepared = _session_fixture()
    interpretation = prepared["interpretation"]
    interpretation.merge_clusters([0, 2])
    interpretation.rename_cluster(0, "Claro")
    optical_rows = build_preview_legend("Clusters interpretados", prepared)
    assert "0+2" in optical_rows[0]["label"]
    assert "Claro" in optical_rows[0]["label"]
    raw_rows = build_preview_legend("Clusters brutos", prepared)
    assert raw_rows[0]["label"] == "Cluster 0"
    family_rows = build_preview_legend("Famílias", prepared)
    assert family_rows[0]["label"] == "Feldspato"
    assert "0+2" in family_rows[0]["detail"]


def test_blend_overlay_uint8_matches_reference_and_preserves_invalid():
    """Relocated from the old petro_similarity core test suite: this compositing helper
    moved to riima_viewer since it is presentation, not scientific, logic."""
    from riima.viewer import blend_overlay_uint8
    rng = np.random.default_rng(99)
    base = rng.integers(0, 256, size=(17, 19, 3), dtype=np.uint8)
    color = rng.integers(0, 256, size=base.shape, dtype=np.uint8)
    valid = rng.random(base.shape[:2]) > 0.25
    alpha = 0.48
    got = blend_overlay_uint8(base, color, alpha, valid)
    ref = np.clip((1 - alpha) * base.astype(np.float32) + alpha * color.astype(np.float32), 0, 255).astype(np.uint8)
    ref[~valid] = base[~valid]
    assert np.max(np.abs(got.astype(np.int16) - ref.astype(np.int16))) <= 1
    assert np.array_equal(got[~valid], base[~valid])


def test_split_divider_geometry_spans_only_the_letterboxed_image_height():
    """The draggable divider spans only the actual displayed
    image height, not the full canvas, when the image is letterboxed inside the viewport."""
    from riima.viewer import ViewportState, split_divider_geometry
    image_shape = (100, 200)  # much shorter than the viewport -> vertical letterboxing
    viewport_size = (200, 400)
    state = ViewportState.fit(image_shape, viewport_size)
    geo = split_divider_geometry(state, image_shape, viewport_size, split_fraction=0.5)
    assert geo["y_top"] == pytest.approx(150.0)
    assert geo["y_bottom"] == pytest.approx(250.0)
    # Must NOT span the full canvas height (the historical bug being fixed here).
    assert geo["y_bottom"] - geo["y_top"] < viewport_size[1]
    assert geo["canvas_x"] == pytest.approx(100.0)
    assert geo["handle_center"] == pytest.approx((100.0, 200.0))


def test_split_divider_hit_box_is_wider_than_the_line_and_hit_tests_correctly():
    from riima.viewer import ViewportState, split_divider_geometry, point_hits_divider
    image_shape = (100, 200)
    viewport_size = (200, 400)
    state = ViewportState.fit(image_shape, viewport_size)
    geo = split_divider_geometry(state, image_shape, viewport_size, split_fraction=0.5, hit_margin=8.0)
    x0, _y0, x1, _y1 = geo["hit_box"]
    assert x1 - x0 == pytest.approx(16.0)  # wider than a 1-2px line, easy to grab with a mouse
    assert point_hits_divider(100, 200, geo)  # dead center of the handle
    assert point_hits_divider(93, 200, geo)  # inside the wide hitbox, off the exact line
    assert not point_hits_divider(80, 200, geo)  # far outside the hitbox
    assert not point_hits_divider(100, 300, geo)  # below the letterboxed image, inside the canvas


def test_render_compare_view_composites_only_the_visible_crop():
    """Perf fix: dragging the divider must not recompose the full-resolution image on
    every mouse-motion event, only the currently visible viewport-sized crop."""
    from riima.viewer import ViewportState, render_compare_view
    left = np.zeros((100, 200, 3), np.uint8)
    right = np.full((100, 200, 3), 255, np.uint8)
    image_shape = (100, 200)
    viewport_size = (50, 50)  # much smaller than the full image
    state = ViewportState(zoom=1.0, center_x=100, center_y=50, mode="manual")
    composed, plan = render_compare_view(left, right, 0.5, state, image_shape, viewport_size)
    x0, y0, x1, y1 = plan["src_box"]
    assert composed.shape[:2] == (y1 - y0, x1 - x0)
    assert composed.shape[0] < image_shape[0] and composed.shape[1] < image_shape[1]
    assert np.all(composed[:, :25] == 0) and np.all(composed[:, 25:] == 255)


def test_raw_and_interpreted_renders_remain_distinct_after_manual_edits():
    from riima.viewer import render_preview_mode
    prepared = _session_fixture()
    raw = render_preview_mode("Clusters brutos", prepared)
    interpretation = prepared["interpretation"]
    interpretation.merge_clusters([0, 2])
    interpretation.recolor_cluster(0, (255, 0, 0))
    interpreted = render_preview_mode("Clusters interpretados", prepared)
    assert np.array_equal(raw, render_preview_mode("Clusters brutos", prepared))
    assert not np.array_equal(raw, interpreted)
    assert np.all(interpreted[prepared["optical"]["labels"] == 2] == (255, 0, 0))


def test_cluster_boundaries_include_zero_label_and_exclude_invalid_pixels():
    from riima.viewer import render_preview_mode
    prepared = _fake_prepared_with_registered_images(["A"])
    prepared["optical"]["valid"][0] = False
    rgb = render_preview_mode("Contornos de clusters", prepared)
    assert np.all(rgb[1:, 3:5] == (0, 255, 255))
    assert np.array_equal(rgb[0], prepared["optical"]["reference"][0])


def test_family_map_uses_manual_color_and_leaves_unassigned_neutral():
    from riima.viewer import render_preview_mode
    prepared = _fake_prepared_with_registered_images(["A"])
    prepared["interpretation"].create_family([1], "Teste", (12, 34, 56))
    rgb = render_preview_mode("Famílias", prepared)
    assert np.all(rgb[:, :4] == 30)
    assert np.all(rgb[:, 4:] == (12, 34, 56))
