#!/usr/bin/env python3
"""RIIMA viewer: presentation, navigation and compositing, no scientific computation.

Everything here consumes raw clusters and manual interpretations from ``riima.core``
and turns them into RGB arrays, legends, titles and viewport geometry. Nothing in
this module recomputes registration or clustering.

Contents:
- ``ViewportState`` and its pure geometry (fit/width/100%, zoom-at-cursor, pan, the
  canvas<->image coordinate transform);
- ``preview_sources``/``compose_layers``: the layer-manager model (any registered image
  or derived product, shown with independent visibility/opacity);
- ``compose_split``/``render_compare_view``: the two-source comparator, including a
  viewport-only compositing path so dragging the divider never recomposes the full image;
  ``split_divider_geometry`` locates the draggable handle/hitbox against the image's
  actual displayed bounds (so it never spans the letterboxed canvas margins);
- ``render_preview_mode``/``build_preview_legend``/``preview_title``/
  ``human_analysis_summary``: mode-to-RGB/text presentation logic.

This module has no ``tkinter`` import; ``riima.gui`` is the only place Tk widgets and
event bindings live.
"""
from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np

import cv2

from .core import boundary_mask
from .data_colors import (ERASE_RGB, EXCLUDED_RGB, LEGACY_CONTOUR_RGB, NEGATIVE_RGB,
                          POSITIVE_RGB, UNASSIGNED_RGB, WORKSPACE_CONTOUR_RGB)



def blend_overlay_uint8(base: np.ndarray, color: np.ndarray, alpha: float, valid: np.ndarray | None = None) -> np.ndarray:
    """Blend uint8 RGB arrays without multiple full-sized float copies.

    Used by the GUI, notebook figures and manually exported comparisons. Standard
    scientific outputs retain their own pixels without this display blend.
    """
    out = cv2.addWeighted(base, 1.0 - float(alpha), color, float(alpha), 0.0)
    if valid is not None:
        out[~valid] = base[~valid]
    return out


def render_review_overlay(base: np.ndarray, review: dict, cluster: int, opacity: float = 0.5) -> np.ndarray:
    """Selected-cluster overlay with explicit examples on the original RGB image."""
    out = base.copy()
    selected = review['valid'] & (review['labels'] == cluster)
    alpha = float(np.clip(opacity, 0, 1))
    out[selected] = np.clip(base[selected].astype(np.float32) * (1 - alpha) +
                            review['palette'][cluster].astype(np.float32) * alpha, 0, 255).astype(np.uint8)
    marks = review['annotations'].get(cluster)
    if marks is not None:
        out[review['valid'] & (marks == 1)] = POSITIVE_RGB
        out[review['valid'] & (marks == -1)] = NEGATIVE_RGB
    return out


def workspace_data(prepared: dict) -> dict:
    """One current classification on the selected image, with stable group identities."""
    state = prepared.get('review')
    if state is None:
        current = _interpreted(prepared)
        return dict(base=prepared['optical']['reference'], labels=current['labels'],
                    valid=current['valid'], palette=current['palette'],
                    groups={i: set(g) for i, g in enumerate(prepared['interpretation'].groups)},
                    completed=set(), dirty=set(), annotations={})
    item = state.items[state.selected_image if state.mode == 'independent' else 0]
    base = item['rgb'] if state.mode == 'independent' else item['optical']['images'][state.selected_image]
    return dict(base=base, **state.view(state.selected_image), dirty=state.dirty)


def workspace_mask(prepared: dict, selected_groups=None) -> np.ndarray:
    current = workspace_data(prepared)
    mask = current['valid'] & (current['labels'] >= 0)
    if selected_groups is not None:
        mask &= np.isin(current['labels'], list(selected_groups))
    return mask


def workspace_group_label(prepared: dict, group: int) -> str:
    state = prepared.get('review')
    if state is not None:
        return state.group_name(group, prepared.get('interpretation'))
    members = workspace_data(prepared)['groups'][group]
    label = f'Grupo {group}'
    for i, existing in enumerate(prepared['interpretation'].groups):
        if existing == members:
            return prepared['interpretation'].names.get(i) or label
    return label


def render_workspace(prepared: dict, display: str, *, selected_groups=None,
                     opacity: float = .5, contours: bool = False,
                     annotation_group: int | None = None, show_applied: bool = False) -> np.ndarray:
    """Image, classification, selection mask or current valid analysis area."""
    current = workspace_data(prepared)
    labels = current['labels']
    assigned = current['valid'] & (labels >= 0)
    selected = workspace_mask(prepared, selected_groups)
    out = workspace_pixels(prepared, display, selected_groups=selected_groups, opacity=opacity,
                           annotation_group=annotation_group, show_applied=show_applied).reshape(current['base'].shape)
    if contours and display != 'Máscara':
        edges = boundary_mask(labels, valid=assigned)
        out[edges & selected] = WORKSPACE_CONTOUR_RGB
    return out


def workspace_pixels(prepared: dict, display: str, *, indices=None, selected_groups=None,
                     opacity=.5, annotation_group=None, show_applied=False) -> np.ndarray:
    """Render only touched pixels while drawing, using the same rules as a full preview."""
    current = workspace_data(prepared)
    def pixels(array):
        flat = array.reshape(-1, 3) if array.ndim == 3 else array.ravel()
        return flat if indices is None else flat[indices]
    base, labels, valid = pixels(current['base']), pixels(current['labels']), pixels(current['valid'])
    if display == 'Máscara':
        return np.repeat((valid.astype(np.uint8) * 255)[:, None], 3, axis=1)
    assigned = valid & (labels >= 0)
    selected = assigned.copy()
    if selected_groups is not None:
        selected &= np.isin(labels, list(selected_groups))
    if display in ('Imagem', 'Original'):
        out = base.copy()
    elif display == 'Máscara selecionada':
        out = np.repeat((selected.astype(np.uint8) * 255)[:, None], 3, axis=1)
    elif display in ('Classificação', 'Classificação atual', 'Classificação inicial'):
        out = np.full_like(base, UNASSIGNED_RGB)
        out[assigned] = current['palette'][labels[assigned]]
    elif display in ('Sobreposição', 'Grupo selecionado'):
        out = base.copy()
        alpha = float(np.clip(opacity, 0, 1))
        out[selected] = np.clip(base[selected].astype(np.float32) * (1 - alpha) +
                                current['palette'][labels[selected]].astype(np.float32) * alpha, 0, 255).astype(np.uint8)
    else:
        raise ValueError(f'Visualização desconhecida: {display}')
    if annotation_group is not None:
        analysis_valid = pixels(current.get('analysis_valid', current['valid']))
        marks = current['annotations'].get(annotation_group)
        if marks is not None:
            marks = pixels(marks)
            applied = current.get('applied_annotations', {}).get(annotation_group)
            applied = pixels(applied) if applied is not None else 0
            visible = analysis_valid & ((marks != 0) if show_applied else (marks != applied))
            out[visible & (marks == 1)] = POSITIVE_RGB
            out[visible & (marks == -1)] = NEGATIVE_RGB
            out[visible & (marks == 0)] = ERASE_RGB
        if 'excluded' in current:
            excluded, applied = pixels(current['excluded']), pixels(current['applied_excluded'])
            visible = analysis_valid & (excluded if show_applied else excluded != applied)
            out[visible & excluded] = EXCLUDED_RGB
            out[visible & ~excluded] = ERASE_RGB
    return out


def workspace_sources(prepared: dict, selected_groups=None) -> list[dict]:
    """A fixed comparator menu that always refers to the current workspace."""
    return [dict(id='original', label='Imagem original',
                 render=lambda: render_workspace(prepared, 'Imagem')),
            dict(id='classification', label='Classificação atual',
                 render=lambda: render_workspace(prepared, 'Classificação')),
            dict(id='selection_mask', label='Máscara selecionada',
                 render=lambda: render_workspace(prepared, 'Máscara selecionada', selected_groups=selected_groups)),
            dict(id='analysis_mask', label='Máscara',
                 render=lambda: render_workspace(prepared, 'Máscara'))]


def workspace_legend(prepared: dict, selected_groups=None, *, mask=False) -> list[dict]:
    current = workspace_data(prepared)
    total = max(1, int(current['valid'].sum()))
    counts = np.bincount(current['labels'][current['valid'] & (current['labels'] >= 0)],
                         minlength=len(current['palette']))
    groups = sorted(current['groups']) if selected_groups is None else sorted(selected_groups)
    rows = []
    for c in groups:
        if c not in current['groups']:
            continue
        status = ('fundo' if c in current.get('background_groups', set()) else
                  'pronto' if c in current['completed'] else
                  'edições pendentes' if c in current['dirty'] else 'em revisão')
        rows.append(dict(label=workspace_group_label(prepared, c),
                         color=WORKSPACE_CONTOUR_RGB if mask else tuple(int(v) for v in current['palette'][c]),
                         detail=f'{int(counts[c])} pixels · {100 * counts[c] / total:.1f}% · {status}'.replace('.', ',')))
    return rows


def brush_indices(shape: tuple[int, int], start: tuple[float, float], end: tuple[float, float],
                  radius: int) -> np.ndarray:
    """Rasterize a continuous image-coordinate stroke, clipping before allocation."""
    import cv2
    h, w = shape
    radius = max(1, int(radius))
    x0 = max(0, int(np.floor(min(start[0], end[0]))) - radius)
    y0 = max(0, int(np.floor(min(start[1], end[1]))) - radius)
    x1 = min(w, int(np.ceil(max(start[0], end[0]))) + radius + 1)
    y1 = min(h, int(np.ceil(max(start[1], end[1]))) + radius + 1)
    if x0 >= x1 or y0 >= y1:
        return np.empty(0, np.int64)
    mask = np.zeros((y1 - y0, x1 - x0), np.uint8)
    a = (int(round(start[0])) - x0, int(round(start[1])) - y0)
    b = (int(round(end[0])) - x0, int(round(end[1])) - y0)
    cv2.line(mask, a, b, 1, thickness=radius * 2 + 1)
    rows, cols = np.nonzero(mask)
    return (rows.astype(np.int64) + y0) * w + cols + x0


def confidence_label(value: float) -> str:
    """Translate a numeric confidence into a coarse user-facing label."""
    value = float(value)
    if value < 0.40:
        return "baixa"
    if value < 0.70:
        return "média"
    return "alta"


def fit_preview_size(image_shape: tuple[int, int], viewport_size: tuple[int, int]) -> tuple[int, int]:
    """Fit HxW image geometry into a fixed WxH viewport without changing aspect ratio."""
    image_h, image_w = (max(1, int(v)) for v in image_shape)
    view_w, view_h = (max(1, int(v)) for v in viewport_size)
    scale = min(view_w / image_w, view_h / image_h, 1.0)
    return max(1, int(round(image_w * scale))), max(1, int(round(image_h * scale)))


# ---------- persistent viewport model (zoom/pan/fit/width/100%) ----------
# Pure geometry only: no scientific recomputation may ever be triggered from here.

MIN_ZOOM = 0.02
MAX_ZOOM = 40.0


def fit_scale(image_shape: tuple[int, int], viewport_size: tuple[int, int]) -> float:
    """Scale that fits the whole image inside the viewport without upscaling past 100%."""
    image_h, image_w = (max(1, int(v)) for v in image_shape)
    view_w, view_h = (max(1, int(v)) for v in viewport_size)
    return min(view_w / image_w, view_h / image_h, 1.0)


def width_scale(image_shape: tuple[int, int], viewport_size: tuple[int, int]) -> float:
    """Scale that fits the image width to the viewport width (may exceed 100%)."""
    _image_h, image_w = (max(1, int(v)) for v in image_shape)
    view_w, _view_h = (max(1, int(v)) for v in viewport_size)
    return view_w / max(1, image_w)


@dataclass
class ViewportState:
    """Persistent image-space zoom/pan shared by every visual mode.

    Switching between source images, clusters, families, confidence, etc. must
    never change this state; only an explicit zoom/pan/fit/width/100% action does.
    """
    zoom: float
    center_x: float
    center_y: float
    mode: str = "manual"  # "fit", "width", "100%", or "manual" after a user zoom/pan

    @classmethod
    def fit(cls, image_shape: tuple[int, int], viewport_size: tuple[int, int]) -> "ViewportState":
        image_h, image_w = image_shape
        return cls(zoom=fit_scale(image_shape, viewport_size), center_x=image_w / 2.0, center_y=image_h / 2.0, mode="fit")

    @classmethod
    def width(cls, image_shape: tuple[int, int], viewport_size: tuple[int, int]) -> "ViewportState":
        image_h, image_w = image_shape
        return cls(zoom=width_scale(image_shape, viewport_size), center_x=image_w / 2.0, center_y=image_h / 2.0, mode="width")

    @classmethod
    def for_100_percent(cls, image_shape: tuple[int, int]) -> "ViewportState":
        image_h, image_w = image_shape
        return cls(zoom=1.0, center_x=image_w / 2.0, center_y=image_h / 2.0, mode="100%")


def canvas_to_image(
    canvas_xy: tuple[float, float], state: ViewportState, image_shape: tuple[int, int], viewport_size: tuple[int, int]
) -> tuple[float, float]:
    view_w, view_h = viewport_size
    cx, cy = canvas_xy
    img_x = state.center_x + (cx - view_w / 2.0) / state.zoom
    img_y = state.center_y + (cy - view_h / 2.0) / state.zoom
    return img_x, img_y


def image_to_canvas(
    image_xy: tuple[float, float], state: ViewportState, image_shape: tuple[int, int], viewport_size: tuple[int, int]
) -> tuple[float, float]:
    view_w, view_h = viewport_size
    ix, iy = image_xy
    cx = view_w / 2.0 + (ix - state.center_x) * state.zoom
    cy = view_h / 2.0 + (iy - state.center_y) * state.zoom
    return cx, cy


def _clamp_center(center: float, extent: float, viewport_extent: float, zoom: float) -> float:
    """Clamp so the image can be panned far but never lost completely off-screen."""
    visible_extent = viewport_extent / max(zoom, 1e-6)
    margin = visible_extent * 0.5
    lo, hi = -margin, extent + margin
    if hi < lo:
        lo, hi = hi, lo
    return float(np.clip(center, lo, hi))


def clamp_viewport(state: ViewportState, image_shape: tuple[int, int], viewport_size: tuple[int, int]) -> ViewportState:
    image_h, image_w = image_shape
    view_w, view_h = viewport_size
    zoom = float(np.clip(state.zoom, MIN_ZOOM, MAX_ZOOM))
    center_x = _clamp_center(state.center_x, image_w, view_w, zoom)
    center_y = _clamp_center(state.center_y, image_h, view_h, zoom)
    return replace(state, zoom=zoom, center_x=center_x, center_y=center_y)


def zoom_about_point(
    state: ViewportState, factor: float, cursor_xy: tuple[float, float],
    image_shape: tuple[int, int], viewport_size: tuple[int, int],
) -> ViewportState:
    """Zoom by ``factor`` while keeping the image point under the cursor fixed on screen."""
    new_zoom = float(np.clip(state.zoom * factor, MIN_ZOOM, MAX_ZOOM))
    px, py = canvas_to_image(cursor_xy, state, image_shape, viewport_size)
    view_w, view_h = viewport_size
    cx, cy = cursor_xy
    new_center_x = px - (cx - view_w / 2.0) / new_zoom
    new_center_y = py - (cy - view_h / 2.0) / new_zoom
    new_state = ViewportState(zoom=new_zoom, center_x=new_center_x, center_y=new_center_y, mode="manual")
    return clamp_viewport(new_state, image_shape, viewport_size)


def pan_view(
    state: ViewportState, dx_canvas: float, dy_canvas: float,
    image_shape: tuple[int, int], viewport_size: tuple[int, int],
) -> ViewportState:
    """Pan by a canvas-pixel drag delta; the image moves together with the cursor."""
    new_state = replace(
        state,
        center_x=state.center_x - dx_canvas / state.zoom,
        center_y=state.center_y - dy_canvas / state.zoom,
        mode="manual",
    )
    return clamp_viewport(new_state, image_shape, viewport_size)


def viewport_visible_box(
    state: ViewportState, image_shape: tuple[int, int], viewport_size: tuple[int, int]
) -> tuple[float, float, float, float]:
    """The visible image-space rectangle (x0, y0, x1, y1), clipped to the image bounds."""
    image_h, image_w = image_shape
    view_w, view_h = viewport_size
    half_w = (view_w / 2.0) / state.zoom
    half_h = (view_h / 2.0) / state.zoom
    x0 = max(0.0, state.center_x - half_w)
    y0 = max(0.0, state.center_y - half_h)
    x1 = min(float(image_w), state.center_x + half_w)
    y1 = min(float(image_h), state.center_y + half_h)
    return x0, y0, x1, y1


def viewport_render_plan(
    state: ViewportState, image_shape: tuple[int, int], viewport_size: tuple[int, int]
) -> dict:
    """Geometry a renderer needs: which image region to crop/scale and where to place it."""
    x0, y0, x1, y1 = viewport_visible_box(state, image_shape, viewport_size)
    view_w, view_h = viewport_size
    src_box = (int(np.floor(x0)), int(np.floor(y0)), int(np.ceil(x1)), int(np.ceil(y1)))
    src_w = max(1, src_box[2] - src_box[0])
    src_h = max(1, src_box[3] - src_box[1])
    dst_w = max(1, int(round(src_w * state.zoom)))
    dst_h = max(1, int(round(src_h * state.zoom)))
    offset_x = (view_w / 2.0) + (src_box[0] - state.center_x) * state.zoom
    offset_y = (view_h / 2.0) + (src_box[1] - state.center_y) * state.zoom
    return {
        "src_box": src_box,
        "dst_size": (dst_w, dst_h),
        "canvas_offset": (offset_x, offset_y),
        "zoom": state.zoom,
    }


# ---------- draggable split comparator: viewport-only compositing + handle geometry ----------

def render_compare_view(
    left_rgb: np.ndarray, right_rgb: np.ndarray, split_fraction: float,
    state: ViewportState, image_shape: tuple[int, int], viewport_size: tuple[int, int],
) -> tuple[np.ndarray, dict]:
    """Crop both (already-rendered, cached) sources to the current viewport before compositing.

    This is the fix for slow divider dragging: composing and scaling only the visible
    region keeps per-frame cost proportional to the viewport size rather than the full
    image resolution, so a drag on a multi-thousand-pixel slide stays smooth. Callers
    should render/cache ``left_rgb``/``right_rgb`` once per source *selection* (not once
    per drag event) and pass the same cached arrays on every call here.

    Returns ``(composed_crop, plan)`` where ``plan`` is a ``viewport_render_plan`` dict
    (crop is already at ``plan["src_box"]``'s size; the caller resizes it to
    ``plan["dst_size"]`` and places it at ``plan["canvas_offset"]``, exactly as for any
    other viewport render).
    """
    if left_rgb.shape != right_rgb.shape:
        raise ValueError("render_compare_view requires two images of the same shape")
    plan = viewport_render_plan(state, image_shape, viewport_size)
    x0, y0, x1, y1 = plan["src_box"]
    x0, y0 = max(0, x0), max(0, y0)
    x1, y1 = min(image_shape[1], x1), min(image_shape[0], y1)
    left_crop = left_rgb[y0:y1, x0:x1]
    right_crop = right_rgb[y0:y1, x0:x1]
    crop_w = max(1, x1 - x0)
    image_w = image_shape[1]
    local_fraction = float(np.clip((split_fraction * image_w - x0) / crop_w, 0.0, 1.0))
    composed = compose_split(left_crop, right_crop, local_fraction)
    return composed, plan


def split_divider_geometry(
    state: ViewportState, image_shape: tuple[int, int], viewport_size: tuple[int, int],
    split_fraction: float, *, hit_margin: float = 8.0,
) -> dict:
    """Where to draw/hit-test the comparator's draggable divider.

    The divider spans only the image's *actual displayed* height (accounting for
    letterboxing when the image is narrower/shorter than the viewport), not the full
    canvas.
    The returned hit box is wider than the 1-2px visible line so the divider is easy to
    grab with the mouse.
    """
    plan = viewport_render_plan(state, image_shape, viewport_size)
    offset_x, offset_y = plan["canvas_offset"]
    dst_w, dst_h = plan["dst_size"]
    image_w = image_shape[1]
    divider_image_x = float(np.clip(split_fraction, 0.0, 1.0)) * image_w
    canvas_x, _ = image_to_canvas((divider_image_x, 0.0), state, image_shape, viewport_size)
    y_top = offset_y
    y_bottom = offset_y + dst_h
    handle_center = (canvas_x, (y_top + y_bottom) / 2.0)
    return {
        "canvas_x": canvas_x,
        "y_top": y_top,
        "y_bottom": y_bottom,
        "handle_center": handle_center,
        "hit_box": (canvas_x - hit_margin, y_top, canvas_x + hit_margin, y_bottom),
        "image_bounds_x": (offset_x, offset_x + dst_w),
    }


def point_hits_divider(x: float, y: float, geometry: dict) -> bool:
    """Whether canvas point (x, y) falls within the divider's (wide) hit box."""
    x0, y0, x1, y1 = geometry["hit_box"]
    return x0 <= x <= x1 and y0 <= y <= y1


def compose_split(left_rgb: np.ndarray, right_rgb: np.ndarray, fraction: float) -> np.ndarray:
    """Vertical draggable-divider composition in image space: left up to ``fraction``, right after.

    No resizing happens here; both halves must already share the same viewport transform,
    which the caller applies before compositing.
    """
    if left_rgb.shape != right_rgb.shape:
        raise ValueError("compose_split requires two images of the same shape")
    width = left_rgb.shape[1]
    split_x = int(round(float(np.clip(fraction, 0.0, 1.0)) * width))
    out = left_rgb.copy()
    out[:, split_x:] = right_rgb[:, split_x:]
    return out


# ---------- layer manager: any source, independent visibility + opacity ----------

def compose_layers(entries: list[dict], layer_order: list[str], layer_states: dict) -> np.ndarray:
    """Alpha-composite visible layers bottom-to-top, each at its own opacity.

    ``entries`` is a ``preview_sources``-style list (``{"id", "render", ...}``).
    ``layer_order`` lists ids from bottom to top of the stack. ``layer_states[id]`` is
    ``{"visible": bool, "opacity": float in [0, 1]}``; ids missing a state, or with
    ``visible=False`` or ``opacity<=0``, are skipped entirely. When nothing is visible,
    returns a neutral dark canvas (the same convention used for masked/invalid pixels
    elsewhere) the same shape as any entry's render.
    """
    by_id = {e["id"]: e for e in entries}
    visible_ids = [
        lid for lid in layer_order
        if lid in by_id and layer_states.get(lid, {}).get("visible")
        and float(layer_states.get(lid, {}).get("opacity", 0.0)) > 0.0
    ]
    if not visible_ids:
        sample = entries[0]["render"]() if entries else np.zeros((1, 1, 3), dtype=np.uint8)
        return np.full_like(sample, UNASSIGNED_RGB)

    result = None
    for lid in visible_ids:
        rgb = by_id[lid]["render"]().astype(np.float32)
        opacity = float(np.clip(layer_states[lid].get("opacity", 1.0), 0.0, 1.0))
        if result is None:
            base = np.full_like(rgb, UNASSIGNED_RGB)
            result = base * (1.0 - opacity) + rgb * opacity
        else:
            result = result * (1.0 - opacity) + rgb * opacity
    return np.clip(result, 0, 255).astype(np.uint8)


_PREVIEW_PRODUCT_MODES = (
    ("raw_clusters", "Clusters brutos"),
    ("clusters", "Clusters interpretados"),
    ("families", "Famílias"),
    ("optical_confidence", "Confiança óptica"),
    ("mask", "Máscara"),
    ("cluster_boundaries", "Contornos de clusters"),
    ("family_boundaries", "Contornos de famílias"),
)
_PRODUCT_ID_TO_MODE = dict(_PREVIEW_PRODUCT_MODES)
VIEW_MODES = tuple(_PRODUCT_ID_TO_MODE.values())
_PRODUCT_ID_TO_MODE['reviewed_clusters'] = 'Clusters em revisão'


def _interpreted(prepared: dict) -> dict:
    interpretation = prepared.get("interpretation")
    return interpretation.interpreted() if interpretation is not None else prepared["optical"]


def _families(prepared: dict) -> dict | None:
    interpretation = prepared.get("interpretation")
    return interpretation.family_result() if interpretation is not None else None


def preview_sources(prepared: dict) -> list[dict]:
    """Registered images and products, lazily rendered on their common geometry."""
    optical = prepared["optical"]
    images = optical.get("images", [optical["reference"]])
    registered = optical.get("registered_images") or prepared.get("registered_images") or []
    sources = []
    for i, img in enumerate(images):
        name = registered[i].get("name", f"Imagem {i}") if i < len(registered) else f"Imagem {i}"
        sources.append({"id": f"image:{i}", "label": name, "render": lambda im=img: im})
    for pid, mode in _PREVIEW_PRODUCT_MODES:
        sources.append({"id": pid, "label": mode,
                        "render": lambda m=mode: render_preview_mode(m, prepared)})
    if 'review' in prepared:
        sources.append(dict(id='reviewed_clusters', label='Clusters em revisão',
                            render=lambda: render_preview_mode('Clusters em revisão', prepared)))
    return sources


def render_preview_mode(mode: str, prepared: dict) -> np.ndarray:
    """Render a raw or interpreted product without changing scientific state."""
    optical = prepared["optical"]
    valid = optical["valid"]
    reference = optical["reference"]
    if mode == 'Clusters em revisão':
        state = prepared['review']
        current = state.view(state.selected_image)
        rgb = np.full_like(reference, UNASSIGNED_RGB)
        assigned = current['valid'] & (current['labels'] >= 0)
        rgb[assigned] = current['palette'][current['labels'][assigned]]
        return rgb
    if mode in {"Clusters brutos", "Clusters interpretados"}:
        result = optical if mode == "Clusters brutos" else _interpreted(prepared)
        rgb = np.full_like(reference, UNASSIGNED_RGB)
        rgb[valid] = result["palette"][result["labels"][valid]]
        return rgb
    if mode == "Confiança óptica":
        gray = np.clip(optical["confidence"] * 255.0, 0, 255).astype(np.uint8)
        gray[~valid] = 0
        return np.repeat(gray[..., None], 3, axis=2)
    if mode == "Famílias":
        result = _families(prepared)
        rgb = np.full_like(reference, UNASSIGNED_RGB)
        if result is not None:
            assigned = valid & (result["family_map"] > 0)
            rgb[assigned] = result["palette"][result["family_map"][assigned] - 1]
        return rgb
    if mode == "Máscara":
        out = reference.copy()
        out[~valid] = (out[~valid].astype(np.uint16) // 5).astype(np.uint8)
        return out
    if mode in {"Contornos de clusters", "Contornos de famílias"}:
        out = reference.copy()
        if mode == "Contornos de clusters":
            labels = _interpreted(prepared)["labels"]
        else:
            result = _families(prepared)
            if result is None:
                return out
            labels = result["family_map"]
        boundary = boundary_mask(labels, valid=valid)
        out[boundary] = LEGACY_CONTOUR_RGB
        return out
    raise ValueError(f"Unknown preview mode: {mode}")


def build_preview_legend(mode: str, prepared: dict) -> list[dict]:
    """Keep raw identities, interpreted names and family memberships distinct."""
    if mode == 'Clusters em revisão':
        state = prepared['review']
        return [dict(label=workspace_group_label(prepared, c),
                     color=tuple(int(v) for v in state.palette[c]),
                     detail=f"{state.counts(c)} pixels no conjunto; {'pronto' if c in state.completed else 'em revisão'}")
                for c in state.class_ids]
    if mode in {"Clusters brutos", "Clusters interpretados"}:
        optical = prepared["optical"] if mode == "Clusters brutos" else _interpreted(prepared)
        rows = []
        for st in optical.get("stats", []):
            c = int(st["cluster"])
            src = tuple(st.get("source_clusters", (c,)))
            source = "+".join(str(v) for v in src)
            label = f"Cluster {src[0]}" if len(src) == 1 else f"Grupo {c} · clusters {source}"
            if st.get("name"):
                label = f"{st['name']} ({label})"
            rows.append({"label": label, "color": tuple(int(v) for v in optical["palette"][c]),
                         "detail": f"{float(st['area']):.1f}% da área · confiança {confidence_label(st['confidence'])}".replace('.', ',')})
        return rows
    if mode == "Famílias":
        result = _families(prepared)
        if result is None or not result["family_count"]:
            return [{"label": "Nenhuma família criada", "detail": "Selecione clusters na seção Interpretação."}]
        rows = []
        for i, st in enumerate(result["stats"]):
            source = "+".join(str(v) for v in st["source_clusters"])
            rows.append({"label": st.get("name") or f"Família {i + 1}",
                         "color": tuple(int(v) for v in st["color"]),
                         "detail": f"clusters {source} · {int(st['pixels'])} pixels · {float(st['area']):.1f}% da área".replace('.', ',')})
        return rows
    if mode == "Confiança óptica":
        return [{"label": "Baixa", "color": (45, 45, 45), "detail": "atribuição ambígua"},
                {"label": "Alta", "color": (235, 235, 235), "detail": "atribuição clara"}]
    if mode == "Máscara":
        return [{"label": "Área analisada", "color": (235, 235, 235), "detail": "entra na análise"},
                {"label": "Área excluída", "color": (45, 45, 45), "detail": "fundo, anotação ou sem sobreposição"}]
    if mode in {"Contornos de clusters", "Contornos de famílias"}:
        return [{"label": mode, "color": LEGACY_CONTOUR_RGB, "detail": "transições entre rótulos"}]
    return []


def human_analysis_summary(prepared: dict) -> str:
    optical = prepared["optical"]
    review = prepared.get('review')
    group_count = len(review.groups) if review is not None else _interpreted(prepared)['clusters']
    families = _families(prepared)
    n = int(families["family_count"]) if families is not None else 0
    valid_pct = 100.0 * float(np.mean(optical["valid"]))
    return (f"{optical['clusters']} clusters brutos · {group_count} interpretados · "
            f"{n} famílias · área analisada {valid_pct:.1f}%").replace('.', ',')


def preview_title(mode: str, prepared: dict) -> str:
    if mode in {"Clusters brutos", "Clusters interpretados"}:
        result = prepared["optical"] if mode == "Clusters brutos" else _interpreted(prepared)
        return f"{mode}: {result['clusters']} grupos"
    if mode == "Famílias":
        result = _families(prepared)
        n = result["family_count"] if result is not None else 0
        return f"Famílias: {n} " + ("família" if n == 1 else "famílias")
    return "Área analisada" if mode == "Máscara" else mode
