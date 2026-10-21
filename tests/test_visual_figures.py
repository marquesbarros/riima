"""Composed figures share typography while their image panels retain data pixels."""
import numpy as np

from riima import core, visual_style


def test_contact_sheet_uses_portable_font_and_keeps_patches(tmp_path, monkeypatch):
    from PIL import Image, ImageDraw, ImageFont
    sentinel = ImageFont.load_default()
    monkeypatch.setattr(visual_style, 'pillow_font', lambda: sentinel)
    drawn = []
    original = ImageDraw.ImageDraw.text
    def capture(draw, *args, **kwargs):
        drawn.append(kwargs.get('font'))
        return original(draw, *args, **kwargs)
    monkeypatch.setattr(ImageDraw.ImageDraw, 'text', capture)
    rgb = np.random.default_rng(13).integers(0, 255, (20, 24, 3), np.uint8)
    path = tmp_path / 'sheet.png'
    core.make_contact_sheet(rgb, rgb, {0: [0]}, path, np.array([[9, 136, 66]], np.uint8), 12)
    assert drawn == [sentinel]
    result = np.asarray(Image.open(path))
    np.testing.assert_array_equal(result[44:56, 10:22], core.crop_patch(rgb, 0, 0, 12))
    np.testing.assert_array_equal(result[44:56, 28:40], core.crop_patch(rgb, 0, 0, 12))
