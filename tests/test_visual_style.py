"""Portable typography and contrasts do not require a rendering backend."""
import subprocess
import sys

import pytest


@pytest.mark.parametrize('available, expected', [
    (['Segoe UI', 'Arial'], 'Segoe UI'),
    (['DejaVu Sans', 'Noto Sans'], 'DejaVu Sans'),
    (['Helvetica Neue', 'Helvetica'], 'Helvetica Neue'),
    (['liberation sans'], 'liberation sans'),
    ([], 'sans-serif'),
])
def test_font_fallback_matches_available_platform_families(available, expected):
    from riima.visual_style import choose_font
    assert choose_font(available) == expected


def _luminance(color):
    channels = [int(color[i:i+2], 16) / 255 for i in (1, 3, 5)]
    channels = [v / 12.92 if v <= .04045 else ((v + .055) / 1.055) ** 2.4 for v in channels]
    return sum(v * w for v, w in zip(channels, (.2126, .7152, .0722)))


def test_enabled_text_and_control_boundaries_have_sufficient_contrast():
    from riima.visual_style import UI, DOCUMENT
    pairs = [('text', 'background'), ('muted', 'background'), ('warning', 'background'),
             ('danger', 'background'), ('on_accent', 'accent'), ('on_accent', 'accent_hover'),
             ('on_canvas', 'canvas'), ('canvas_muted', 'canvas'), ('on_accent', 'suggested')]
    for foreground, background in pairs:
        light, dark = sorted((_luminance(UI[foreground]), _luminance(UI[background])), reverse=True)
        assert (light + .05) / (dark + .05) >= 4.5, (foreground, background)
    for foreground in ('text', 'muted', 'accent'):
        light, dark = sorted((_luminance(DOCUMENT[foreground]), _luminance('#ffffff')), reverse=True)
        assert (light + .05) / (dark + .05) >= 4.5
    assert (_luminance('#ffffff') + .05) / (_luminance(UI['control_border']) + .05) >= 3


def test_figure_units_are_explicit_and_do_not_override_backend_defaults():
    from riima.visual_style import PLOT_FONT_PT, pt_to_svg
    assert pt_to_svg(9) == 12
    assert PLOT_FONT_PT['title'] > PLOT_FONT_PT['label'] >= PLOT_FONT_PT['tick']
    completed = subprocess.run([sys.executable, '-c',
        "import sys; from riima import visual_style, data_colors; "
        "assert not any(k.startswith(('matplotlib', 'tkinter', 'PIL')) for k in sys.modules)"],
        capture_output=True, text=True)
    assert completed.returncode == 0, completed.stderr


def test_scientific_color_constants_keep_existing_values_and_ranges():
    from riima import data_colors as colors
    assert colors.UNASSIGNED_RGB == (30, 30, 30)
    assert colors.POSITIVE_RGB == (0, 255, 100)
    assert colors.NEGATIVE_RGB == (255, 70, 70)
    assert colors.ERASE_RGB == (70, 210, 255)
    assert colors.EXCLUDED_RGB == (190, 90, 255)
    assert colors.DIVERGING_CMAP == 'RdBu_r'
    assert colors.DIVERGING_STOPS == ('#2166ac', '#f7f7f7', '#b2182b')
    assert colors.ZSCORE_LIMITS == (-2.7, 2.7)
    assert colors.SIMILARITY_LIMITS == (0., 1.)


@pytest.mark.parametrize('background, foreground', [
    ('#ffffff', '#000000'), ('#000000', '#ffffff'),
    ('#777777', '#000000'), ('#2166ac', '#ffffff'),
])
def test_scientific_cell_annotation_uses_readable_neutral_text(background, foreground):
    from riima.visual_style import contrasting_text
    assert contrasting_text(background) == foreground
