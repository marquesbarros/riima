"""Shared visual roles, with explicit units and no rendering dependencies."""

FONT_FAMILIES = ('DejaVu Sans', 'Segoe UI', 'Noto Sans', 'Helvetica Neue',
                 'Arial', 'Liberation Sans')

UI = dict(background='#F4F5F7', surface='#FFFFFF', canvas='#202020', border='#D6DDE5',
          control_border='#7B8794', text='#1F2328', muted='#59636F', on_accent='#FFFFFF',
          accent='#2568B0', accent_hover='#1C528D', accent_soft='#E8F1FB',
          warning='#806000', danger='#B33226', on_canvas='#E8E8E8',
          canvas_muted='#BAC3CE', canvas_accent='#75B5FA', suggested='#098842')

DOCUMENT = dict(background=UI['background'], surface=UI['surface'], text=UI['text'],
                muted=UI['muted'], border=UI['border'], accent=UI['accent'],
                accent_soft=UI['accent_soft'], grid=UI['border'], axis=UI['muted'],
                reference='#39434D')

# Tk uses points for positive font sizes; widget spacing uses pixels.
UI_FONT_SIZES = dict(title=16, section=10, body=9, meta=8, viewer=12, empty=11,
                     chart_title=8, chart_tick=7)
SPACE = dict(xs=4, sm=8, md=12, lg=16)
PLOT_FONT_PT = dict(body=10, title=12, section=11, label=10, tick=9, legend=9,
                    annotation=9, small=8)
DOCUMENT_FONT_PX = dict(body=16, title=28, section=20, caption=14, table=14)
CONTACT_FONT_PX = 18


def choose_font(available, default='sans-serif'):
    """Choose an installed family without importing a platform font engine."""
    installed = {str(name).casefold(): str(name) for name in available}
    return next((installed[name.casefold()] for name in FONT_FAMILIES
                 if name.casefold() in installed), default)


def css_font_family():
    return ', '.join(f'"{name}"' for name in FONT_FAMILIES) + ', sans-serif'


def svg_font_family():
    return css_font_family()


def pt_to_svg(points):
    """Convert points to CSS pixels in the SVG viewBox at 96 px per inch."""
    return points * 96 / 72


def contrasting_text(background):
    """Choose black or white for maximum contrast on an opaque RGB background.

    Accept a hex color or normalized RGB(A) channels, without changing the
    scientific color underneath the annotation.
    """
    channels = ([int(background[i:i+2], 16) / 255 for i in (1, 3, 5)]
                if isinstance(background, str) else list(background[:3]))
    linear = [v / 12.92 if v <= .04045 else ((v + .055) / 1.055) ** 2.4 for v in channels]
    luminance = sum(v * weight for v, weight in zip(linear, (.2126, .7152, .0722)))
    return '#000000' if (luminance + .05) / .05 >= 1.05 / (luminance + .05) else '#ffffff'


def document_css():
    """CSS variables for the self-contained HTML document."""
    roles = dict(bg=DOCUMENT['background'], card=DOCUMENT['surface'], ink=DOCUMENT['text'],
                 muted=DOCUMENT['muted'], line=DOCUMENT['border'], accent=DOCUMENT['accent'],
                 note=DOCUMENT['accent_soft'], **{'font-family': css_font_family()})
    roles.update({f'font-{name}': f'{size}px' for name, size in DOCUMENT_FONT_PX.items()})
    roles.update({f'space-{name}': f'{value}px'
                  for name, value in dict(sm=8, md=12, lg=16, xl=24).items()})
    return ':root { ' + ' '.join(f'--{name}:{value};' for name, value in roles.items()) + ' }'


def pillow_font(size_px=CONTACT_FONT_PX):
    """Use a local font file if available; never download or bundle font files."""
    from PIL import ImageFont
    for filename in ('DejaVuSans.ttf', 'segoeui.ttf', 'NotoSans-Regular.ttf',
                     '/System/Library/Fonts/HelveticaNeue.ttc', 'Arial.ttf',
                     'LiberationSans-Regular.ttf'):
        try:
            return ImageFont.truetype(filename, size_px)
        except OSError:
            continue
    try:
        return ImageFont.load_default(size=size_px)
    except TypeError:
        return ImageFont.load_default()
