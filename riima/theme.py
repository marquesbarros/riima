"""Tk/ttk adapter for the shared RIIMA presentation tokens.

The scientific palette lives independently in :mod:`riima.data_colors`.
Tk is imported only when applying the theme; no plotting backend is loaded.
"""
from .visual_style import FONT_FAMILIES, SPACE, UI, UI_FONT_SIZES, choose_font

# ---------------------------------------------------------------------------
# Palette (light, neutral desktop theme)
# ---------------------------------------------------------------------------

BG_APP = UI['background']
BG_PANEL = UI['surface']
BG_CANVAS = UI['canvas']
BORDER = UI['border']
CONTROL_BORDER = UI['control_border']
TEXT_PRIMARY = UI['text']
TEXT_SECONDARY = UI['muted']
TEXT_ON_ACCENT = UI['on_accent']
TEXT_ON_CANVAS = UI['on_canvas']
CANVAS_MUTED = UI['canvas_muted']
CANVAS_ACCENT = UI['canvas_accent']

ACCENT = UI['accent']
ACCENT_HOVER = UI['accent_hover']
ACCENT_SOFT = UI['accent_soft']

WARNING = UI['warning']
DANGER = UI['danger']
SUGGESTED = UI['suggested']

# ---------------------------------------------------------------------------
# Spacing tokens (px)
# ---------------------------------------------------------------------------

SPACE_XS = SPACE['xs']
SPACE_SM = SPACE['sm']
SPACE_MD = SPACE['md']
SPACE_LG = SPACE['lg']

# ---------------------------------------------------------------------------
# Typography
# ---------------------------------------------------------------------------

FONT_FAMILY = FONT_FAMILIES[0]


def font(role='body', *, bold=False):
    """Return a Tk font tuple in points, using the installed desktop family."""
    result = (FONT_FAMILY, UI_FONT_SIZES[role])
    return result + ('bold',) if bold else result


FONT_TITLE = font('title', bold=True)
FONT_SECTION = font('section', bold=True)
FONT_BODY = font()
FONT_META = font('meta')


def icon_image(name, size=18, *, color=None):
    """Draw the desktop's small line icons in memory, independent of font glyphs."""
    from PIL import Image, ImageDraw
    scale = 3
    image = Image.new('RGBA', (24 * scale, 24 * scale))
    draw = ImageDraw.Draw(image)
    color = color or TEXT_PRIMARY
    def line(points):
        draw.line([(x * scale, y * scale) for x, y in points], fill=color, width=2 * scale, joint='curve')
    def box(bounds):
        draw.rectangle(tuple(int(v * scale) for v in bounds), outline=color, width=2 * scale)
    def circle(bounds):
        draw.ellipse(tuple(int(v * scale) for v in bounds), outline=color, width=2 * scale)
    if name == 'open':
        line([(3, 19), (3, 6), (9, 6), (11, 9), (21, 9), (18, 19), (3, 19), (6, 11), (20, 11)])
    elif name == 'save':
        box((4, 3, 20, 21)); box((8, 3, 16, 9)); box((8, 14, 16, 21))
    elif name in ('export', 'restore'):
        line([(12, 16), (12, 3), (7, 8), (12, 3), (17, 8)])
        line([(4, 14), (4, 20), (20, 20), (20, 14)])
    elif name in ('positive', 'negative'):
        line([(4, 19), (8, 14), (11, 17), (6, 21), (4, 19), (15, 5), (19, 9), (11, 17)])
        line([(16, 3), (22, 3)])
        if name == 'positive': line([(19, 0), (19, 6)])
    elif name == 'erase':
        line([(3, 15), (13, 4), (21, 11), (12, 21), (9, 21), (3, 15), (11, 15), (16, 20), (21, 20)])
    elif name == 'pan':
        line([(12, 3), (12, 21)]); line([(3, 12), (21, 12)])
        for points in ([(8, 7), (12, 3), (16, 7)], [(8, 17), (12, 21), (16, 17)],
                       [(7, 8), (3, 12), (7, 16)], [(17, 8), (21, 12), (17, 16)]): line(points)
    elif name == 'background':
        circle((3, 3, 21, 21)); line([(6, 18), (18, 6)])
    elif name == 'undo':
        line([(3, 9), (14, 9), (20, 13), (20, 18), (13, 21)])
        line([(8, 4), (3, 9), (8, 14)])
    elif name == 'merge':
        line([(3, 4), (10, 11), (12, 11), (12, 21)])
        line([(21, 4), (14, 11), (12, 11)]); line([(8, 17), (12, 21), (16, 17)])
    elif name == 'color':
        circle((3, 3, 21, 21)); circle((7, 6, 9, 8)); circle((14, 6, 16, 8)); circle((6, 13, 8, 15))
    elif name == 'rename':
        line([(4, 20), (6, 14), (17, 3), (21, 7), (10, 18), (4, 20), (6, 14), (10, 18)])
    elif name == 'similarity':
        circle((2, 6, 13, 18)); circle((11, 6, 22, 18))
    elif name in ('zoom_in', 'zoom_out'):
        circle((3, 3, 16, 16)); line([(15, 15), (22, 22)]); line([(6, 10), (13, 10)])
        if name == 'zoom_in': line([(10, 6), (10, 13)])
    elif name == 'clear':
        line([(6, 6), (18, 18)]); line([(6, 18), (18, 6)])
    elif name == 'create':
        circle((3, 3, 21, 21)); line([(7, 12), (17, 12)]); line([(12, 7), (12, 17)])
    elif name == 'delete':
        line([(4, 6), (20, 6)]); line([(8, 3), (16, 3)])
        line([(6, 6), (7, 21), (17, 21), (18, 6)]); line([(10, 10), (10, 17)]); line([(14, 10), (14, 17)])
    elif name == 'activate':
        circle((3, 3, 21, 21)); line([(7, 12), (11, 16), (18, 8)])
    else:
        circle((3, 3, 21, 21)); line([(12, 10), (12, 17)]); box((11, 6, 12, 7))
    return image.resize((size, size), Image.Resampling.LANCZOS)


def apply(root, style) -> None:
    """Apply the RIIMA theme to an existing ``tk.Tk`` root and ``ttk.Style``.

    Switches the ttk base theme to ``clam`` first: Windows' native themes
    (``vista``/``winnative``) delegate most drawing to the OS and ignore
    background/foreground color overrides, so they cannot render a custom
    palette. ``clam`` is a plain, cross-platform, dependency-free ttk theme
    that does respect these overrides.
    """
    from tkinter import font as tkfont

    global FONT_FAMILY, FONT_TITLE, FONT_SECTION, FONT_BODY, FONT_META
    FONT_FAMILY = choose_font(tkfont.families(root=root))
    FONT_TITLE = font('title', bold=True)
    FONT_SECTION = font('section', bold=True)
    FONT_BODY = font()
    FONT_META = font('meta')
    for name in ('TkDefaultFont', 'TkTextFont', 'TkMenuFont', 'TkHeadingFont',
                 'TkCaptionFont', 'TkSmallCaptionFont', 'TkIconFont', 'TkTooltipFont'):
        tkfont.nametofont(name, root=root).configure(family=FONT_FAMILY, size=UI_FONT_SIZES['body'])
    # Native Listbox is also used by lazily-created ttk Combobox dropdowns.
    for key, value in {
        '*Listbox.background': BG_PANEL, '*Listbox.foreground': TEXT_PRIMARY,
        '*Listbox.selectBackground': ACCENT, '*Listbox.selectForeground': TEXT_ON_ACCENT,
        '*Listbox.highlightBackground': CONTROL_BORDER, '*Listbox.highlightColor': ACCENT,
        '*Listbox.font': FONT_BODY,
    }.items():
        root.option_add(key, value)

    try:
        style.theme_use("clam")
    except Exception:
        pass  # if clam isn't available for some reason, keep the platform default rather than fail

    root.configure(bg=BG_APP)

    style.configure(".", background=BG_APP, foreground=TEXT_PRIMARY, font=FONT_BODY)
    style.configure("TFrame", background=BG_APP)
    style.configure("Panel.TFrame", background=BG_PANEL)
    style.configure("TLabel", background=BG_APP, foreground=TEXT_PRIMARY, font=FONT_BODY)
    style.configure("Panel.TLabel", background=BG_PANEL, foreground=TEXT_PRIMARY, font=FONT_BODY)
    style.configure("Muted.TLabel", foreground=TEXT_SECONDARY, font=FONT_META)
    style.configure("Title.TLabel", font=FONT_TITLE, foreground=TEXT_PRIMARY)
    style.configure("Section.TLabel", font=FONT_SECTION, foreground=TEXT_PRIMARY)
    style.configure("Summary.TLabel", font=FONT_SECTION, foreground=TEXT_PRIMARY)
    style.configure("Warning.TLabel", foreground=WARNING, font=FONT_BODY)
    style.configure("Danger.TLabel", foreground=DANGER, font=FONT_BODY)

    # Buttons: one clear primary action (Analisar), everything else secondary/flat.
    style.configure(
        "TButton", font=FONT_BODY, padding=(SPACE_SM, SPACE_XS),
        background=BG_PANEL, foreground=TEXT_PRIMARY, borderwidth=1, bordercolor=CONTROL_BORDER, relief="solid",
    )
    style.map("TButton", background=[("active", ACCENT_SOFT)])
    style.configure('Selected.TButton', background=ACCENT, foreground=TEXT_ON_ACCENT, font=FONT_BODY)
    style.map('Selected.TButton', background=[('disabled', BORDER), ('active', ACCENT_HOVER)],
              foreground=[('disabled', TEXT_SECONDARY)])
    style.configure('Icon.TButton', padding=(4, 4), width=0)
    style.configure('Selected.Icon.TButton', background=ACCENT, foreground=TEXT_ON_ACCENT)
    style.map('Selected.Icon.TButton', background=[('disabled', BORDER), ('active', ACCENT_HOVER)],
              foreground=[('disabled', TEXT_SECONDARY)])
    style.configure('Toolbar.TMenubutton', padding=(SPACE_SM, SPACE_XS), background=BG_PANEL,
                    foreground=TEXT_PRIMARY, bordercolor=CONTROL_BORDER, font=FONT_BODY)
    style.map('Toolbar.TMenubutton', background=[('active', ACCENT_SOFT)],
              foreground=[('disabled', TEXT_SECONDARY)])
    style.configure('Operation.Horizontal.TProgressbar', thickness=3, borderwidth=0,
                    background=ACCENT, troughcolor=BG_APP, lightcolor=ACCENT, darkcolor=ACCENT)
    style.configure(
        "Primary.TButton", font=FONT_SECTION, padding=(SPACE_MD, SPACE_SM),
        background=ACCENT, foreground=TEXT_ON_ACCENT, borderwidth=0,
    )
    style.map(
        "Primary.TButton",
        background=[("disabled", BORDER), ("active", ACCENT_HOVER)],
        foreground=[("disabled", TEXT_SECONDARY)],
    )

    style.configure("TCheckbutton", background=BG_APP, font=FONT_BODY)
    style.configure("Panel.TCheckbutton", background=BG_PANEL, font=FONT_BODY)
    style.configure("TRadiobutton", background=BG_APP, font=FONT_BODY)
    style.configure("Panel.TRadiobutton", background=BG_PANEL, font=FONT_BODY)

    for widget in ('TEntry', 'TSpinbox', 'TCombobox'):
        style.configure(widget, fieldbackground=BG_PANEL, foreground=TEXT_PRIMARY,
                        bordercolor=CONTROL_BORDER, lightcolor=CONTROL_BORDER, darkcolor=CONTROL_BORDER,
                        selectbackground=ACCENT, selectforeground=TEXT_ON_ACCENT,
                        insertcolor=TEXT_PRIMARY, arrowsize=12)
        style.map(widget, fieldbackground=[('disabled', BG_APP), ('readonly', BG_PANEL)],
                  foreground=[('disabled', TEXT_SECONDARY), ('readonly', TEXT_PRIMARY)],
                  selectbackground=[('!disabled', ACCENT)], selectforeground=[('!disabled', TEXT_ON_ACCENT)])
    style.configure('Treeview', background=BG_PANEL, fieldbackground=BG_PANEL,
                    foreground=TEXT_PRIMARY, font=FONT_BODY, bordercolor=CONTROL_BORDER,
                    rowheight=max(22, tkfont.Font(root=root, font=FONT_BODY).metrics('linespace') + SPACE_XS))
    style.map('Treeview', background=[('selected', ACCENT)], foreground=[('selected', TEXT_ON_ACCENT)])
    style.configure('Treeview.Heading', background=BG_APP, foreground=TEXT_PRIMARY,
                    font=font('body', bold=True), padding=SPACE_XS, bordercolor=CONTROL_BORDER)

    style.configure(
        "TScale", background=BG_APP, troughcolor=BORDER,
    )
    style.configure("Horizontal.TScale", background=BG_APP)

    style.configure("View.Toolbutton", padding=(7, 4), font=FONT_BODY)
    style.map("View.Toolbutton", background=[("selected", ACCENT_SOFT)])

    style.configure("TSeparator", background=BORDER)

    style.configure(
        "TLabelframe", background=BG_APP, bordercolor=BORDER, relief="solid", borderwidth=1,
    )
    style.configure("TLabelframe.Label", background=BG_APP, foreground=TEXT_PRIMARY, font=FONT_SECTION)
