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
        "TButton", font=FONT_BODY, padding=(SPACE_SM, SPACE_XS + 2),
        background=BG_PANEL, foreground=TEXT_PRIMARY, borderwidth=1, bordercolor=CONTROL_BORDER, relief="solid",
    )
    style.map("TButton", background=[("active", ACCENT_SOFT)])
    style.configure('Selected.TButton', background=ACCENT, foreground=TEXT_ON_ACCENT, font=FONT_BODY)
    style.map('Selected.TButton', background=[('disabled', BORDER), ('active', ACCENT_HOVER)],
              foreground=[('disabled', TEXT_SECONDARY)])
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
