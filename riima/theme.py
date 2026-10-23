"""Tk/ttk adapter for the shared RIIMA presentation tokens.

The scientific palette lives independently in :mod:`riima.data_colors`.
Tk is imported only when applying the theme; no plotting backend is loaded.
"""
from .visual_style import DESKTOP as UI
from .visual_style import FONT_FAMILIES, SPACE, UI_FONT_SIZES, choose_font

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


def icon_image(name, size=16, *, color=None):
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
    elif name == 'panel':
        box((3, 4, 21, 20)); line([(9, 4), (9, 20)])
    elif name == 'legend':
        for y in (5, 11, 17):
            box((3, y, 7, y + 3)); line([(11, y + 1), (21, y + 1)])
    elif name == 'chevron_down':
        line([(5, 9), (12, 16), (19, 9)])
    elif name == 'chevron_right':
        line([(9, 5), (16, 12), (9, 19)])
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


def _install_flat_elements(root, style):
    """Install stretchable surfaces and indicators in this Tcl interpreter."""
    from PIL import Image, ImageDraw, ImageTk

    if hasattr(root, '_riima_theme_images'):
        return
    images = []

    def picture(size, draw_image, *, focus=False):
        scale = 3
        image = Image.new('RGBA', (size[0] * scale, size[1] * scale))
        draw_image(ImageDraw.Draw(image), scale)
        image = image.resize(size, Image.Resampling.LANCZOS)
        if focus:
            ImageDraw.Draw(image).rounded_rectangle((0, 0, size[0]-1, size[1]-1),
                                                     radius=5, outline=ACCENT, width=1)
        photo = ImageTk.PhotoImage(image, master=root)
        images.append(photo)
        return photo

    def surface(fill, outline=None, *, focus=False):
        def draw(d, s):
            if focus:
                d.rounded_rectangle((0, 0, 23*s, 23*s), radius=5*s, fill=ACCENT)
                d.rounded_rectangle((s, s, 22*s, 22*s), radius=4*s, fill=BG_APP)
            d.rounded_rectangle((2*s, 2*s, 21*s, 21*s), radius=3*s,
                                fill=fill, outline=outline, width=s)
        return picture((24, 24), draw, focus=focus)

    def element(name, default, states=(), *, border=0, padding=0, sticky=''):
        style.element_create('RIIMA.' + name, 'image', default, *states,
                             border=border, padding=padding, sticky=sticky)

    for name, base, hover, pressed in (
        ('button', BG_APP, ACCENT_SOFT, BORDER),
        ('primary', ACCENT, ACCENT_HOVER, TEXT_PRIMARY),
    ):
        element(name, surface(base), [('disabled', surface(BG_APP if name == 'button' else BORDER)),
            ('focus', 'pressed', surface(pressed, focus=True)),
            ('focus', 'active', surface(hover, focus=True)),
            ('focus', surface(base, focus=True)), ('pressed', surface(pressed)),
            ('active', surface(hover))], border=6, padding=2, sticky='nswe')
    for name, outline in (('field', CONTROL_BORDER), ('treefield', BORDER)):
        element(name, surface(BG_PANEL, outline), [
            ('disabled', surface(BG_APP, BORDER)), ('focus', surface(BG_PANEL, focus=True)),
            ('active', surface(BG_PANEL, TEXT_SECONDARY))], border=6, padding=3, sticky='nswe')

    def arrow(direction, color=TEXT_SECONDARY, height=16):
        points = {'down': [(5, 6), (9, 10), (13, 6)],
                  'up': [(5, 10), (9, 6), (13, 10)]}[direction]
        return picture((18, height), lambda d, s: d.line(
            [(x*s, (y - (16-height)/2)*s) for x, y in points], fill=color, width=s, joint='curve'))

    for direction in ('down', 'up'):
        element(direction + 'arrow', arrow(direction),
                [('disabled', arrow(direction, CONTROL_BORDER)),
                 ('active', arrow(direction, TEXT_PRIMARY))])
        element('spin.' + direction + 'arrow', arrow(direction, height=10),
                [('disabled', arrow(direction, CONTROL_BORDER, 10)),
                 ('active', arrow(direction, TEXT_PRIMARY, 10))])

    def indicator(kind, selected=False, alternate=False, disabled=False):
        def draw(d, s):
            fill = BORDER if disabled else ACCENT if selected or alternate else BG_PANEL
            outline = BORDER if disabled else ACCENT if selected or alternate else CONTROL_BORDER
            bounds = (3*s, 3*s, 17*s, 17*s)
            if kind == 'check':
                d.rounded_rectangle(bounds, radius=3*s, fill=fill, outline=outline, width=s)
                if alternate:
                    d.line([(6*s, 10*s), (14*s, 10*s)], fill=BG_PANEL, width=2*s)
                elif selected:
                    d.line([(6*s, 10*s), (9*s, 13*s), (14*s, 7*s)],
                           fill=BG_PANEL, width=2*s, joint='curve')
            else:
                d.ellipse(bounds, fill=BG_PANEL if not disabled else BG_APP, outline=outline, width=s)
                if selected:
                    d.ellipse((7*s, 7*s, 13*s, 13*s), fill=outline)
        return picture((24, 20), draw)

    for kind in ('check', 'radio'):
        element(kind + '.indicator', indicator(kind), [
            ('disabled', 'selected', indicator(kind, selected=True, disabled=True)),
            ('disabled', indicator(kind, disabled=True)),
            ('selected', indicator(kind, selected=True)),
            ('alternate', indicator(kind, alternate=True))])
    element('option', surface(BG_APP), [('disabled', surface(BG_APP)),
        ('focus', 'pressed', surface(BORDER, focus=True)),
        ('focus', 'active', surface(ACCENT_SOFT, focus=True)),
        ('focus', surface(BG_APP, focus=True)), ('pressed', surface(BORDER)),
        ('active', surface(ACCENT_SOFT))], border=6, padding=2, sticky='nswe')
    element('heading', surface(BG_APP), [('active', surface(ACCENT_SOFT))], border=6, sticky='nswe')
    element('group', surface(BG_APP), border=6, sticky='nswe')

    for orient in ('Horizontal', 'Vertical'):
        vertical = orient == 'Vertical'
        size = (18, 24) if vertical else (24, 18)
        def trough(d, s, vertical=vertical):
            bounds = (8*s, 0, 10*s, 24*s) if vertical else (0, 8*s, 24*s, 10*s)
            d.rectangle(bounds, fill=BORDER)
        element(orient + '.Scale.trough', picture(size, trough),
                border=(0, 6, 0, 6) if vertical else (6, 0, 6, 0), sticky='nswe')
        def knob(fill, focused=False):
            def draw(d, s):
                if focused:
                    d.ellipse((s, s, 17*s, 17*s), outline=ACCENT, width=s)
                d.ellipse((4*s, 4*s, 14*s, 14*s), fill=fill)
            return picture((18, 18), draw)
        element(orient + '.Scale.slider', knob(ACCENT), [
            ('disabled', knob(CONTROL_BORDER)), ('focus', knob(ACCENT, True)),
            ('active', knob(ACCENT_HOVER))])
        size = (12, 30) if vertical else (30, 12)
        def thumb(color):
            return picture(size, lambda d, s: d.rounded_rectangle(
                (2*s, 2*s, (size[0]-2)*s, (size[1]-2)*s), radius=4*s, fill=color))
        element(orient + '.Scrollbar.trough', picture(size, lambda d, s: None),
                border=4, sticky='nswe')
        element(orient + '.Scrollbar.thumb', thumb(CONTROL_BORDER), [
            ('disabled', thumb(BORDER)), ('pressed', thumb(TEXT_PRIMARY)),
            ('active', thumb(TEXT_SECONDARY))], border=5, sticky='nswe')
        for direction in (('up', 'down') if vertical else ('left', 'right')):
            points = {'up': [(3, 8), (6, 5), (9, 8)], 'down': [(3, 4), (6, 7), (9, 4)],
                      'left': [(8, 3), (5, 6), (8, 9)], 'right': [(4, 3), (7, 6), (4, 9)]}[direction]
            def scroll_arrow(color):
                return picture((12, 12), lambda d, s: d.line(
                    [(x*s, y*s) for x, y in points], fill=color, width=s, joint='curve'))
            element(orient + '.Scrollbar.' + direction + 'arrow', scroll_arrow(TEXT_SECONDARY), [
                ('disabled', scroll_arrow(CONTROL_BORDER)), ('active', scroll_arrow(TEXT_PRIMARY))])

    def button_layout(surface_name):
        return [('RIIMA.' + surface_name, dict(sticky='nswe', children=[
            ('Button.padding', dict(sticky='nswe', children=[('Button.label', dict(sticky='nswe'))]))]))]

    style.layout('TButton', button_layout('button'))
    for name in ('Primary.TButton', 'Selected.TButton', 'Selected.Icon.TButton'):
        style.layout(name, button_layout('primary'))
    style.layout('TMenubutton', [('RIIMA.button', dict(sticky='nswe', children=[
        ('Menubutton.padding', dict(sticky='nswe', children=[
            ('RIIMA.downarrow', dict(side='right', sticky='')),
            ('Menubutton.label', dict(side='left', sticky='nswe'))]))]))])
    for name, widget in (('TEntry', 'Entry'), ('TCombobox', 'Combobox'), ('TSpinbox', 'Spinbox')):
        children = []
        if widget == 'Combobox':
            children.append(('RIIMA.downarrow', dict(side='right', sticky='')))
        elif widget == 'Spinbox':
            children.append(('null', dict(side='right', sticky='', children=[
                ('RIIMA.spin.uparrow', dict(side='top', sticky='e')),
                ('RIIMA.spin.downarrow', dict(side='bottom', sticky='e'))])))
        children.append((widget + '.padding', dict(sticky='nswe', children=[
            (widget + '.textarea', dict(sticky='nswe'))])))
        style.layout(name, [('RIIMA.field', dict(sticky='nswe', children=children))])
    for widget, kind in (('Checkbutton', 'check'), ('Radiobutton', 'radio')):
        style.layout('T' + widget, [('RIIMA.option', dict(sticky='nswe', children=[
            (widget + '.padding', dict(sticky='nswe', children=[
                ('RIIMA.' + kind + '.indicator', dict(side='left', sticky='')),
                (widget + '.label', dict(side='left', sticky='nswe'))]))]))])
    style.layout('Treeview', [('RIIMA.treefield', dict(sticky='nswe', children=[
        ('Treeview.padding', dict(sticky='nswe', children=[('Treeview.treearea', dict(sticky='nswe'))]))]))])
    style.layout('Treeview.Heading', [('RIIMA.heading', dict(sticky='nswe', children=[
        ('Treeheading.padding', dict(sticky='nswe', children=[
            ('Treeheading.image', dict(side='right', sticky='')),
            ('Treeheading.text', dict(sticky='we'))]))]))])
    style.layout('TLabelframe', [('RIIMA.group', dict(sticky='nswe'))])
    for orient in ('Horizontal', 'Vertical'):
        style.layout(orient + '.TScale', [('RIIMA.' + orient + '.Scale.trough',
            dict(sticky='nswe', children=[('RIIMA.' + orient + '.Scale.slider',
                dict(side='left' if orient == 'Horizontal' else 'top', sticky=''))]))])
        directions = ('up', 'down') if orient == 'Vertical' else ('left', 'right')
        style.layout(orient + '.TScrollbar', [('RIIMA.' + orient + '.Scrollbar.trough',
            dict(sticky='nswe', children=[
                ('RIIMA.' + orient + '.Scrollbar.' + direction + 'arrow',
                 dict(side=side, sticky='')) for direction, side in zip(directions,
                     ('top', 'bottom') if orient == 'Vertical' else ('left', 'right'))
            ] + [('RIIMA.' + orient + '.Scrollbar.thumb', dict(sticky='nswe'))]))])
    root._riima_theme_images = images


def apply(root, style) -> None:
    """Apply flat ttk elements while retaining standard widget bindings."""
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
    for key, value in {
        '*Listbox.background': BG_PANEL, '*Listbox.foreground': TEXT_PRIMARY,
        '*Listbox.selectBackground': ACCENT, '*Listbox.selectForeground': TEXT_ON_ACCENT,
        '*Listbox.highlightBackground': BORDER, '*Listbox.highlightColor': ACCENT,
        '*Listbox.highlightThickness': 1, '*Listbox.borderWidth': 0,
        '*Listbox.relief': 'flat', '*Listbox.font': FONT_BODY,
    }.items():
        root.option_add(key, value)
    if 'riima' not in style.theme_names():
        style.theme_create('riima', parent='clam')
    style.theme_use('riima')
    _install_flat_elements(root, style)
    root.configure(bg=BG_APP)
    style.configure('.', background=BG_APP, foreground=TEXT_PRIMARY, font=FONT_BODY,
                    relief='flat', borderwidth=0, lightcolor=BG_APP, darkcolor=BG_APP)
    style.map('.', background=[('disabled', BG_APP)], foreground=[('disabled', TEXT_SECONDARY)])
    style.configure('TFrame', background=BG_APP)
    style.configure('Panel.TFrame', background=BG_PANEL)
    style.configure('TLabel', background=BG_APP, foreground=TEXT_PRIMARY, font=FONT_BODY)
    style.configure('Panel.TLabel', background=BG_PANEL, foreground=TEXT_PRIMARY, font=FONT_BODY)
    for name, role, color, bold in (
        ('Muted', 'meta', TEXT_SECONDARY, False), ('Title', 'title', TEXT_PRIMARY, True),
        ('Section', 'section', TEXT_PRIMARY, True), ('Summary', 'section', TEXT_PRIMARY, True),
        ('Warning', 'body', WARNING, False), ('Danger', 'body', DANGER, False),
    ):
        style.configure(name + '.TLabel', font=font(role, bold=bold), foreground=color)
    style.configure('TButton', font=FONT_BODY, padding=(SPACE_SM, SPACE_XS), width=0, anchor='center',
                    background=BG_APP, foreground=TEXT_PRIMARY, relief='flat', borderwidth=0)
    style.map('TButton', background=[('disabled', BG_APP), ('pressed', BORDER), ('active', ACCENT_SOFT)],
              foreground=[('disabled', TEXT_SECONDARY)], relief=[('pressed', 'flat'), ('active', 'flat')])
    style.configure('Section.TButton', anchor='w', padding=(SPACE_XS, 6), font=font('body', bold=True))
    style.configure('Icon.TButton', padding=(6, 6), width=0)
    for name in ('Selected.TButton', 'Selected.Icon.TButton', 'Primary.TButton'):
        style.configure(name, background=ACCENT, foreground=TEXT_ON_ACCENT, relief='flat')
        style.map(name, background=[('disabled', BORDER), ('active', ACCENT_HOVER)],
                  foreground=[('disabled', TEXT_SECONDARY)])
    style.configure('Primary.TButton', font=font('body', bold=True), padding=(SPACE_MD, 6))
    style.configure('TMenubutton', padding=(SPACE_SM, SPACE_XS), background=BG_APP,
                    foreground=TEXT_PRIMARY, font=FONT_BODY, relief='flat', width=0)
    style.map('TMenubutton', background=[('active', ACCENT_SOFT)], foreground=[('disabled', TEXT_SECONDARY)])
    for widget in ('TCheckbutton', 'TRadiobutton'):
        style.configure(widget, background=BG_APP, font=FONT_BODY, padding=0, relief='flat')
        style.map(widget, background=[('active', ACCENT_SOFT), ('!active', BG_APP)],
                  foreground=[('disabled', TEXT_SECONDARY)])
        style.configure('Panel.' + widget, background=BG_PANEL)
    for widget in ('TEntry', 'TSpinbox', 'TCombobox'):
        style.configure(widget, padding=(5, 3), fieldbackground=BG_PANEL, foreground=TEXT_PRIMARY,
                        selectbackground=ACCENT, selectforeground=TEXT_ON_ACCENT,
                        insertcolor=TEXT_PRIMARY, insertwidth=1, relief='flat', borderwidth=0)
        style.map(widget, fieldbackground=[('disabled', BG_APP), ('readonly', BG_PANEL)],
                  foreground=[('disabled', TEXT_SECONDARY), ('readonly', TEXT_PRIMARY)],
                  selectbackground=[('!disabled', ACCENT)], selectforeground=[('!disabled', TEXT_ON_ACCENT)])
    style.configure('ComboboxPopdownFrame', relief='flat', borderwidth=0)
    style.configure('Treeview', background=BG_PANEL, fieldbackground=BG_PANEL,
                    foreground=TEXT_PRIMARY, font=FONT_BODY, relief='flat', borderwidth=0,
                    rowheight=max(22, tkfont.Font(root=root, font=FONT_BODY).metrics('linespace') + SPACE_XS))
    style.map('Treeview', background=[('selected', ACCENT)], foreground=[('selected', TEXT_ON_ACCENT)])
    style.configure('Treeview.Heading', background=BG_APP, foreground=TEXT_PRIMARY,
                    font=font('body', bold=True), padding=SPACE_XS, relief='flat', borderwidth=0)
    for orient in ('Horizontal', 'Vertical'):
        style.configure(orient + '.TScale', background=BG_APP, relief='flat', borderwidth=0)
        style.configure(orient + '.TScrollbar', background=BG_APP, relief='flat', borderwidth=0)
    style.configure('TSeparator', background=BORDER)
    style.configure('TLabelframe', background=BG_APP, relief='flat', borderwidth=0, labelmargins=(0, 0, 0, 6))
    style.configure('TLabelframe.Label', background=BG_APP, foreground=TEXT_PRIMARY, font=FONT_SECTION)
    style.configure('Operation.Horizontal.TProgressbar', thickness=3, borderwidth=0,
                    background=ACCENT, troughcolor=BG_APP, lightcolor=ACCENT, darkcolor=ACCENT)
    style.configure('View.Toolbutton', padding=(7, 4), font=FONT_BODY, relief='flat')
    style.map('View.Toolbutton', background=[('selected', ACCENT_SOFT)], relief=[('selected', 'flat')])
