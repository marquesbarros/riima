"""Report presentation, numeric scales and labels shared by both renderers."""
import math
import textwrap

import numpy as np

from .visual_style import DOCUMENT, FONT_FAMILIES, PLOT_FONT_PT


CHART_TITLES = {
    'fig1a': 'Área relativa e contribuição acumulada',
    'fig1b': 'Número de regiões e sua área mediana',
    'fig1c': 'Distribuição das regiões por classe de área',
    'fig2': 'Descritores',
    'fig3a': 'Separação descritiva dos atributos',
    'fig3b': 'Perfis padronizados dos grupos',
    'fig4a': 'Agrupamento por similaridade dos perfis',
    'fig4b': 'Similaridade entre grupos',
    'fig5': 'Roseta',
    'fig6': 'Distribuição axial',
    'fig7': 'Eixos das regiões: medianas e intervalos interquartis',
    'fig8': 'Área',
    'fig9': 'Anisotropia',
    'fig10': 'Compacidade e forma das regiões',
}


def matplotlib_style():
    """Return an isolated report profile for ``matplotlib.rc_context``.

    Colors encoding measurements and groups remain the caller's responsibility.
    Importing the profile never loads Matplotlib or changes global preferences.
    """
    return {
        'font.family': 'sans-serif', 'font.sans-serif': list(FONT_FAMILIES),
        'font.size': PLOT_FONT_PT['body'],
        'figure.titlesize': PLOT_FONT_PT['title'],
        'figure.titleweight': 'bold',
        'axes.titlesize': PLOT_FONT_PT['section'],
        'axes.titleweight': 'bold',
        'axes.labelsize': PLOT_FONT_PT['label'],
        'xtick.labelsize': PLOT_FONT_PT['tick'], 'ytick.labelsize': PLOT_FONT_PT['tick'],
        'legend.fontsize': PLOT_FONT_PT['legend'],
        'legend.title_fontsize': PLOT_FONT_PT['label'],
        'figure.facecolor': DOCUMENT['surface'], 'axes.facecolor': DOCUMENT['surface'],
        'savefig.facecolor': DOCUMENT['surface'], 'text.color': DOCUMENT['text'],
        'axes.labelcolor': DOCUMENT['text'], 'axes.titlecolor': DOCUMENT['text'],
        'axes.edgecolor': DOCUMENT['axis'], 'axes.linewidth': .8,
        'xtick.color': DOCUMENT['axis'], 'ytick.color': DOCUMENT['axis'],
        'grid.color': DOCUMENT['grid'], 'grid.linewidth': .8, 'grid.alpha': .65,
        'text.parse_math': False,
    }


def axis_number(value):
    """Plain decimal or ASCII scientific notation, without offsets or math markup."""
    if not np.isfinite(value):
        return 'N/D'
    if value == 0:
        return '0'
    if abs(value) >= 10000 or abs(value) < .001:
        mantissa, exponent = f'{value:.11e}'.split('e')
        return f"{mantissa.rstrip('0').rstrip('.')}e{int(exponent):+d}"
    text = f'{value:.12g}'
    if 'e' in text:
        mantissa, exponent = text.split('e')
        return f'{mantissa}e{int(exponent):+d}'
    return text


def log_ticks(limits):
    """At most eight labeled ticks; retain unlabeled intermediate log ticks."""
    lo, hi = limits
    if not (0 < lo < hi and np.isfinite(lo) and np.isfinite(hi)):
        raise ValueError('A escala logarítmica exige limites positivos, finitos e distintos.')
    exponents = range(math.floor(math.log10(lo)), math.ceil(math.log10(hi))+1)
    candidates = [m * 10.**e for e in exponents for m in (1, 2, 5) if lo <= m*10.**e <= hi]
    if len(candidates) > 8:
        candidates = [10.**e for e in exponents if lo <= 10.**e <= hi]
        candidates = candidates[::max(1, math.ceil(len(candidates)/8))]
    if len(candidates) < 2:
        step = 10.**math.floor(math.log10((hi-lo)/4))
        step *= next(m for m in (1, 2, 2.5, 5, 10) if m*step >= (hi-lo)/4)
        candidates = [i*step for i in range(math.ceil(lo/step), math.floor(hi/step)+1)]
    major = np.unique(np.clip(candidates, lo, hi))
    minor = [m*10.**e for e in exponents for m in range(1, 10)
             if lo <= m*10.**e <= hi and not np.any(np.isclose(m*10.**e, major, rtol=1e-12, atol=0))]
    if not minor:
        minor = np.exp((np.log(major[:-1]) + np.log(major[1:]))/2)
    return major, np.asarray(minor)


def linear_ticks(limits):
    lo, hi = sorted(limits)
    target = (hi-lo)/6
    if not target or not np.isfinite(target):
        return np.asarray([lo])
    step = 10.**math.floor(math.log10(target))
    step *= next(m for m in (1, 2, 2.5, 5, 10) if m*step >= target)
    values = [i*step for i in range(math.ceil(lo/step-1e-12), math.floor(hi/step+1e-12)+1)]
    return np.unique(np.clip(values, lo, hi))


def plot_limits(values, start_zero=False):
    vals = np.asarray(values, float)
    vals = vals[np.isfinite(vals)]
    if not len(vals):
        return (0., 1.)
    lo, hi = float(vals.min()), float(vals.max())
    if start_zero:
        lo = min(0., lo)
    width = hi-lo
    if not width:
        width = max(1., abs(lo))
    return (lo if start_zero else lo-.1*width, hi+.15*width)


def axis_limits(major, minor):
    xlo, xhi = min(major)*.8, max(major)*1.2
    return (xlo, xhi), (min([*minor, xlo/10])*.8, max([*minor, xhi])*1.1)


def wrap_lines(value, width):
    return [line for paragraph in str(value).splitlines() or ['']
            for line in textwrap.wrap(paragraph, width, break_long_words=True) or ['']]


def point_identifiers(points):
    """One callout per coordinate, listing all groups without displacing data."""
    positions = {}
    for group, x, y in points:
        positions.setdefault((float(x), float(y)), []).append(str(group['id']))
    return [(x, y, ', '.join(ids)) for (x, y), ids in positions.items()]
