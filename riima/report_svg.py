"""Portable SVG report figures using the existing descriptive data and NumPy.

This renderer deliberately needs no font engine or plotting extension. Histogram,
KDE, axial and Ward definitions follow report_charts; only their presentation is
vector based. Hexagonal density uses the same two staggered lattices, extent and
grid size as the Matplotlib hexbin chart. Numeric SVG metadata records the values
actually drawn, including missing values as JSON null rather than invented zeros.
"""
from __future__ import annotations

import base64
import html
import json
import math
import re
import textwrap

import numpy as np

from .report_data import gaussian_density, number
from .visual_style import DOCUMENT, PLOT_FONT_PT, pt_to_svg, svg_font_family, contrasting_text
from .data_colors import (DIVERGING_STOPS, MISSING_COLOR, SEPARATION_COLOR,
                          ZSCORE_LIMITS, SIMILARITY_LIMITS)
from .report_plot_style import (axis_number, log_ticks, linear_ticks, plot_limits,
                                axis_limits, wrap_lines, point_identifiers, CHART_TITLES)


_KEYS = ('fig1a', 'fig1b', 'fig1c', 'fig2', 'fig3a', 'fig3b', 'fig4a',
         'fig4b', 'fig5', 'fig6', 'fig7', 'fig8', 'fig9', 'fig10')
_EMPTY = 'Sem dados de regiões medidas'


def _finite(values):
    values = np.asarray(values, float)
    return values[np.isfinite(values)]


def _median(values):
    values = _finite(values)
    return float(np.median(values)) if len(values) else np.nan


def _color(value):
    if isinstance(value, str):
        if re.fullmatch(r'#[0-9a-fA-F]{6}', value):
            return value.lower()
        if re.fullmatch(r'#[0-9a-fA-F]{3}', value):
            return '#' + ''.join(c * 2 for c in value[1:]).lower()
        return '#64748b'
    array = np.asarray(value, float).ravel()
    if len(array) < 3 or not np.all(np.isfinite(array[:3])):
        return '#64748b'
    return '#' + ''.join(f'{int(round(c * 255)):02x}' for c in np.clip(array[:3], 0, 1))


def _mix(a, b, amount):
    av = np.array([int(a[i:i+2], 16) for i in (1, 3, 5)])
    bv = np.array([int(b[i:i+2], 16) for i in (1, 3, 5)])
    return _color((av + (bv - av) * np.clip(amount, 0, 1)) / 255)


def _diverging(value, lo, hi):
    if not np.isfinite(value):
        return MISSING_COLOR
    fraction = float(np.clip((value - lo) / (hi - lo), 0, 1))
    if fraction <= .5:
        return _mix(DIVERGING_STOPS[0], DIVERGING_STOPS[1], 2 * fraction)
    return _mix(DIVERGING_STOPS[1], DIVERGING_STOPS[2], 2 * fraction - 1)


def _clean(value):
    if isinstance(value, np.ndarray):
        return _clean(value.tolist())
    if isinstance(value, dict):
        return {str(k): _clean(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_clean(v) for v in value]
    if isinstance(value, (float, np.floating)):
        return float(value) if np.isfinite(value) else None
    if isinstance(value, np.integer):
        return int(value)
    return value


def _fmt(value):
    return axis_number(value)


class _SVG:
    def __init__(self, title, width=1000, height=620):
        self.width, self.height = width, height
        self.parts = []
        title_size = pt_to_svg(PLOT_FONT_PT['title'])
        title_wrap = max(1, int((width-40)/title_size))
        self.header_offset = (len(wrap_lines(title, title_wrap))-1)*title_size*1.18
        self.text(width/2, 27, title, role='title', anchor='middle',
                  wrap=title_wrap, class_='chart-title', font_weight='bold')
        self.heading = self.parts.pop()

    def tag(self, name, **attrs):
        attributes = ' '.join(f'{key.rstrip("_").replace("_", "-")}="{html.escape(str(value), quote=True)}"'
                              for key, value in attrs.items() if value is not None)
        self.parts.append(f'<{name} {attributes}/>')

    def rect(self, x, y, width, height, **attrs):
        self.tag('rect', x=x, y=y, width=max(0, width), height=max(0, height), **attrs)

    def line(self, x1, y1, x2, y2, **attrs):
        self.tag('line', x1=x1, y1=y1, x2=x2, y2=y2, stroke=attrs.pop('stroke', DOCUMENT['axis']), **attrs)

    def circle(self, x, y, radius, **attrs):
        self.tag('circle', cx=x, cy=y, r=max(0, radius), **attrs)

    def text(self, x, y, value, size=None, anchor='start', wrap=0, role='body', **attrs):
        size = pt_to_svg(PLOT_FONT_PT[role]) if size is None else size
        lines = wrap_lines(value, wrap) if wrap else [str(value)]
        attributes = dict(x=x, y=y, font_size=size, text_anchor=anchor, fill=DOCUMENT['text'])
        attributes.update(attrs)
        attr = ' '.join(f'{k.rstrip("_").replace("_", "-")}="{html.escape(str(v), quote=True)}"'
                        for k, v in attributes.items())
        body = ''.join(f'<tspan x="{x}" dy="{0 if i == 0 else size * 1.18}">'
                       f'{html.escape(line)}</tspan>' for i, line in enumerate(lines or ['']))
        self.parts.append(f'<text {attr}>{body}</text>')

    def path(self, points, **attrs):
        if len(points):
            d = 'M ' + ' L '.join(f'{float(x):.6g},{float(y):.6g}' for x, y in points)
            self.tag('path', d=d, fill=attrs.pop('fill', 'none'), **attrs)

    def empty(self, text=_EMPTY):
        self.text(self.width/2, self.height/2, text, role='label', anchor='middle', wrap=65)

    def save(self, key, **metadata):
        metadata = dict(renderer='riima-svg', chart=key, **metadata)
        encoded = html.escape(json.dumps(_clean(metadata), ensure_ascii=False, allow_nan=False))
        title = f'RIIMA: {key}'
        height = self.height + self.header_offset
        svg = (f'<svg xmlns="http://www.w3.org/2000/svg" width="100%" height="{height}" '
               f'viewBox="0 0 {self.width} {height}" role="img" '
               f'aria-label="{html.escape(title)}">'
               f'<title>{title}</title><metadata>{encoded}</metadata>'
               f'<g font-family="{html.escape(svg_font_family(), quote=True)}">'
               f'<rect width="{self.width}" height="{height}" fill="{DOCUMENT["surface"]}"/>'
               f'{self.heading}<g class="chart-body" transform="translate(0 {self.header_offset})">'
               f'{"".join(self.parts)}</g></g></svg>')
        return base64.b64encode(svg.encode('utf-8')).decode('ascii')


class _Plot:
    def __init__(self, svg, xlim=(0, 1), ylim=(0, 1), box=None, log=False):
        self.svg, self.xlim, self.ylim, self.log = svg, xlim, ylim, log
        self.left, self.top, self.width, self.height = box or (105, 70, svg.width-155, svg.height-185)

    def x(self, value):
        lo, hi = self.xlim
        if self.log:
            value, lo, hi = np.log10([value, lo, hi])
        return float(self.left + self.width * (value-lo)/(hi-lo))

    def y(self, value):
        lo, hi = self.ylim
        if self.log:
            value, lo, hi = np.log10([value, lo, hi])
        return float(self.top + self.height * (1 - (value-lo)/(hi-lo)))

    def axes(self, xlabel='', ylabel='', xticks=None, yticks=None, xlabels=None):
        s = self.svg
        if xticks is None:
            xticks = self._ticks(self.xlim)
        if yticks is None:
            yticks = self._ticks(self.ylim)
        for value in yticks:
            y = self.y(value)
            s.line(self.left, y, self.left+self.width, y, stroke=DOCUMENT['grid'],
                   stroke_width=pt_to_svg(.8), stroke_opacity=.65)
            s.text(self.left-12, y+4, _fmt(value), anchor='end', role='tick',
                   fill=DOCUMENT['axis'], class_='axis-tick', data_axis='y', data_value=value)
        for i, value in enumerate(xticks):
            x = self.x(value)
            s.line(x, self.top+self.height, x, self.top+self.height+5)
            label = xlabels[i] if xlabels is not None else _fmt(value)
            s.text(x, self.top+self.height+24, label, anchor='middle', role='tick',
                   wrap=20 if xlabels is not None else 0, fill=DOCUMENT['axis'],
                   class_='axis-tick', data_axis='x', data_value=value)
        if self.log:
            for value in log_ticks(self.xlim)[1]:
                x = self.x(value)
                s.line(x, self.top+self.height, x, self.top+self.height+3,
                       class_='minor-tick', data_axis='x', data_value=value)
            for value in log_ticks(self.ylim)[1]:
                y = self.y(value)
                s.line(self.left-3, y, self.left, y, class_='minor-tick', data_axis='y', data_value=value)
        s.line(self.left, self.top, self.left, self.top+self.height)
        s.line(self.left, self.top+self.height, self.left+self.width, self.top+self.height)
        if xlabel:
            s.text(self.left+self.width/2, self.top+self.height+60, xlabel, anchor='middle', role='label')
        if ylabel:
            s.text(25, self.top+self.height/2, ylabel, anchor='middle', role='label',
                   transform=f'rotate(-90 25 {self.top+self.height/2})')

    def _ticks(self, limits):
        if self.log:
            return log_ticks(limits)[0]
        return linear_ticks(limits)

    def line(self, xs, ys, **attrs):
        points = [(self.x(x), self.y(y)) for x, y in zip(xs, ys)
                  if np.isfinite(x) and np.isfinite(y)]
        self.svg.path(points, stroke=attrs.pop('stroke', DOCUMENT['reference']), stroke_width=1.8, **attrs)

    def point(self, x, y, color, radius=6, **attrs):
        self.svg.circle(self.x(x), self.y(y), radius, fill=color, stroke=DOCUMENT['reference'], stroke_width=.6, **attrs)


def _limits(values, start_zero=False):
    return plot_limits(values, start_zero)


def _bars(svg, labels, values, colors, xlabel, xlim=None):
    plot = _Plot(svg, xlim=xlim or _limits(values, True), ylim=(-.5, max(.5, len(labels)-.5)),
                 box=(315, 75, svg.width-365, svg.height-160))
    plot.axes(xlabel, yticks=[])
    for i, (label, value, color) in enumerate(zip(labels, values, colors)):
        if not np.isfinite(value):
            continue
        y = plot.y(len(labels)-1-i)
        height = min(32, plot.height / max(1, len(labels)) * .65)
        a, b = plot.x(0), plot.x(value)
        svg.rect(min(a, b), y-height/2, abs(b-a), height, fill=color)
        svg.text(plot.left-15, y-5, label, anchor='end', role='legend', wrap=28)
    if not labels:
        svg.empty()
    return plot


def _legend_height(groups, width, size=None):
    size = pt_to_svg(PLOT_FONT_PT['legend']) if size is None else size
    wrap = max(1, int((width-76)/size))
    return sum(len(wrap_lines(g['label'], wrap)) * size*1.18 + 16 for g in groups)


def _scatter_legend(svg, groups, colors, x, y, width, size=None):
    """Each name occupies its own row, independently of coincident plot points."""
    size = pt_to_svg(PLOT_FONT_PT['legend']) if size is None else size
    wrap, entries = max(1, int((width-76)/size)), []
    for group in groups:
        height = len(wrap_lines(group['label'], wrap)) * size*1.18
        svg.circle(x+8, y+size*.65, 6, fill=colors[group['id']], stroke=DOCUMENT['reference'], stroke_width=.6)
        svg.text(x+23, y+size, f"[{group['id']}]", size=size, class_='legend-id')
        svg.text(x+68, y+size, group['label'], size=size, wrap=wrap,
                 class_='scatter-legend', data_group_id=group['id'])
        entries.append(dict(id=group['id'], label=group['label'], x=x, y=y, width=width, height=height))
        y += height + 16
    return entries


def _point_ids(plot, points, radius=7):
    for x, y, text in point_identifiers(points):
        px, py = plot.x(x), plot.y(y)
        left = px > plot.left+plot.width/2
        below = py < plot.top+plot.height/2
        offset = radius+5
        plot.svg.text(px-offset if left else px+offset, py+offset+12 if below else py-offset,
                      text, role='annotation', wrap=26, anchor='end' if left else 'start', class_='scatter-id')


def _hexagonal_counts(x, y, gridsize=18):
    """Counts on the same regular/staggered hexagon lattice as the PNG figure."""
    nx, ny = gridsize, int(gridsize / np.sqrt(3))
    xmin, xmax, ymin, ymax = -1.-2e-9, 1.+2e-9, -1., 1.
    sx, sy = (xmax-xmin)/nx, (ymax-ymin)/ny
    regular = np.array([(i, j) for i in range(nx+1) for j in range(ny+1)], float)
    staggered = np.array([(i+.5, j+.5) for i in range(nx) for j in range(ny)], float)
    # Staggered cells win exact ties, as in the original nearest-lattice test.
    index_centers = np.concatenate([staggered, regular])
    counts = np.zeros(len(index_centers), int)
    if len(x):
        samples = np.column_stack(((x-xmin)/sx, (y-ymin)/sy))
        distance = samples[:, None, :] - index_centers[None, :, :]
        nearest = np.argmin(distance[:, :, 0]**2 + 3*distance[:, :, 1]**2, axis=1)
        counts = np.bincount(nearest, minlength=len(index_centers))
    keep = counts > 0
    centers = index_centers[keep] * np.array([sx, sy]) + np.array([xmin, ymin])
    return centers, counts[keep], sx, sy


def make_svg_charts(data, colors):
    """Return all report chart keys as base64-encoded, self-contained SVG images."""
    groups, regions = data['groups'], data['regions']
    palette = {g['id']: _color(colors[g['id']]) for g in groups}
    charts = {key: [] for key in _KEYS}

    def save(key, svg, **metadata):
        charts[key].append(svg.save(key, **metadata))

    ordered = sorted(groups, key=lambda g: -g['area_percent'])
    values = np.array([g['area_percent'] for g in ordered])
    svg = _SVG(CHART_TITLES['fig1a'], width=max(1000, len(groups)*135+140), height=680)
    plot = _Plot(svg, (-.7, max(.7, len(ordered)-.3)), (0, 108), box=(100, 70, svg.width-145, 405))
    plot.axes(ylabel='Área relativa (%)', xticks=np.arange(len(ordered)), xlabels=[g['label'] for g in ordered])
    for i, (g, value) in enumerate(zip(ordered, values)):
        svg.rect(plot.x(i-.31), plot.y(value), plot.x(i+.31)-plot.x(i-.31), plot.y(0)-plot.y(value), fill=palette[g['id']])
        svg.text(plot.x(i), plot.y(value)-10, f'{value:.1f}', anchor='middle')
    if len(ordered):
        cumulative = np.cumsum(values)
        plot.line(np.arange(len(values)), cumulative)
        for i, value in enumerate(cumulative):
            plot.point(i, value, DOCUMENT['reference'], 4)
    else:
        svg.empty()
    save('fig1a', svg, group_ids=[g['id'] for g in ordered], area_percent=values, cumulative=np.cumsum(values))

    valid = [g for g in groups if np.isfinite(g['component_area_median_px'])]
    svg = _SVG(CHART_TITLES['fig1b'], width=1100,
               height=max(650, _legend_height(valid, 295)+160))
    plot = _Plot(svg, _limits([g['components_measured'] for g in valid], True),
                 _limits([g['component_area_median_px'] for g in valid], True),
                 box=(105, 70, 600, svg.height-185))
    plot.axes('Número de regiões medidas', 'Área mediana das regiões (px)')
    for g in valid:
        x, y = g['components_measured'], g['component_area_median_px']
        plot.point(x, y, palette[g['id']], np.sqrt(max(30, g['area_percent']*18)/np.pi),
                   class_='scatter-point', data_group_id=g['id'])
    legend = _scatter_legend(svg, valid, palette, 765, 75, 295)
    _point_ids(plot, [(g, g['components_measured'], g['component_area_median_px']) for g in valid],
               radius=max([np.sqrt(max(30, g['area_percent']*18)/np.pi) for g in valid]+[6]))
    if not valid:
        svg.empty()
    save('fig1b', svg, groups=[dict(id=g['id'], components=g['components_measured'],
                                  median_area=g['component_area_median_px'], marker_area=max(30, g['area_percent']*18)) for g in valid],
         legend=legend, limits=[plot.xlim, plot.ylim])

    svg = _SVG(CHART_TITLES['fig1c'], width=1100, height=max(400, len(groups)*65+190))
    plot = _Plot(svg, (0, 100), (-.5, max(.5, len(groups)-.5)), box=(310, 95, 735, svg.height-185))
    plot.axes('Fração de regiões por classe de área (%)', yticks=[])
    size_labels = ['<25', '25-100', '100-500', '500-2000', '>=2000']
    svg.text(315, 68, 'Área (px): ' + ' | '.join(size_labels), role='body')
    fractions = []
    for i, g in enumerate(groups):
        counts, _ = np.histogram(regions[g['id']]['area_pixels'], [0, 25, 100, 500, 2000, np.inf])
        percentages = 100 * counts / max(1, counts.sum())
        y, left = plot.y(len(groups)-1-i), 0.
        height = min(36, plot.height/max(1, len(groups))*.66)
        svg.text(295, y-4, g['label'], anchor='end', wrap=28)
        for j, value in enumerate(percentages):
            svg.rect(plot.x(left), y-height/2, plot.x(left+value)-plot.x(left), height,
                     fill=palette[g['id']], fill_opacity=.28+.18*j, stroke='white')
            left += value
        fractions.append(dict(id=g['id'], counts=counts, percent=percentages))
    if not groups:
        svg.empty()
    save('fig1c', svg, groups=fractions)

    for g in groups:
        descriptors = sorted(data['descriptors'][g['id']], key=lambda d: d[1])
        svg = _SVG(CHART_TITLES['fig2'] + ': ' + g['label'], width=1100, height=max(430, len(descriptors)*65+180))
        descriptors = [(name, value) for name, value in descriptors if np.isfinite(value)]
        vals = [value for _, value in descriptors]
        lim = max([abs(v) for v in vals] + [1.]) * 1.15
        plot = _bars(svg, [name for name, _ in descriptors], vals, [palette[g['id']]]*len(vals),
                     'z-score entre grupos', (-lim, lim))
        svg.line(plot.x(0), plot.top, plot.x(0), plot.top+plot.height, stroke=DOCUMENT['reference'])
        if not descriptors:
            svg.empty('Sem contraste variável entre grupos')
        save('fig2', svg, group_id=g['id'], descriptors=descriptors)

    ranks = data['ranks']
    for start in range(0, max(1, len(ranks)), 24):
        page = sorted(ranks[start:start+24], key=lambda r: number(r['separation_eta2']))
        svg = _SVG(CHART_TITLES['fig3a'], width=1100, height=max(430, len(page)*55+160))
        _bars(svg, [r['feature'] for r in page], [number(r['separation_eta2']) for r in page],
              [SEPARATION_COLOR]*len(page), 'η² descritivo', (0, 1.05))
        if not page:
            svg.empty('Sem atributos amostrados')
        save('fig3a', svg, ranks=page)

    feature_index = {f: i+7 for i, f in enumerate(data['features'])}
    optical = [feature_index[r['feature']] for r in ranks[:16] if r['feature'] in feature_index]
    columns = list(dict.fromkeys(optical + [0, 2, 5, 6]))
    heat = data['z'][:, columns]
    svg = _heatmap(CHART_TITLES['fig3b'], heat, [g['label'] for g in groups],
                   [data['columns'][i] for i in columns], [palette[g['id']] for g in groups], *ZSCORE_LIMITS,
                   'z-score entre grupos', decimals=1, annotate=heat.size <= 400)
    save('fig3b', svg, columns=columns, values=heat)

    active = [groups[i] for i in data['active']]
    order = data['order']
    svg = _SVG(CHART_TITLES['fig4a'], width=1100, height=max(430, len(active)*65+150))
    distances = [row[2] for row in data['linkage']]
    plot = _Plot(svg, (0, max(distances, default=1.)*1.1 or 1.), (-.6, max(.6, len(order)-.4)),
                 box=(320, 65, 720, svg.height-145))
    plot.axes('Distância de Ward', yticks=[])
    positions = {leaf: len(order)-1-i for i, leaf in enumerate(order)}
    heights = {i: 0. for i in range(len(active))}
    for j, (a, b, distance, _) in enumerate(data['linkage']):
        a, b = int(a), int(b)
        plot.line([heights[a], distance, distance, heights[b]],
                  [positions[a], positions[a], positions[b], positions[b]], stroke=DOCUMENT['reference'])
        positions[len(active)+j] = (positions[a]+positions[b])/2
        heights[len(active)+j] = distance
    for i, leaf in enumerate(order):
        y = plot.y(len(order)-1-i)
        svg.rect(12, y-22, 292, 42, fill=palette[active[leaf]['id']], fill_opacity=.22)
        svg.text(294, y-4, active[leaf]['label'], role='legend', anchor='end', wrap=28)
    if len(active) < 2:
        svg.empty('Sem pares de grupos para comparar')
    save('fig4a', svg, linkage=data['linkage'], order=order)

    similarity = data['similarity'][np.ix_(order, order)]
    labels = [active[i]['label'] for i in order]
    svg = _heatmap(CHART_TITLES['fig4b'], similarity, labels, labels,
                   [palette[active[i]['id']] for i in order], *SIMILARITY_LIMITS, 'Similaridade',
                   decimals=2, annotate=len(order) <= 20)
    save('fig4b', svg, values=similarity, order=order)

    for g in groups:
        r, color = regions[g['id']], palette[g['id']]
        keep = np.isfinite(r['orientation_deg'])
        angles = np.deg2rad(r['orientation_deg'][keep]) % np.pi
        weights = r['anisotropy'][keep] * np.sqrt(r['area_pixels'][keep])
        edges = np.linspace(0, np.pi, 19)
        counts, _ = np.histogram(angles, edges, weights=weights)
        peak = max(_finite(counts), default=0.)
        target_step = max(5., peak/4)
        magnitude = 10.**np.floor(np.log10(target_step))
        label_step = next(step*magnitude for step in (1, 2, 5, 10) if step*magnitude >= target_step)
        label_count = max(1, int(np.ceil(peak*1.05/label_step)))
        circles = np.arange(1, label_count*2+1)*label_step/2
        svg = _SVG(CHART_TITLES['fig5'] + ': ' + g['label'], width=820, height=720)
        cx, cy, radius = 410, 360, 255
        scale = radius/circles[-1]
        for half in (0, np.pi):
            for a, b, value in zip(edges[:-1]+half, edges[1:]+half, counts):
                if not np.isfinite(value):
                    continue
                rr = max(0., value*scale)
                start = (cx+rr*np.cos(a), cy+rr*np.sin(a))
                end = (cx+rr*np.cos(b), cy+rr*np.sin(b))
                d = (f'M {cx},{cy} L {start[0]:.6g},{start[1]:.6g} '
                     f'A {rr:.6g},{rr:.6g} 0 0 1 {end[0]:.6g},{end[1]:.6g} Z')
                svg.tag('path', d=d, fill=color, fill_opacity=.88, stroke='white', stroke_width=.7, class_='rose-bin')
        for i, value in enumerate(circles):
            rr = value*scale
            svg.circle(cx, cy, rr, fill='none', stroke=DOCUMENT['reference'], stroke_opacity=.4, stroke_width=.8, class_='rose-ring')
            if i % 2:
                # Text nodes use direct content so simple readers also see each numeric label.
                svg.parts.append(f'<text x="{cx+rr*.707:.5g}" y="{cy-rr*.707:.5g}" '
                                 f'font-size="{pt_to_svg(PLOT_FONT_PT["tick"])}" fill="{DOCUMENT["axis"]}" class="rose-radius">{value:.0f}</text>')
        for deg, a in [(0, 0), (90, np.pi/2), (180, np.pi), (270, 3*np.pi/2)]:
            svg.line(cx, cy, cx+radius*np.cos(a), cy+radius*np.sin(a), stroke=DOCUMENT['axis'], stroke_opacity=.5)
            svg.text(cx+(radius+27)*np.cos(a), cy+(radius+27)*np.sin(a)+5, f'{deg}°', anchor='middle', role='tick')
        svg.text(410, 680, f"C = {g['direction_coherence']:.2f}; peso = anisotropia × √área", anchor='middle', role='body')
        if not len(angles):
            svg.text(cx, cy, 'Não definida', anchor='middle', role='label')
        save('fig5', svg, group_id=g['id'], counts=counts, edges_rad=edges, radius_ticks=circles,
             label_step=label_step, weights=weights)

        keep = np.flatnonzero(np.isfinite(r['orientation_deg']))
        if len(keep) > 1200:
            keep = np.random.default_rng(42).choice(keep, 1200, replace=False)
        theta, radii = np.deg2rad(2*r['orientation_deg'][keep]), r['anisotropy'][keep]
        x, y = radii*np.cos(theta), radii*np.sin(theta)
        finite_xy = np.isfinite(x) & np.isfinite(y)
        x, y = x[finite_xy], y[finite_xy]
        centers, hex_counts, sx, sy = _hexagonal_counts(x, y)
        svg = _SVG(CHART_TITLES['fig6'] + ': ' + g['label'], width=820, height=720)
        plot = _Plot(svg, (-1.05, 1.05), (-1.05, 1.05), box=(145, 85, 545, 545))
        plot.axes('A cos(2θ)', 'A sin(2θ)', xticks=[-1, 0, 1], yticks=[-1, 0, 1])
        if len(hex_counts):
            lo, hi = hex_counts.min(), hex_counts.max()
            for (hx, hy), count in zip(centers, hex_counts):
                amount = (count-lo)/(hi-lo) if hi > lo else .35
                poly = [(hx+dx*sx, hy+dy*sy/3) for dx, dy in [(.5, -.5), (.5, .5), (0, 1), (-.5, .5), (-.5, -.5), (0, -1)]]
                svg.tag('polygon', points=' '.join(f'{plot.x(px):.6g},{plot.y(py):.6g}' for px, py in poly),
                        fill=_mix('#ffffff', color, amount), stroke='none')
        else:
            svg.text(410, 360, 'Não definida', anchor='middle', role='label')
        svg.circle(plot.x(0), plot.y(0), plot.width/2.1, fill='none', stroke=DOCUMENT['reference'], stroke_width=1.3)
        if np.isfinite(g['dominant_direction_deg']):
            direction = np.deg2rad(2*g['dominant_direction_deg'])
            coherence = g['direction_coherence']
            _arrow(svg, plot.x(0), plot.y(0), plot.x(coherence*np.cos(direction)), plot.y(coherence*np.sin(direction)))
        save('fig6', svg, group_id=g['id'], x=x, y=y, hex_centers=centers,
             hex_counts=hex_counts, gridsize=18, sampled_indices=keep)

    _flinn(groups, regions, palette, save)

    for key, field, logarithm in [('fig8', 'area_pixels', True), ('fig9', 'anisotropy', False)]:
        for g in groups:
            values = _finite(regions[g['id']][field])
            if logarithm:
                values = np.log10(values[values > 0])
            svg = _SVG(CHART_TITLES[key] + ': ' + g['label'], height=640)
            density, edges, xs, kde, median = [], [], [], None, None
            if len(values):
                density, edges = np.histogram(values, bins=28 if logarithm else 24, density=True)
                lo, hi = (float(values.min()), float(values.max())) if logarithm else (0., 1.)
                if hi-lo < 1e-12:
                    lo, hi = lo-.5, hi+.5
                xs = np.linspace(lo, hi, 160)
                kde = gaussian_density(values, xs)
                ymax = max(float(np.max(density)), float(np.max(kde)) if kde is not None else 0.) * 1.15
                xlim = (min(lo, edges[0]), max(hi, edges[-1])) if logarithm else (0., 1.)
                plot = _Plot(svg, xlim, (0, ymax or 1))
                plot.axes('log10(área em pixels)' if logarithm else 'Anisotropia', 'Densidade')
                for left, right, value in zip(edges[:-1], edges[1:], density):
                    left, right = max(left, xlim[0]), min(right, xlim[1])
                    if right > left:
                        svg.rect(plot.x(left), plot.y(value), plot.x(right)-plot.x(left), plot.y(0)-plot.y(value),
                                 fill=palette[g['id']], fill_opacity=.55, stroke='white', stroke_width=.6)
                if kde is not None:
                    plot.line(xs, kde)
                median = float(np.median(values))
                svg.line(plot.x(median), plot.top, plot.x(median), plot.top+plot.height,
                         stroke=DOCUMENT['reference'], stroke_dasharray='6 5')
            else:
                svg.empty()
            save(key, svg, group_id=g['id'], density=density, edges=edges, kde_x=xs, kde=kde, median=median)

    compact = sorted([(g, _median(regions[g['id']]['compactness_grid'])) for g in groups],
                     key=lambda pair: -pair[1] if np.isfinite(pair[1]) else np.inf)
    compact = [(g, value) for g, value in compact if np.isfinite(value)]
    points = []
    for g in groups:
        aspect, eccentricity = _median(regions[g['id']]['aspect_ratio']), _median(regions[g['id']]['eccentricity'])
        if np.isfinite(aspect) and np.isfinite(eccentricity):
            points.append((g, aspect, eccentricity))
    legend_ids = {g['id'] for g, _, _ in points} | {g['id'] for g, _ in compact}
    legend_groups = [g for g in groups if g['id'] in legend_ids]
    svg = _SVG(CHART_TITLES['fig10'], width=max(1600, len(groups)*90+900),
               height=max(700, _legend_height(legend_groups, 340)+160))
    left = _Plot(svg, (-.6, max(.6, len(compact)-.4)), (0, max([v for _, v in compact]+[1.])*1.1),
                 box=(90, 75, svg.width*.36-90, 440))
    left.axes(ylabel='Compacidade mediana', xticks=range(len(compact)), xlabels=[f"[{g['id']}]" for g, _ in compact])
    for i, (g, value) in enumerate(compact):
        svg.rect(left.x(i-.3), left.y(value), left.x(i+.3)-left.x(i-.3), left.y(0)-left.y(value), fill=palette[g['id']])
    right = _Plot(svg, _limits([x for _, x, _ in points]), _limits([y for _, _, y in points]),
                  box=(svg.width*.44, 75, svg.width*.30-50, 440))
    right.axes('Razão axial mediana', '')
    label_x, label_y = right.left-68, right.top+right.height/2
    svg.text(label_x, label_y, 'Excentricidade mediana', anchor='middle', role='label',
             transform=f'rotate(-90 {label_x} {label_y})')
    for g, aspect, eccentricity in points:
        right.point(aspect, eccentricity, palette[g['id']], 7,
                    class_='scatter-point', data_group_id=g['id'])
    legend = _scatter_legend(svg, legend_groups, palette, svg.width*.76, 75, 340)
    _point_ids(right, points)
    if not compact and not points:
        svg.empty()
    save('fig10', svg, compactness=[dict(id=g['id'], median=v) for g, v in compact],
         shape=[dict(id=g['id'], median_aspect=x, median_eccentricity=y) for g, x, y in points], legend=legend,
         limits=[right.xlim, right.ylim], compactness_limits=[left.xlim, left.ylim])

    # A report with no groups still has a complete, explicit set of chart sections.
    for key in _KEYS:
        if not charts[key]:
            svg = _SVG('Sem grupos disponíveis')
            svg.empty()
            save(key, svg)
    return charts


def _heatmap(title, values, row_labels, column_labels, colors, lo, hi, scale_label, decimals=1, annotate=True):
    values = np.asarray(values, float)
    rows, cols = values.shape
    row_size, column_size = (pt_to_svg(PLOT_FONT_PT[role]) for role in ('legend', 'tick'))
    row_lines = max((len(wrap_lines(name, 27)) for name in row_labels), default=1)
    column_lines = max((len(wrap_lines(name, 16)) for name in column_labels), default=1)
    bottom = max(200, column_lines*column_size*1.18+65)
    row_height = max(65, row_lines*row_size*1.18+16)
    width, height = max(1000, cols*100+365), max(490, rows*row_height+70+bottom)
    svg = _SVG(title, width, height)
    left, top, cellw, cellh = 285, 70, (width-375)/max(1, cols), (height-70-bottom)/max(1, rows)
    if not rows or not cols:
        svg.empty()
        return svg
    for i, name in enumerate(row_labels):
        svg.rect(8, top+i*cellh+3, 260, cellh-6, fill=colors[i], fill_opacity=.22)
        lines = len(wrap_lines(name, 27))
        baseline = top+(i+.5)*cellh-(lines-1)*row_size*1.18/2+row_size*.35
        svg.text(258, baseline, name, role='legend', anchor='end', wrap=27)
    for j, name in enumerate(column_labels):
        x, y = left+(j+.5)*cellw, top+rows*cellh+25
        svg.text(x, y, name, role='tick', wrap=16, anchor='middle')
    for i, j in np.ndindex(values.shape):
        value = values[i, j]
        background = _diverging(value, lo, hi)
        svg.rect(left+j*cellw, top+i*cellh, cellw, cellh, fill=background,
                 stroke='white', stroke_width=.5, class_='heatmap-cell')
        if annotate:
            svg.text(left+(j+.5)*cellw, top+(i+.5)*cellh+5,
                     f'{value:.{decimals}f}' if np.isfinite(value) else 'N/D', role='small', anchor='middle',
                     class_='heatmap-value', fill=contrasting_text(background))
    bar_x, bar_y, bar_h = width-63, top, max(100, rows*cellh)
    for i in range(80):
        svg.rect(bar_x, bar_y+i*bar_h/80, 18, bar_h/80+1, fill=_diverging(hi-(hi-lo)*i/79, lo, hi))
    for value in np.linspace(lo, hi, 5):
        svg.text(bar_x+24, bar_y+(hi-value)/(hi-lo)*bar_h+4, _fmt(value), role='small')
    svg.text(left+(width-left)/2, height-25, scale_label, anchor='middle', role='label')
    return svg


def _arrow(svg, x1, y1, x2, y2):
    distance = math.hypot(x2-x1, y2-y1)
    if distance < 1e-12:
        return
    ux, uy = (x2-x1)/distance, (y2-y1)/distance
    svg.line(x1, y1, x2, y2, stroke=DOCUMENT['reference'], stroke_width=2.3)
    points = [(x2, y2), (x2-12*ux+5*uy, y2-12*uy-5*ux), (x2-12*ux-5*uy, y2-12*uy+5*ux)]
    svg.tag('polygon', points=' '.join(f'{x:.6g},{y:.6g}' for x, y in points), fill=DOCUMENT['reference'])


def _flinn(groups, regions, colors, save):
    summaries, axes_data, minor_data = [], [], []
    for g in groups:
        major, minor = regions[g['id']]['major_axis_px'], regions[g['id']]['minor_axis_px']
        keep = np.isfinite(major) & np.isfinite(minor) & (major > 0) & (minor > 0)
        if np.any(keep):
            mq, nq = np.percentile(major[keep], [25, 50, 75]), np.percentile(minor[keep], [25, 50, 75])
            summaries.append(dict(id=g['id'], label=g['label'], major_quartiles=mq, minor_quartiles=nq))
            axes_data.extend(mq)
            minor_data.extend(nq)
    svg = _SVG(CHART_TITLES['fig7'], width=1100,
               height=max(680, _legend_height(summaries, 230)+160))
    if summaries:
        (xlo, xhi), (ylo, yhi) = axis_limits(axes_data, minor_data)
        plot = _Plot(svg, (xlo, xhi), (ylo, yhi), box=(105, 75, 660, svg.height-185), log=True)
        plot.axes('Eixo maior (px)', 'Eixo menor (px)')
        for ratio in (1, 2, 5, 10):
            xs = np.geomspace(xlo, xhi, 100)
            plot.line(xs, xs/ratio, stroke=DOCUMENT['reference'], stroke_dasharray='5 5', stroke_opacity=.6)
            svg.text(plot.x(xhi)+8, plot.y(xhi/ratio)+3, f'AR={ratio}', role='tick')
        for i, row in enumerate(summaries):
            mq, nq, color = row['major_quartiles'], row['minor_quartiles'], colors[row['id']]
            x, y = plot.x(mq[1]), plot.y(nq[1])
            attributes = dict(stroke=color, stroke_width=2, stroke_opacity=.35, class_='iqr')
            svg.line(plot.x(mq[0]), y, plot.x(mq[2]), y, **attributes)
            svg.line(x, plot.y(nq[0]), x, plot.y(nq[2]), **attributes)
            for xx in (plot.x(mq[0]), plot.x(mq[2])):
                svg.line(xx, y-4, xx, y+4, **attributes)
            for yy in (plot.y(nq[0]), plot.y(nq[2])):
                svg.line(x-4, yy, x+4, yy, **attributes)
            plot.point(mq[1], nq[1], color, 7, fill_opacity=1, class_='flinn-point')
        _scatter_legend(svg, summaries, colors, 840, 75, 230)
    else:
        svg.empty()
    save('fig7', svg, groups=summaries,
         limits=[(xlo, xhi), (ylo, yhi)] if summaries else None,
         major_ticks=[log_ticks(limits)[0] for limits in ((xlo, xhi), (ylo, yhi))] if summaries else [])
