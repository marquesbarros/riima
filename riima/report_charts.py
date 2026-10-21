"""Descriptive figures rendered from the current analysis on demand."""
from __future__ import annotations

import base64
import io
import textwrap

import numpy as np

from .report_data import gaussian_density, number
from .visual_style import DOCUMENT, PLOT_FONT_PT, contrasting_text
from .data_colors import (DIVERGING_CMAP, MISSING_COLOR, SEPARATION_COLOR,
                          ZSCORE_LIMITS, SIMILARITY_LIMITS)
from .report_plot_style import (axis_number, log_ticks, linear_ticks, plot_limits,
                                axis_limits, wrap_lines, point_identifiers,
                                matplotlib_style, CHART_TITLES)


class _SvgImage(str):
    """Keep the existing base64-string interface while identifying vector payloads."""
    mime_type = 'image/svg+xml'


def make_charts(data, colors):
    """Use native-free SVG if the optional plotting backend cannot be imported."""
    try:
        import matplotlib
        from matplotlib.backends.backend_agg import FigureCanvasAgg
        from matplotlib.colors import LinearSegmentedColormap, to_rgba
        from matplotlib.figure import Figure
        from matplotlib.patches import Circle, Patch
        from matplotlib.ticker import FuncFormatter
    except (ImportError, OSError):
        from .report_svg import make_svg_charts
        return {key: [_SvgImage(page) for page in pages]
                for key, pages in make_svg_charts(data, colors).items()}
    # Rendering/data failures remain visible; only backend loading selects SVG.
    return _make_matplotlib_charts(data, colors)


def _make_matplotlib_charts(data, colors):
    """Matplotlib stays outside RIIMA's startup and interactive classification paths."""
    import matplotlib
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.colors import LinearSegmentedColormap, to_rgba
    from matplotlib.figure import Figure
    from matplotlib.patches import Circle, Patch
    from matplotlib.ticker import FixedLocator, FuncFormatter, NullFormatter, ScalarFormatter
    from matplotlib.lines import Line2D

    groups, regions = data['groups'], data['regions']
    names = {g['id']: g['label'] for g in groups}
    charts = {}

    def newfig(width=10, height=5):
        fig = Figure(figsize=(width, height), layout='constrained')
        FigureCanvasAgg(fig)
        return fig

    def save(key, fig):
        fig.suptitle(CHART_TITLES[key])
        for ax in fig.axes:
            for axis in (ax.xaxis, ax.yaxis):
                if isinstance(axis.get_major_formatter(), ScalarFormatter) and len(axis.get_majorticklocs()):
                    axis.set_major_locator(FixedLocator(linear_ticks(axis.get_view_interval())))
                    axis.set_major_formatter(FuncFormatter(lambda value, _: axis_number(value)))
        buf = io.BytesIO()
        fig.savefig(buf, format='png', dpi=115, facecolor=DOCUMENT['surface'])
        charts.setdefault(key, []).append(base64.b64encode(buf.getvalue()).decode('ascii'))
        fig.clear()

    def label(g):
        return textwrap.fill(g['label'], 28)

    def matrix_height(labels, minimum, margin):
        lines = max((len(text.splitlines()) for text in labels), default=1)
        row_points = lines * PLOT_FONT_PT['annotation'] * 1.2 + 12
        return max(minimum, len(labels) * row_points / 72 + margin)

    def legend_height(groups, minimum=5.5):
        return max(minimum, sum(len(wrap_lines(f"[{g['id']}] {g['label']}", 28))
                                for g in groups)*.18 + len(groups)*.2 + 1)

    def group_legend(ax, groups):
        handles = [Line2D([], [], linestyle='none', marker='o', color=colors[g['id']],
                          markeredgecolor=DOCUMENT['reference'], markersize=7,
                          label='\n'.join(wrap_lines(f"[{g['id']}] {g['label']}", 28))) for g in groups]
        if handles:
            ax.legend(handles=handles, loc='upper left', bbox_to_anchor=(1.02, 1),
                      frameon=False, fontsize=PLOT_FONT_PT['legend'], borderaxespad=0)

    def identify(ax, points, radius=6):
        for x, y, text in point_identifiers(points):
            left = x > np.mean(ax.get_xlim())
            below = y > np.mean(ax.get_ylim())
            offset = radius+5
            ax.annotate('\n'.join(wrap_lines(text, 26)), (x, y),
                        xytext=(-offset if left else offset, -offset if below else offset),
                        ha='right' if left else 'left', va='top' if below else 'bottom',
                        textcoords='offset points', fontsize=PLOT_FONT_PT['annotation'])

    def style(ax):
        ax.grid(axis='y')
        ax.set_axisbelow(True)
        for side in ('top', 'right'):
            ax.spines[side].set_visible(False)

    def empty(ax, text='Sem dados de regiões medidas'):
        ax.text(.5, .5, text, transform=ax.transAxes, ha='center', va='center', wrap=True)

    def finite(values):
        values = np.asarray(values, float)
        return values[np.isfinite(values)]

    def median(values):
        values = finite(values)
        return float(np.median(values)) if len(values) else np.nan

    def panels(polar=False):
        for start in range(0, len(groups), 12):
            page = groups[start:start + 12]
            cols = min(3, len(page))
            rows = (len(page) + cols - 1) // cols
            fig = newfig(4.7 * cols, 4.4 * rows)
            axes = fig.subplots(rows, cols, squeeze=False,
                                subplot_kw={'projection': 'polar'} if polar else {})
            for ax in axes.ravel()[len(page):]:
                ax.set_visible(False)
            yield fig, list(zip(axes.ravel(), page))

    with matplotlib.rc_context(matplotlib_style()):
        ordered = sorted(groups, key=lambda g: -g['area_percent'])
        fig = newfig(11, 5.3)
        ax = fig.subplots()
        x = np.arange(len(ordered))
        values = [g['area_percent'] for g in ordered]
        ax.bar(x, values, color=[colors[g['id']] for g in ordered], width=.62)
        ax.plot(x, np.cumsum(values), color=DOCUMENT['reference'], marker='o', markersize=4)
        for i, value in enumerate(values):
            ax.text(i, value + 1.2, f'{value:.1f}', ha='center', fontsize=PLOT_FONT_PT['annotation'])
        ax.set(ylim=(0, 108), ylabel='Área relativa (%)', xticks=x)
        ax.set_xticklabels([label(g) for g in ordered], rotation=25, ha='right')
        style(ax)
        save('fig1a', fig)

        valid = [g for g in groups if np.isfinite(g['component_area_median_px'])]
        fig = newfig(12.5, legend_height(valid))
        ax = fig.subplots()
        for g in valid:
            ax.scatter(g['components_measured'], g['component_area_median_px'],
                       s=max(30, g['area_percent'] * 18), color=colors[g['id']],
                       edgecolor=DOCUMENT['reference'], linewidth=.5, alpha=.85)
        group_legend(ax, valid)
        ax.set_xlim(plot_limits([g['components_measured'] for g in valid], True))
        ax.set_ylim(plot_limits([g['component_area_median_px'] for g in valid], True))
        identify(ax, [(g, g['components_measured'], g['component_area_median_px']) for g in valid],
                 radius=max([np.sqrt(max(30, g['area_percent']*18)/np.pi) for g in valid]+[6]))
        if not valid:
            empty(ax)
        ax.set(xlabel='Número de regiões medidas', ylabel='Área mediana das regiões (px)')
        style(ax)
        save('fig1b', fig)

        fig = newfig(11, max(4, len(groups) * .45 + 2))
        ax = fig.subplots()
        bins = [0, 25, 100, 500, 2000, np.inf]
        size_labels = ['<25', '25-100', '100-500', '500-2000', '>=2000']
        for i, g in enumerate(groups):
            counts, _ = np.histogram(regions[g['id']]['area_pixels'], bins)
            values = 100 * counts / max(1, counts.sum())
            left = 0.
            for j, value in enumerate(values):
                ax.barh(i, value, left=left, color=colors[g['id']], alpha=.28 + .18 * j,
                        edgecolor='white', height=.66)
                left += value
        ax.set(xlim=(0, 100), xlabel='Fração de regiões por classe de área (%)', yticks=np.arange(len(groups)))
        ax.set_yticklabels([label(g) for g in groups])
        ax.legend(handles=[Patch(facecolor=str(1 - (.28 + .18 * j)), label=s) for j, s in enumerate(size_labels)],
                  title='Área (px): tonalidades da menor para a maior classe', ncols=5,
                  loc='lower center', bbox_to_anchor=(.5, 1.02), frameon=False, fontsize=PLOT_FONT_PT['legend'])
        style(ax)
        save('fig1c', fig)

        for fig, page in panels():
            for ax, g in page:
                descriptors = sorted(data['descriptors'][g['id']], key=lambda d: d[1])
                if descriptors:
                    ax.barh([textwrap.fill(f, 23) for f, _ in descriptors], [v for _, v in descriptors],
                            color=colors[g['id']], height=.64)
                else:
                    empty(ax, 'Sem contraste variável entre grupos')
                ax.axvline(0, color=DOCUMENT['reference'], linewidth=.8)
                ax.set_title(label(g))
                ax.set_xlabel('z-score entre grupos')
                ax.tick_params(axis='y', labelsize=PLOT_FONT_PT['small'])
                style(ax)
            save('fig2', fig)

        ranks = data['ranks']
        for start in range(0, max(1, len(ranks)), 24):
            page = sorted(ranks[start:start + 24], key=lambda r: number(r['separation_eta2']))
            fig = newfig(11, max(4, len(page) * .3 + 1.5))
            ax = fig.subplots()
            if page:
                ax.barh([textwrap.fill(r['feature'], 42) for r in page],
                        [number(r['separation_eta2']) for r in page], color=SEPARATION_COLOR, height=.62)
            else:
                empty(ax, 'Sem atributos amostrados')
            ax.set(xlim=(0, 1.05), xlabel='η² descritivo')
            style(ax)
            save('fig3a', fig)

        feature_index = {f: i + 7 for i, f in enumerate(data['features'])}
        optical = [feature_index[r['feature']] for r in ranks[:16] if r['feature'] in feature_index]
        columns = list(dict.fromkeys(optical + [0, 2, 5, 6]))
        row_labels = [label(g) for g in groups]
        fig = newfig(13, matrix_height(row_labels, max(4, min(14, len(groups) * .45 + 2.5)), 2.5))
        ax = fig.subplots()
        heat = data['z'][:, columns]
        cmap = matplotlib.colormaps[DIVERGING_CMAP].with_extremes(bad=MISSING_COLOR)
        if heat.size:
            im = ax.imshow(heat, cmap=cmap, vmin=ZSCORE_LIMITS[0], vmax=ZSCORE_LIMITS[1], aspect='auto')
            ax.set(xticks=range(len(columns)), yticks=range(len(groups)))
            ax.set_xticklabels([textwrap.fill(data['columns'][j], 22) for j in columns], rotation=45, ha='right', fontsize=PLOT_FONT_PT['small'])
            ax.set_yticklabels(row_labels, fontsize=PLOT_FONT_PT['annotation'])
            for tick, g in zip(ax.get_yticklabels(), groups):
                tick.set_bbox(dict(facecolor=colors[g['id']], alpha=.22, edgecolor='none'))
            if heat.size <= 400:
                for i, j in np.ndindex(heat.shape):
                    v = heat[i, j]
                    ax.text(j, i, f'{v:.1f}' if np.isfinite(v) else 'N/D', ha='center', va='center',
                            fontsize=PLOT_FONT_PT['small'], color=contrasting_text(im.cmap(im.norm(v))))
            fig.colorbar(im, ax=ax, label='z-score entre grupos')
        else:
            empty(ax)
        save('fig3b', fig)

        active = [groups[i] for i in data['active']]
        fig = newfig(8.5, max(4, min(15, len(active) * .42 + 1.5)))
        ax = fig.subplots()
        order = data['order']
        positions = {leaf: i for i, leaf in enumerate(order)}
        heights = {i: 0. for i in range(len(active))}
        for j, (a, b, distance, _) in enumerate(data['linkage']):
            a, b = int(a), int(b)
            ax.plot([heights[a], distance, distance, heights[b]],
                    [positions[a], positions[a], positions[b], positions[b]], color=DOCUMENT['reference'], linewidth=1.3)
            node = len(active) + j
            positions[node] = (positions[a] + positions[b]) / 2
            heights[node] = distance
        ax.set(yticks=range(len(order)), xlabel='Distância de Ward', ylim=(-.6, max(.6, len(order) - .4)))
        ax.set_yticklabels([label(active[i]) for i in order])
        for tick, i in zip(ax.get_yticklabels(), order):
            tick.set_bbox(dict(facecolor=colors[active[i]['id']], alpha=.25, edgecolor='none'))
        if len(active) < 2:
            empty(ax, 'Sem pares de grupos para comparar')
        for spine in ax.spines.values():
            spine.set_visible(False)
        save('fig4a', fig)

        labels = [label(active[i]) for i in order]
        fig = newfig(9.5, matrix_height(labels, max(5, min(13, len(active) * .35 + 2)), 4))
        ax = fig.subplots()
        if active:
            sim = data['similarity'][np.ix_(order, order)]
            im = ax.imshow(sim, cmap=DIVERGING_CMAP, vmin=SIMILARITY_LIMITS[0], vmax=SIMILARITY_LIMITS[1])
            ax.set(xticks=range(len(order)), yticks=range(len(order)))
            ax.set_xticklabels(labels, rotation=45, ha='right', fontsize=PLOT_FONT_PT['annotation'])
            ax.set_yticklabels(labels, fontsize=PLOT_FONT_PT['annotation'])
            if len(order) <= 20:
                for i, j in np.ndindex(sim.shape):
                    ax.text(j, i, f'{sim[i, j]:.2f}', ha='center', va='center', fontsize=PLOT_FONT_PT['small'],
                            color=contrasting_text(im.cmap(im.norm(sim[i, j]))))
            for ticks in (ax.get_xticklabels(), ax.get_yticklabels()):
                for tick, i in zip(ticks, order):
                    tick.set_bbox(dict(facecolor=colors[active[i]['id']], alpha=.22, edgecolor='none'))
            fig.colorbar(im, ax=ax, label='Similaridade')
        else:
            empty(ax)
        save('fig4b', fig)

        for fig, page in panels(polar=True):
            for ax, g in page:
                r = regions[g['id']]
                keep = np.isfinite(r['orientation_deg'])
                th = np.deg2rad(r['orientation_deg'][keep]) % np.pi
                weights = r['anisotropy'][keep] * np.sqrt(r['area_pixels'][keep])
                edges = np.linspace(0, np.pi, 19)
                counts, _ = np.histogram(th, edges, weights=weights)
                centers = (edges[:-1] + edges[1:]) / 2
                ax.bar(np.r_[centers, centers + np.pi], np.tile(counts, 2), width=np.pi / 18,
                       color=colors[g['id']], edgecolor='white', alpha=.88)
                ax.set_theta_zero_location('E')
                ax.set_theta_direction(-1)
                ax.set_thetagrids([0, 90, 180, 270])
                # Label every second ring with round, whole-number weights.
                peak = max(finite(counts), default=0.)
                target_step = max(5., peak / 4)
                magnitude = 10. ** np.floor(np.log10(target_step))
                label_step = next(step * magnitude for step in (1, 2, 5, 10)
                                  if step * magnitude >= target_step)
                label_count = max(1, int(np.ceil(peak * 1.05 / label_step)))
                circles = np.arange(1, label_count * 2 + 1) * label_step / 2
                ax.set_ylim(0, circles[-1])
                ax.set_yticks(circles, labels=[f'{radius:.0f}' if i % 2 else ''
                                             for i, radius in enumerate(circles)])
                ax.tick_params(axis='y', labelsize=PLOT_FONT_PT['tick'])
                ax.set_title(label(g), pad=18)
                if not len(th):
                    empty(ax, 'Não definida')
                ax.text(.98, 1.03, f"C = {g['direction_coherence']:.2f}", transform=ax.transAxes,
                        ha='right', va='bottom', fontsize=PLOT_FONT_PT['annotation'])
            save('fig5', fig)

        for fig, page in panels():
            for ax, g in page:
                r = regions[g['id']]
                keep = np.flatnonzero(np.isfinite(r['orientation_deg']))
                if len(keep) > 1200:
                    keep = np.random.default_rng(42).choice(keep, 1200, replace=False)
                th, radius = np.deg2rad(2 * r['orientation_deg'][keep]), r['anisotropy'][keep]
                if len(keep):
                    cmap = LinearSegmentedColormap.from_list('group_density', ['#ffffff', colors[g['id']]])
                    ax.hexbin(radius * np.cos(th), radius * np.sin(th), gridsize=18,
                              extent=(-1, 1, -1, 1), cmap=cmap, mincnt=1)
                else:
                    empty(ax, 'Não definida')
                ax.add_patch(Circle((0, 0), 1, fill=False, color=DOCUMENT['reference']))
                if np.isfinite(g['dominant_direction_deg']):
                    theta = np.deg2rad(2 * g['dominant_direction_deg'])
                    c = g['direction_coherence']
                    ax.arrow(0, 0, c * np.cos(theta), c * np.sin(theta), width=.01,
                             head_width=.07, length_includes_head=True, color=DOCUMENT['reference'])
                ax.set(xlim=(-1.05, 1.05), ylim=(-1.05, 1.05), aspect='equal',
                       xticks=[-1, 0, 1], yticks=[-1, 0, 1], title=label(g), xlabel='A cos(2θ)', ylabel='A sin(2θ)')
                style(ax)
            save('fig6', fig)

        fig = newfig(12.5, legend_height(groups, 6.4))
        ax = fig.subplots()
        axes_data, minor_data, axes_groups = [], [], []
        for g in groups:
            major, minor = regions[g['id']]['major_axis_px'], regions[g['id']]['minor_axis_px']
            keep = np.isfinite(major) & np.isfinite(minor) & (major > 0) & (minor > 0)
            if not np.any(keep):
                continue
            mq, nq = np.percentile(major[keep], [25, 50, 75]), np.percentile(minor[keep], [25, 50, 75])
            ax.errorbar(mq[1], nq[1], xerr=[[mq[1] - mq[0]], [mq[2] - mq[1]]],
                        yerr=[[nq[1] - nq[0]], [nq[2] - nq[1]]], fmt='o', color=colors[g['id']],
                        ecolor=to_rgba(colors[g['id']], .35), elinewidth=1.2,
                        markersize=7, markeredgecolor=DOCUMENT['reference'], markeredgewidth=.4,
                        capsize=3, capthick=1, barsabove=False, label=g['label'])
            axes_data.extend(mq)
            minor_data.extend(nq)
            axes_groups.append(g)
        if axes_data:
            xs = np.geomspace(min(axes_data) * .8, max(axes_data) * 1.2, 100)
            for ratio in (1, 2, 5, 10):
                ax.plot(xs, xs / ratio, '--', color=DOCUMENT['reference'], alpha=.6, linewidth=.8)
                ax.text(xs[-1], xs[-1] / ratio, f'AR={ratio}', fontsize=PLOT_FONT_PT['annotation'], ha='right', va='bottom')
            ax.set_xscale('log')
            ax.set_yscale('log')
            limits = axis_limits(axes_data, minor_data)
            ax.set_xlim(limits[0])
            ax.set_ylim(limits[1])
            for axis, bounds in zip((ax.xaxis, ax.yaxis), limits):
                major_ticks, minor_ticks = log_ticks(bounds)
                axis.set_major_locator(FixedLocator(major_ticks))
                axis.set_minor_locator(FixedLocator(minor_ticks))
                axis.set_major_formatter(FuncFormatter(lambda value, _: axis_number(value)))
                axis.set_minor_formatter(NullFormatter())
            group_legend(ax, axes_groups)
        else:
            empty(ax)
        ax.set(xlabel='Eixo maior (px)', ylabel='Eixo menor (px)')
        style(ax)
        save('fig7', fig)

        for key, field, logarithm in (('fig8', 'area_pixels', True), ('fig9', 'anisotropy', False)):
            for fig, page in panels():
                for ax, g in page:
                    vals = finite(regions[g['id']][field])
                    if logarithm:
                        vals = np.log10(vals[vals > 0])
                    if len(vals):
                        ax.hist(vals, bins=28 if logarithm else 24, density=True,
                                color=colors[g['id']], alpha=.55, edgecolor='white')
                        lo, hi = (float(vals.min()), float(vals.max())) if logarithm else (0, 1)
                        if hi - lo < 1e-12:
                            lo, hi = lo - .5, hi + .5
                        xs = np.linspace(lo, hi, 160)
                        density = gaussian_density(vals, xs)
                        if density is not None:
                            ax.plot(xs, density, color=DOCUMENT['reference'], linewidth=1.3)
                        ax.axvline(np.median(vals), color=DOCUMENT['reference'], linestyle='--', linewidth=1)
                    else:
                        empty(ax)
                    if not logarithm:
                        ax.set_xlim(0, 1)
                    ax.set(title=label(g), ylabel='Densidade', xlabel='log10(área em pixels)' if logarithm else 'Anisotropia')
                    style(ax)
                save(key, fig)

        fig = newfig(17, legend_height(groups))
        left, right = fig.subplots(1, 2)
        compact = sorted([(g, median(regions[g['id']]['compactness_grid'])) for g in groups],
                         key=lambda pair: -pair[1] if np.isfinite(pair[1]) else np.inf)
        finite_compact = [(g, v) for g, v in compact if np.isfinite(v)]
        if finite_compact:
            left.bar(range(len(finite_compact)), [v for _, v in finite_compact],
                     color=[colors[g['id']] for g, _ in finite_compact])
            left.set_xticks(range(len(finite_compact)))
            left.set_xticklabels([f"[{g['id']}]" for g, _ in finite_compact])
        else:
            empty(left)
        left.set_ylabel('Compacidade mediana')
        left.set_xlim(-.6, max(.6, len(finite_compact)-.4))
        left.set_ylim(0, max([value for _, value in finite_compact]+[1.])*1.1)
        points = []
        for g in groups:
            r = regions[g['id']]
            aspect, eccentricity = median(r['aspect_ratio']), median(r['eccentricity'])
            if np.isfinite(aspect) and np.isfinite(eccentricity):
                right.scatter(aspect, eccentricity, color=colors[g['id']], s=80, edgecolor=DOCUMENT['reference'], linewidth=.5)
                points.append((g, aspect, eccentricity))
        legend_ids = {g['id'] for g, _, _ in points} | {g['id'] for g, _ in finite_compact}
        group_legend(right, [g for g in groups if g['id'] in legend_ids])
        right.set_xlim(plot_limits([x for _, x, _ in points]))
        right.set_ylim(plot_limits([y for _, _, y in points]))
        identify(right, points)
        if not points:
            empty(right)
        right.set(xlabel='Razão axial mediana', ylabel='Excentricidade mediana')
        style(left)
        style(right)
        save('fig10', fig)
        for key in ('fig2', 'fig5', 'fig6', 'fig8', 'fig9'):
            if key not in charts:
                fig = newfig()
                empty(fig.subplots(), 'Sem grupos disponíveis')
                save(key, fig)
    return charts
