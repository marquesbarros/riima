"""Institutional chart styling preserves the scientific rendering contract."""
import numpy as np
import pytest

from test_riima_report_svg import chart_data, decode, NS


COLORS = {0: '#098842', 1: '#d3256f', 2: '#2171b5'}


def test_matplotlib_long_heatmap_labels_have_separate_bounded_rows(monkeypatch):
    from matplotlib.figure import Figure
    from riima.report_charts import make_charts
    data = chart_data(3)
    for group in data['groups']:
        group['label'] = f"Grupo {group['id']} " + 'bordas e textura ' * 14
    captured = []
    def inspect(fig, *args, **kwargs):
        for ax in fig.axes:
            if not ax.images:
                continue
            fig.set_dpi(115)
            fig.canvas.draw()
            renderer = fig.canvas.get_renderer()
            boxes = [label.get_window_extent(renderer) for label in ax.get_yticklabels()]
            captured.append((boxes,ax.get_window_extent(renderer),ax.images[0].get_array().copy()))
    monkeypatch.setattr(Figure, 'savefig', inspect)
    make_charts(data, COLORS)
    assert len(captured) == 2
    for boxes, plot, _ in captured:
        for box in boxes:
            assert plot.y0 <= box.y0 and box.y1 <= plot.y1
        for upper, lower in zip(boxes, boxes[1:]):
            assert lower.y1 + 2 <= upper.y0
    np.testing.assert_equal(captured[0][2], data['z'][:, [7,0,2,5,6]])
    order = data['order']
    np.testing.assert_equal(captured[1][2],data['similarity'][np.ix_(order,order)])


def test_matplotlib_vertical_colorbars_keep_their_horizontal_axis_empty(monkeypatch):
    from matplotlib.figure import Figure
    from riima.report_charts import make_charts
    colorbars = []
    def inspect(fig, *args, **kwargs):
        fig.canvas.draw()
        for ax in fig.axes:
            if ax.get_label() == '<colorbar>':
                colorbars.append([label.get_text() for label in ax.get_xticklabels() if label.get_visible()])
    monkeypatch.setattr(Figure, 'savefig', inspect)
    make_charts(chart_data(), COLORS)
    assert colorbars == [[], []]


@pytest.mark.parametrize('columns', [2,3])
def test_svg_long_heatmap_labels_fit_their_rows_and_column_margin(columns):
    from riima.report_svg import _heatmap
    labels = [f'Grupo {i} ' + 'bordas e textura ' * 14 for i in range(3)]
    column_labels = labels[:columns]
    values = np.arange(3*columns,dtype=float).reshape(3,columns)
    svg = _heatmap('Matriz',values,labels,column_labels,list(COLORS.values()),0,9,'Escala')
    _,root,_ = decode(svg.save('matrix',values=values))
    cells = root.findall('.//svg:rect[@class="heatmap-cell"]',NS)
    texts = root.findall('.//svg:text',NS)
    for row,name in enumerate(labels):
        label = next(t for t in texts if float(t.get('x'))==258 and
                     ''.join(''.join(t.itertext()).split())==''.join(name.split()))
        size,y = float(label.get('font-size')),float(label.get('y'))
        top,bottom = y-size,y+(len(label.findall('svg:tspan',NS))-1)*size*1.18+size*.3
        cell = cells[row*columns]
        assert float(cell.get('y')) <= top < bottom <= float(cell.get('y'))+float(cell.get('height'))
    height = float(root.get('viewBox').split()[3])
    for label in (t for t in texts if t.get('text-anchor')=='middle' and len(t.findall('svg:tspan',NS))>1):
        size,y = float(label.get('font-size')),float(label.get('y'))
        assert y+(len(label.findall('svg:tspan',NS))-1)*size*1.18+size*.3 < height-45


def test_matplotlib_profile_is_explicit_and_does_not_modify_user_preferences():
    import matplotlib
    from riima.report_plot_style import matplotlib_style
    from riima.visual_style import DOCUMENT, FONT_FAMILIES, PLOT_FONT_PT

    with matplotlib.rc_context({'backend': 'agg', 'font.family': ['serif'], 'axes.facecolor': '#ff0000'}):
        before = matplotlib.rcParams.copy()
        profile = matplotlib_style()
        assert matplotlib.rcParams == before
        with matplotlib.rc_context(profile):
            assert matplotlib.rcParams['font.family'] == ['sans-serif']
            assert matplotlib.rcParams['font.sans-serif'] == list(FONT_FAMILIES)
            assert matplotlib.rcParams['font.size'] == PLOT_FONT_PT['body']
            assert matplotlib.rcParams['axes.facecolor'] == DOCUMENT['surface']
            assert matplotlib.rcParams['text.parse_math'] is False
        assert matplotlib.rcParams == before


def test_svg_uses_shared_font_stack_and_explicit_point_conversion():
    from riima.report_svg import _SVG, _Plot
    from riima.visual_style import DOCUMENT, PLOT_FONT_PT, pt_to_svg, svg_font_family

    svg = _SVG('Título')
    _Plot(svg).axes('Eixo', 'Valor')
    _, root, _ = decode(svg.save('sample'))
    group = root.find('svg:g', NS)
    assert group.get('font-family') == svg_font_family()
    title = root.find('.//svg:text[@class="chart-title"]', NS)
    assert float(title.get('font-size')) == pt_to_svg(PLOT_FONT_PT['title'])
    assert title.get('fill') == DOCUMENT['text']
    for tick in root.findall('.//svg:text[@class="axis-tick"]', NS):
        assert float(tick.get('font-size')) == pt_to_svg(PLOT_FONT_PT['tick'])


def test_svg_long_legend_names_remain_bounded_and_keep_group_colors():
    from riima.report_svg import make_svg_charts

    data = chart_data()
    for group in data['groups']:
        group['label'] = 'Nome de grupo longo ' + 'W' * 70 + '\n' + 'textura e bordas ' * 9
    for key in ('fig1b', 'fig7', 'fig10'):
        _, root, _ = decode(make_svg_charts(data, COLORS)[key][0])
        width, height = map(float, root.get('viewBox').split()[2:])
        previous_bottom = 0
        texts = root.findall('.//svg:text[@class="scatter-legend"]', NS)
        assert len(texts) == len(data['groups'])
        for text in texts:
            x, y, size = (float(text.get(k)) for k in ('x', 'y', 'font-size'))
            lines = text.findall('svg:tspan', NS)
            bottom = y + (len(lines)-1)*size*1.18 + size*.3
            assert 0 < x < width and previous_bottom + 5 < y < bottom < height-20
            previous_bottom = bottom
        points = root.findall('.//svg:circle[@class="scatter-point"]', NS)
        if key == 'fig7':
            points = root.findall('.//svg:circle[@class="flinn-point"]', NS)
        assert [point.get('fill') for point in points] == list(COLORS.values())


def test_svg_wrapped_title_reserves_space_above_the_plot():
    from riima.report_svg import _SVG, _Plot

    title = 'Roseta: ' + 'W'*180 + '\n' + 'Nome completo do grupo ' * 8
    svg = _SVG(title, width=820, height=720)
    _Plot(svg).axes('Eixo', 'Valor')
    _, root, _ = decode(svg.save('fig5'))
    heading = root.find('.//svg:text[@class="chart-title"]', NS)
    lines = heading.findall('svg:tspan', NS)
    size = float(heading.get('font-size'))
    last_baseline = float(heading.get('y')) + (len(lines)-1)*size*1.18
    plot_group = root.find('.//svg:g[@class="chart-body"]', NS)
    offset = float(plot_group.get('transform').removeprefix('translate(0 ').removesuffix(')'))
    assert last_baseline + size*.3 < 70 + offset
    assert float(root.get('viewBox').split()[3]) == pytest.approx(720 + offset)
    assert ''.join(title.split()) == ''.join(''.join(heading.itertext()).split())


def test_shared_titles_and_quantitative_svg_profiles_keep_their_original_values():
    from riima.report_plot_style import CHART_TITLES
    from riima.report_svg import make_svg_charts, _diverging

    charts = make_svg_charts(chart_data(), COLORS)
    for key, pages in charts.items():
        _, root, _ = decode(pages[0])
        title = root.find('.//svg:text[@class="chart-title"]', NS)
        assert ''.join(title.itertext()).startswith(CHART_TITLES[key])
    assert [_diverging(v, -2.7, 2.7) for v in (-2.7, 0, 2.7, np.nan)] == [
        '#2166ac', '#f7f7f7', '#b2182b', '#e5e7eb']
    _, _, heat = decode(charts['fig3b'][0])
    np.testing.assert_equal(np.asarray(heat['values'], float), chart_data()['z'][:, [7, 0, 2, 5, 6]])
    _, _, flinn = decode(charts['fig7'][0])
    np.testing.assert_allclose(flinn['limits'], [[6.4, 24], [.512, 26.4]])


def test_chart_rendering_restores_user_rcparams_even_when_saving_fails(monkeypatch):
    import matplotlib
    from matplotlib.figure import Figure
    from riima.report_charts import make_charts

    def fail(*args, **kwargs):
        raise RuntimeError('save failed')
    monkeypatch.setattr(Figure, 'savefig', fail)
    with matplotlib.rc_context({'backend': 'agg', 'font.family': ['serif'], 'text.color': '#aabbcc'}):
        before = matplotlib.rcParams.copy()
        with pytest.raises(RuntimeError, match='save failed'):
            make_charts(chart_data(1), COLORS)
        assert matplotlib.rcParams == before


def test_matplotlib_uses_one_shared_global_title_and_keeps_legacy_heatmap_profiles(monkeypatch):
    import matplotlib
    from matplotlib.figure import Figure
    from matplotlib.colors import to_rgba
    from riima.report_charts import make_charts
    from riima.report_plot_style import CHART_TITLES
    from riima.visual_style import DOCUMENT, FONT_FAMILIES, PLOT_FONT_PT

    titles, heatmaps = [], []
    def inspect(fig, *args, **kwargs):
        title = fig._suptitle
        titles.append(title.get_text())
        assert title.get_fontsize() == PLOT_FONT_PT['title']
        assert title.get_fontfamily() == ['sans-serif']
        assert matplotlib.rcParams['font.sans-serif'] == list(FONT_FAMILIES)
        assert fig.get_facecolor() == to_rgba(DOCUMENT['surface'])
        assert all(ax.get_title() != title.get_text() for ax in fig.axes)
        for ax in fig.axes:
            for image in ax.images:
                heatmaps.append((image.get_cmap().name, image.get_clim()))
    monkeypatch.setattr(Figure, 'savefig', inspect)
    make_charts(chart_data(), COLORS)
    assert set(titles) == set(CHART_TITLES.values())
    assert heatmaps == [('RdBu_r', (-2.7, 2.7)), ('RdBu_r', (0, 1))]


def _contrast(foreground, background):
    """Independent WCAG relative luminance check of rendered RGB colors."""
    from matplotlib.colors import to_rgb
    def luminance(color):
        channels = np.asarray(to_rgb(color))
        linear = np.where(channels <= .04045, channels/12.92, ((channels+.055)/1.055)**2.4)
        return float(linear @ [.2126, .7152, .0722])
    lo, hi = sorted((luminance(foreground), luminance(background)))
    return (hi+.05)/(lo+.05)


@pytest.mark.parametrize('limits', [(-2.7, 2.7), (0, 1)])
def test_svg_heatmap_annotation_contrast_uses_each_original_cell_color(limits):
    from riima.report_svg import _heatmap, _diverging

    lo, hi = limits
    fractions = np.unique(np.r_[np.linspace(0, 1, 201), .1788, .1836, .8477, np.nan])
    values = (lo+(hi-lo)*fractions)[None, :]
    svg = _heatmap('Contraste', values, ['Grupo'], [str(i) for i in range(values.size)],
                   ['#098842'], lo, hi, 'Escala')
    _, root, _ = decode(svg.save('contrast'))
    cells = root.findall('.//svg:rect[@class="heatmap-cell"]', NS)
    annotations = root.findall('.//svg:text[@class="heatmap-value"]', NS)
    assert len(cells) == len(annotations) == values.size
    for cell, annotation, value in zip(cells, annotations, values[0]):
        assert cell.get('fill') == _diverging(value, lo, hi)
        assert _contrast(annotation.get('fill'), cell.get('fill')) >= 4.5


def test_matplotlib_heatmap_annotation_contrast_preserves_original_cmap_and_ranges(monkeypatch):
    from matplotlib.figure import Figure
    from riima.report_charts import make_charts

    data = chart_data()
    data['z'][0, 7] = -1.70856
    data['z'][1, 7] = 1.73448
    data['similarity'][0, 1] = .8477
    data['similarity'][1, 0] = .1788
    maps = []
    def inspect(fig, *args, **kwargs):
        for ax in fig.axes:
            for image in ax.images:
                maps.append((image.get_cmap().name, image.get_clim()))
                for text in ax.texts:
                    j, i = map(int, text.get_position())
                    value = np.ma.filled(image.get_array(), np.nan)[i, j]
                    background = image.cmap(image.norm(value))
                    assert _contrast(text.get_color(), background) >= 4.5
    monkeypatch.setattr(Figure, 'savefig', inspect)
    make_charts(data, COLORS)
    assert maps == [('RdBu_r', (-2.7, 2.7)), ('RdBu_r', (0, 1))]
