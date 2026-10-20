"""Numeric scales and identities remain readable across report renderers."""
import copy
import warnings

import numpy as np
import pytest

from test_riima_report_svg import chart_data, decode, NS, KEYS


COLORS = {0: '#098842', 1: '#d3256f', 2: '#2171b5'}


@pytest.mark.parametrize('value, label', [(0., '0'), (-0., '0'), (.1, '0.1'), (1.25, '1.25'),
                                       (-2., '-2'), (1e-8, '1e-8'), (-1e-8, '-1e-8'), (1e6, '1e+6')])
def test_numeric_labels_use_plain_decimals_or_ascii_scientific_notation(value, label):
    from riima.report_plot_style import axis_number
    assert axis_number(value) == label
    assert float(label) == value


def extreme_data():
    data = chart_data(3)
    for group in data['groups']:
        group['label'] = 'Mesmo nome ' + 'W' * 40 + '\n' + 'bordas e textura ' * 12
        regions = data['regions'][group['id']]
        regions['major_axis_px'] = np.full(5, 3.3)
        regions['minor_axis_px'] = np.full(5, .33)
    return data


@pytest.mark.parametrize('limits', [(2.64, 3.96), (.264, .396), (1e-8, 1.2e-8),
                                   (1.000001, 1.000004), (12., 1200.)])
def test_svg_logarithmic_intervals_have_distinct_in_range_numeric_ticks(limits):
    from riima.report_svg import _Plot, _SVG, _fmt
    plot = _Plot(_SVG('Escala'), limits, limits, log=True)
    ticks = np.asarray(plot._ticks(limits))
    assert 2 <= len(ticks) <= 8
    assert np.all((ticks >= limits[0]) & (ticks <= limits[1]))
    labels = [_fmt(value) for value in ticks]
    assert len(set(labels)) == len(labels)
    np.testing.assert_allclose([float(label) for label in labels], ticks, rtol=1e-11)
    assert all('\u2212' not in label for label in labels)


def test_svg_axis_legend_allocates_space_for_every_wrapped_line():
    from riima.report_svg import make_svg_charts
    _, root, _ = decode(make_svg_charts(extreme_data(), COLORS)['fig7'][0])
    height = float(root.get('viewBox').split()[3])
    names = [el for el in root.findall('.//svg:text', NS)
             if ''.join(el.itertext()).startswith('Mesmo nome')]
    assert len(names) == 3
    bottom = 0
    for text in names:
        y, size = float(text.get('y')), float(text.get('font-size'))
        assert y > bottom + 5
        bottom = y + (len(text.findall('svg:tspan', NS))-1) * size * 1.18 + size*.3
        assert bottom < height-20


@pytest.fixture(scope='module')
def extreme_rendering():
    from matplotlib.figure import Figure
    from matplotlib.collections import PathCollection
    from riima.report_charts import make_charts
    data, captured = extreme_data(), {}
    original = Figure.savefig
    def save(fig, *args, **kwargs):
        original(fig, *args, **kwargs)
        fig.canvas.draw()
        renderer = fig.canvas.get_renderer()
        for ax in fig.axes:
            xlabel = ax.get_xlabel()
            if xlabel not in ('Número de regiões medidas', 'Razão axial mediana', 'Eixo maior (px)'):
                continue
            legend = ax.get_legend()
            captured[xlabel] = dict(
                points=[(c.get_offsets().copy(), c.get_facecolors().copy())
                        for c in ax.collections if isinstance(c, PathCollection)],
                legend=[text.get_text() for text in legend.get_texts()] if legend else [],
                legend_boxes=[text.get_window_extent(renderer).bounds for text in legend.get_texts()] if legend else [],
                figure=fig.bbox.bounds, limits=(ax.get_xlim(), ax.get_ylim()),
                scales=(ax.get_xscale(), ax.get_yscale()),
                categories=[tick.get_text() for tick in fig.axes[0].get_xticklabels()]
                           if xlabel == 'Razão axial mediana' else [],
                category_limits=(fig.axes[0].get_xlim(), fig.axes[0].get_ylim()),
                annotations=[text.get_text() for text in ax.texts],
                annotation_boxes=[text.get_window_extent(renderer).bounds for text in ax.texts],
                plot_box=ax.get_window_extent(renderer).bounds,
                ticks=[[(float(value), text.get_text()) for value, text in
                        zip(axis.get_majorticklocs(), axis.get_majorticklabels())]
                       for axis in (ax.xaxis, ax.yaxis)])
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(Figure, 'savefig', save)
        make_charts(data, COLORS)
    return data, captured


@pytest.mark.parametrize('axis', ['Número de regiões medidas', 'Razão axial mediana'])
def test_matplotlib_coincident_points_use_bounded_legends_with_unique_ids(extreme_rendering, axis):
    data, captured = extreme_rendering
    plot = captured[axis]
    assert len(plot['legend']) == 3
    for group, label in zip(data['groups'], plot['legend']):
        assert f"[{group['id']}]" in label
        assert ''.join(group['label'].split()) in ''.join(label.split())
    previous_bottom = float('inf')
    _, _, width, height = plot['figure']
    for x, y, w, h in plot['legend_boxes']:
        assert 0 <= x < x+w <= width
        assert 0 <= y < y+h <= height
        assert y+h <= previous_bottom + 1
        previous_bottom = y
    assert len(plot['points']) == 3
    assert len({tuple(point[0][0]) for point in plot['points']}) == 1
    assert '0, 1, 2' in plot['annotations']


def test_shape_bar_categories_use_short_ids_linked_to_the_full_legend(extreme_rendering):
    from riima.report_svg import make_svg_charts
    data, captured = extreme_rendering
    expected = [f"[{group['id']}]" for group in data['groups']]
    assert captured['Razão axial mediana']['categories'] == expected
    _, root, metadata = decode(make_svg_charts(data, COLORS)['fig10'][0])
    labels = root.findall('.//svg:text[@class="axis-tick"][@data-axis="x"]', NS)
    assert [''.join(text.itertext()) for text in labels[:3]] == expected
    np.testing.assert_allclose(captured['Razão axial mediana']['category_limits'], metadata['compactness_limits'])


def test_linear_numeric_axes_share_round_tick_positions(extreme_rendering):
    from riima.report_svg import make_svg_charts
    data, captured = extreme_rendering
    _, root, _ = decode(make_svg_charts(data, COLORS)['fig1b'][0])
    plot = captured['Número de regiões medidas']
    for name, bounds, ticks in zip(('x', 'y'), plot['limits'], plot['ticks']):
        visible = [value for value, _ in ticks if bounds[0] <= value <= bounds[1]]
        labels = root.findall(f'.//svg:text[@class="axis-tick"][@data-axis="{name}"]', NS)
        np.testing.assert_allclose([float(text.get('data-value')) for text in labels], visible)


def test_many_coincident_ids_remain_inside_the_plot_away_from_the_legend(monkeypatch):
    from matplotlib.figure import Figure
    from riima.report_charts import make_charts
    data = chart_data(12)
    for group in data['groups']:
        group['components_measured'] = 1 if group['id'] == 0 else 100
    colors = {i: list(COLORS.values())[i % 3] for i in range(12)}
    original, boxes = Figure.savefig, []
    def capture(fig, *args, **kwargs):
        original(fig, *args, **kwargs)
        ax = fig.axes[0]
        if ax.get_xlabel() == 'Número de regiões medidas':
            fig.canvas.draw()
            renderer = fig.canvas.get_renderer()
            boxes.extend((text.get_window_extent(renderer), ax.get_window_extent(renderer))
                         for text in ax.texts)
    monkeypatch.setattr(Figure, 'savefig', capture)
    make_charts(data, colors)
    assert len(boxes) == 2
    for text, plot in boxes:
        assert plot.x0 <= text.x0 < text.x1 <= plot.x1
        assert plot.y0 <= text.y0 < text.y1 <= plot.y1


@pytest.mark.parametrize('key, axis', [('fig1b', 'Número de regiões medidas'),
                                     ('fig10', 'Razão axial mediana')])
def test_backends_preserve_scatter_values_colors_and_ranges(extreme_rendering, key, axis):
    from matplotlib.colors import to_rgba
    from riima.report_svg import make_svg_charts
    data, captured = extreme_rendering
    _, root, metadata = decode(make_svg_charts(data, COLORS)[key][0])
    points = root.findall('.//svg:circle[@class="scatter-point"]', NS)
    assert len(points) == len(captured[axis]['points']) == 3
    for point, (offsets, rgba), group in zip(points, captured[axis]['points'], data['groups']):
        assert point.get('fill') == COLORS[group['id']]
        np.testing.assert_allclose(rgba[0, :3], to_rgba(COLORS[group['id']])[:3])
        row = (metadata['groups'] if key == 'fig1b' else metadata['shape'])[group['id']]
        expected = (row['components'], row['median_area']) if key == 'fig1b' else (
            row['median_aspect'], row['median_eccentricity'])
        np.testing.assert_allclose(offsets[0], expected)
    annotations = root.findall('.//svg:text[@class="scatter-id"]', NS)
    assert [''.join(text.itertext()) for text in annotations] == ['0, 1, 2']
    np.testing.assert_allclose(captured[axis]['limits'], metadata['limits'])


def test_axis_scales_and_tick_values_match_between_backends(extreme_rendering):
    from riima.report_svg import make_svg_charts
    data, captured = extreme_rendering
    _, root, metadata = decode(make_svg_charts(data, COLORS)['fig7'][0])
    plot = captured['Eixo maior (px)']
    assert plot['scales'] == ('log', 'log')
    left, _, width, _ = plot['plot_box']
    assert all(left <= x and x+w <= left+width+1 for x, _, w, _ in plot['annotation_boxes'])
    for limits, ticks in zip(plot['limits'], plot['ticks']):
        visible = [(value, label) for value, label in ticks if limits[0] <= value <= limits[1] and label]
        assert len(visible) >= 2
        assert len({label for _, label in visible}) == len(visible)
        np.testing.assert_allclose([float(label) for _, label in visible],
                                   [value for value, _ in visible], rtol=1e-11)
    np.testing.assert_allclose(plot['limits'], metadata['limits'])
    for name, actual, expected in zip(('x', 'y'), plot['ticks'], metadata['major_ticks']):
        np.testing.assert_allclose([value for value, _ in actual], expected)
        labels = root.findall(f'.//svg:text[@class="axis-tick"][@data-axis="{name}"]', NS)
        np.testing.assert_allclose([float(''.join(text.itertext())) for text in labels], expected)
        assert root.findall(f'.//svg:line[@class="minor-tick"][@data-axis="{name}"]', NS)


@pytest.mark.parametrize('count, empty', [(0, True), (1, True), (1, False)])
def test_missing_groups_and_undefined_orientation_do_not_invent_data(count, empty):
    from riima.report_charts import make_charts
    from riima.report_svg import make_svg_charts
    data = chart_data(count, empty)
    for group in data['groups']:
        group['dominant_direction_deg'] = np.nan
        data['regions'][group['id']]['orientation_deg'][:] = np.nan
    before = copy.deepcopy(data)
    with warnings.catch_warnings(record=True) as recorded:
        warnings.simplefilter('always')
        rendered = (make_charts(data, COLORS), make_svg_charts(data, COLORS))
    assert not [warning for warning in recorded if 'singular' in str(warning.message)]
    for charts in rendered:
        assert set(charts) == KEYS and all(charts.values())
    for group in data['groups']:
        for field, values in data['regions'][group['id']].items():
            np.testing.assert_array_equal(values, before['regions'][group['id']][field])
    np.testing.assert_equal(data, before)
