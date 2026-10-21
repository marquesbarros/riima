"""Readable report scales and visual hierarchy survive figure rendering."""
import numpy as np
import pytest

from riima import metrics
from test_riima_review import prepared_dataset


@pytest.fixture(scope='module')
def rendered_chart_axes(tmp_path_factory):
    from matplotlib.colors import to_rgba
    from matplotlib.container import ErrorbarContainer
    from matplotlib.figure import Figure
    from riima.report_charts import make_charts
    from riima.report_data import load_metrics

    prepared = prepared_dataset(tmp_path_factory.mktemp('report_scales'), count=1)
    data = load_metrics(metrics.export_metrics(prepared))
    colors = {0: '#098842', 1: '#d3256f'}
    for i, group in enumerate(data['groups']):
        group['label'] = f"Grupo {group['id']}"
        regions = data['regions'][group['id']]
        # A concentrated distribution, including a weakly anisotropic group.
        regions['orientation_deg'] = np.full(4, 35.)
        regions['area_pixels'] = np.full(4, 10000. if i == 0 else 9.)
        regions['anisotropy'] = np.array([.5, .75, .25, .35]) if i == 0 else np.full(4, .01)
        regions['major_axis_px'] = np.array([10., 20., 40., 50.])
        regions['minor_axis_px'] = np.array([2., 5., 8., 12.])

    result = {'roses': [], 'flinn': None}
    savefig = Figure.savefig

    def capture(fig, *args, **kwargs):
        savefig(fig, *args, **kwargs)
        if fig.axes[0].name == 'polar':
            for ax in fig.axes:
                result['roses'].append({
                    'circles': ax.get_yticks(),
                    'labels': [tick.get_text() for tick in ax.get_yticklabels()],
                    'limit': ax.get_ylim()[1],
                    'peak': max(bar.get_height() for bar in ax.patches),
                })
        elif fig.axes[0].get_xlabel() == 'Eixo maior (px)':
            ax = fig.axes[0]
            errors = []
            for container in ax.containers:
                if isinstance(container, ErrorbarContainer):
                    point, caps, bars = container.lines
                    errors.append({
                        'point_color': to_rgba(point.get_color(), point.get_alpha()),
                        'cap_colors': [to_rgba(cap.get_markeredgecolor(), cap.get_alpha()) for cap in caps],
                        'bar_colors': [bar.get_colors() for bar in bars],
                        'point_order': point.get_zorder(),
                        'bar_orders': [bar.get_zorder() for bar in bars],
                    })
            result['flinn'] = {
                'axis_names': (ax.get_xlabel(), ax.get_ylabel()),
                'ticks': [[tick.get_text() for tick in axis.get_ticklabels(which='both')]
                          for axis in (ax.xaxis, ax.yaxis)],
                'errors': errors,
            }

    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(Figure, 'savefig', capture)
        make_charts(data, colors)
    return result


def test_rosettes_label_every_second_circle_with_round_values(rendered_chart_axes):
    for rose in rendered_chart_axes['roses']:
        circles, labels = rose['circles'], rose['labels']
        assert len(circles) >= 2 and len(circles) % 2 == 0
        assert labels[::2] == [''] * (len(labels) // 2)
        numbers = np.array([float(text) for text in labels[1::2]])
        assert np.all(numbers >= 5) and np.all(numbers % 5 == 0)
        np.testing.assert_allclose(numbers, circles[1::2])
        np.testing.assert_allclose(np.diff(circles), circles[0])
        assert rose['limit'] >= rose['peak']


def test_flinn_log_axis_ticks_are_readable_numbers(rendered_chart_axes):
    flinn = rendered_chart_axes['flinn']
    assert flinn['axis_names'] == ('Eixo maior (px)', 'Eixo menor (px)')
    for axis in flinn['ticks']:
        labels = [text for text in axis if text]
        assert labels
        assert all('$' not in text and '\\' not in text for text in labels)
        assert all(float(text) > 0 for text in labels)


def test_flinn_interquartile_bars_are_transparent_below_opaque_group_points(rendered_chart_axes):
    from matplotlib.colors import to_rgba
    errors = rendered_chart_axes['flinn']['errors']
    assert len(errors) == 2
    for error, color in zip(errors, ('#098842', '#d3256f')):
        np.testing.assert_allclose(error['point_color'], to_rgba(color))
        assert all(0 < rgba[3] < 1 for rgba in error['cap_colors'])
        assert all(np.all((rgba[:, 3] > 0) & (rgba[:, 3] < 1)) for rgba in error['bar_colors'])
        assert all(error['point_order'] > order for order in error['bar_orders'])
