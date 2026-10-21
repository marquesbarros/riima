"""The portable report renderer retains the numerical definitions of each chart."""
import base64
import builtins
import json
import xml.etree.ElementTree as ET

import numpy as np
import pytest

from riima.report_data import gaussian_density, ward_linkage


KEYS = {'fig1a', 'fig1b', 'fig1c', 'fig2', 'fig3a', 'fig3b', 'fig4a',
        'fig4b', 'fig5', 'fig6', 'fig7', 'fig8', 'fig9', 'fig10'}
NS = {'svg': 'http://www.w3.org/2000/svg'}


def chart_data(count=3, empty=False):
    groups, regions = [], {}
    for i in range(count):
        area = np.array([] if empty else [10., 30., 120., 600., 2300.])
        n = len(area)
        regions[i] = dict(area_pixels=area,
                          major_axis_px=np.array([5., 8., 12., 20., 30.])[:n],
                          minor_axis_px=np.array([3., 4., 6., 8., 9.])[:n],
                          aspect_ratio=np.array([5/3, 2, 2, 2.5, 30/9])[:n],
                          eccentricity=np.array([.3, .4, .5, .7, .9])[:n],
                          anisotropy=np.array([0., .2, .4, .6, .8])[:n],
                          orientation_deg=np.array([np.nan, 0., 10., 90., 180.])[:n],
                          compactness_grid=np.array([.2, .4, .5, .7, .8])[:n])
        groups.append(dict(id=i, label=f'Grupo {i + 1}', area_percent=100/max(1, count),
                           components_measured=n,
                           component_area_median_px=float(np.median(area)) if n else np.nan,
                           direction_coherence=.7 if n else 0.,
                           dominant_direction_deg=20. if n else np.nan))
    columns = ['Área (%)', 'Regiões medidas', 'Área mediana (px)', 'Área Q75 (px)',
               'Excentricidade média', 'Anisotropia média', 'Coerência direcional',
               'Cor & textura', 'Direção (cos 2θ)', 'Direção (sin 2θ)']
    z = np.arange(count*len(columns), dtype=float).reshape(count, len(columns)) / 10
    if count:
        z[0, 5] = np.nan
    linkage, order = ward_linkage(z)
    return dict(groups=groups, regions=regions, features=['Cor & textura'], columns=columns,
                descriptors={i: [('Cor & textura', -.7), ('Área (%)', .8)] for i in range(count)},
                ranks=[{'feature': 'Cor & textura', 'separation_eta2': '.75'}], z=z,
                active=list(range(count)), order=order, linkage=linkage,
                similarity=np.eye(count))


def decode(chart):
    raw = base64.b64decode(chart)
    root = ET.fromstring(raw)
    metadata = root.find('svg:metadata', NS)
    return raw.decode('utf-8'), root, json.loads(metadata.text)


@pytest.mark.parametrize(('count', 'empty'), [(3, False), (1, True), (0, True)])
def test_svg_all_sections_are_valid_self_contained_and_do_not_import_matplotlib(monkeypatch, count, empty):
    from riima.report_svg import make_svg_charts
    original_import = builtins.__import__

    def without_native_plotting(name, *args, **kwargs):
        if name.split('.')[0] in {'matplotlib', 'scipy'}:
            pytest.fail(f'Portable charts imported {name}')
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, '__import__', without_native_plotting)
    data = chart_data(count, empty)
    if count:
        data['groups'][0]['label'] = '<script>alert("nome")</script> & Classe longa ' * 3
    before = {c: {k: v.copy() for k, v in r.items()} for c, r in data['regions'].items()}
    charts = make_svg_charts(data, {i: np.array([9, 136, 66])/255 for i in range(count)})
    assert set(charts) == KEYS
    for values in charts.values():
        assert values
        for chart in values:
            text, root, metadata = decode(chart)
            assert root.tag == '{http://www.w3.org/2000/svg}svg'
            assert root.get('viewBox') and root.get('width') == '100%'
            assert float(root.get('height')) > 200
            assert '<script>' not in text and 'href=' not in text
            assert not root.findall('.//svg:script', NS)
            assert metadata['renderer'] == 'riima-svg'
            for element in root.iter():
                for key in ('x', 'y', 'cx', 'cy', 'r', 'height', 'stroke-width'):
                    if key in element.attrib:
                        assert np.isfinite(float(element.get(key)))
    for c in before:
        for field in before[c]:
            np.testing.assert_array_equal(before[c][field], data['regions'][c][field])


def test_rosette_retains_axial_weighting_and_every_second_round_scale_label():
    from riima.report_svg import make_svg_charts
    data = chart_data(1)
    charts = make_svg_charts(data, {0: '#098842'})
    text, root, rose = decode(charts['fig5'][0])
    r = data['regions'][0]
    keep = np.isfinite(r['orientation_deg'])
    expected, edges = np.histogram(np.deg2rad(r['orientation_deg'][keep]) % np.pi,
                                  np.linspace(0, np.pi, 19),
                                  weights=r['anisotropy'][keep] * np.sqrt(r['area_pixels'][keep]))
    np.testing.assert_allclose(rose['counts'], expected)
    np.testing.assert_allclose(rose['edges_rad'], edges)
    assert len(root.findall('.//svg:path[@class="rose-bin"]', NS)) == 36
    rings = root.findall('.//svg:circle[@class="rose-ring"]', NS)
    labels = root.findall('.//svg:text[@class="rose-radius"]', NS)
    assert len(rings) == 2*len(labels)
    assert all(float(el.text) % 5 == 0 for el in labels)
    assert '#098842' in text


def test_histograms_density_quartiles_heat_and_linkage_are_the_existing_report_values():
    from riima.report_svg import make_svg_charts
    data = chart_data()
    charts = make_svg_charts(data, {i: '#098842' for i in range(3)})
    for key, field, bins in [('fig8', 'area_pixels', 28), ('fig9', 'anisotropy', 24)]:
        _, _, meta = decode(charts[key][0])
        vals = data['regions'][0][field]
        vals = np.log10(vals) if key == 'fig8' else vals
        counts, edges = np.histogram(vals, bins=bins, density=True)
        np.testing.assert_allclose(meta['density'], counts)
        np.testing.assert_allclose(meta['edges'], edges)
        np.testing.assert_allclose(meta['kde'], gaussian_density(vals, np.asarray(meta['kde_x'])))
        assert meta['median'] == pytest.approx(np.median(vals))
    _, axes_root, axes = decode(charts['fig7'][0])
    np.testing.assert_allclose(axes['groups'][0]['major_quartiles'], [8, 12, 20])
    np.testing.assert_allclose(axes['groups'][0]['minor_quartiles'], [4, 6, 8])
    assert 'Eixo maior (px)' in ''.join(axes_root.itertext())
    assert 'Eixo menor (px)' in ''.join(axes_root.itertext())
    assert all(e.get('stroke-opacity') == '0.35'
               for e in axes_root.findall('.//svg:line[@class="iqr"]', NS))
    assert all(e.get('fill-opacity', '1') == '1'
               for e in axes_root.findall('.//svg:circle[@class="flinn-point"]', NS))
    _, _, heat = decode(charts['fig3b'][0])
    np.testing.assert_allclose(np.asarray(heat['values'], float), data['z'][:, [7, 0, 2, 5, 6]], equal_nan=True)
    _, _, tree = decode(charts['fig4a'][0])
    np.testing.assert_allclose(tree['linkage'], data['linkage'])
    _, _, similarity = decode(charts['fig4b'][0])
    np.testing.assert_allclose(similarity['values'], data['similarity'][np.ix_(data['order'], data['order'])])


def test_hexagonal_density_retains_sample_limit_and_axial_coordinates():
    from riima.report_svg import make_svg_charts
    data = chart_data(1)
    _, _, meta = decode(make_svg_charts(data, {0: '#098842'})['fig6'][0])
    r = data['regions'][0]
    keep = np.flatnonzero(np.isfinite(r['orientation_deg']))
    np.testing.assert_allclose(meta['x'], r['anisotropy'][keep] * np.cos(np.deg2rad(2*r['orientation_deg'][keep])))
    np.testing.assert_allclose(meta['y'], r['anisotropy'][keep] * np.sin(np.deg2rad(2*r['orientation_deg'][keep])))
    assert sum(meta['hex_counts']) == len(keep)
    assert meta['gridsize'] == 18
    for field, values in r.items():
        r[field] = np.resize(values, 1600)
    _, _, sampled = decode(make_svg_charts(data, {0: '#098842'})['fig6'][0])
    assert len(sampled['x']) == 1200
    assert sum(sampled['hex_counts']) == 1200


def test_single_constant_distribution_does_not_invent_a_density_curve():
    from riima.report_svg import make_svg_charts
    data = chart_data(1)
    for field in data['regions'][0]:
        data['regions'][0][field] = data['regions'][0][field][1:2]
    charts = make_svg_charts(data, {0: '#098842'})
    for key in ('fig8', 'fig9'):
        _, _, meta = decode(charts[key][0])
        assert meta['kde'] is None
        assert meta['median'] is not None


def test_coincident_scatter_points_keep_each_long_name_in_a_separate_bounded_legend():
    from riima.report_svg import make_svg_charts
    data = chart_data(3)
    for group in data['groups']:
        group['label'] = f"Grupo {group['id']}: " + 'Minerais alongados com bordas e textura pouco definidas ' * 5
    charts = make_svg_charts(data, {0: '#098842', 1: '#d3256f', 2: '#2171b5'})
    for key in ('fig1b', 'fig10'):
        _, root, metadata = decode(charts[key][0])
        legends = root.findall('.//svg:text[@class="scatter-legend"]', NS)
        points = root.findall('.//svg:circle[@class="scatter-point"]', NS)
        assert len(points) == len(legends) == 3
        assert {legend.get('data-group-id') for legend in legends} == {'0', '1', '2'}
        assert len({(point.get('cx'), point.get('cy')) for point in points}) == 1
        assert len({legend.get('y') for legend in legends}) == 3
        assert [''.join(el.itertext()).replace(' ', '') for el in legends] == [
            group['label'].replace(' ', '') for group in data['groups']]
        width, height = [float(v) for v in root.get('viewBox').split()[2:]]
        previous_bottom = 0
        for entry in metadata['legend']:
            assert 0 <= entry['x'] < entry['x'] + entry['width'] <= width
            assert previous_bottom <= entry['y'] < entry['y'] + entry['height'] < height-20
            previous_bottom = entry['y'] + entry['height']
            assert entry['x'] > max(float(point.get('cx')) for point in points)
