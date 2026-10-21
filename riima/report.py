"""Self-contained HTML export of the current RIIMA analysis and descriptive metrics."""
from __future__ import annotations

import base64
from collections import Counter
from html import escape
import io
import os
from pathlib import Path
from string import Template
import tempfile

import numpy as np
from PIL import Image

from . import metrics
from .report_data import load_metrics
from .exports import export_operation, staged_directory, staged_file
from .visual_style import document_css


def _format(value, decimals=2, *, missing='Não definida'):
    if value is None or not np.isfinite(value):
        return missing
    return f'{value:.{decimals}f}'.replace('.', ',')


def _classification(prepared, data):
    state = prepared['review']
    legend = ''.join(
        f'<span data-group-id="{g["id"]}"><i class="swatch" style="background:{g["hex"]}"></i>'
        f'{escape(g["label"])} ({_format(g["area_percent"])}%)</span>' for g in data['groups'])
    figures = []
    for item in state.items:
        labels, valid = item['labels'], item['display_valid']
        assigned = valid & (labels >= 0)
        rgb = np.full((*labels.shape, 3), 30, np.uint8)
        rgb[assigned] = state.palette[labels[assigned]]
        image = Image.fromarray(rgb)
        original_size = image.size
        image.thumbnail((1600, 1600), Image.Resampling.NEAREST)
        buffer = io.BytesIO()
        image.save(buffer, format='PNG', compress_level=3, optimize=False)
        encoded = base64.b64encode(buffer.getvalue()).decode('ascii')
        caption = escape(item['name'])
        reduced = ' Prévia reduzida; as métricas usam a resolução original.' if image.size != original_size else ''
        figures.append(f'<p>{caption}: {labels.shape[1]} × {labels.shape[0]} pixels.{reduced}</p>'
                       f'<img class="classification-map" alt="Classificação final de {caption}" '
                       f'src="data:image/png;base64,{encoded}">')
    names = (state.items[0]['optical']['image_names'] if state.mode == 'registered' else
             [item['name'] for item in state.items])
    approved = sum(g['approved'] for g in data['groups'] if not g['background'])
    return ('<section id="classificacao" class="figure-card"><h2>Classificação analisada</h2>'
            '<p>Máscara inteira da classificação atual, com os nomes e as cores definidos no RIIMA. '
            'Regiões excluídas e pixels sem classe aparecem em cinza escuro.</p>'
            f'<p class="analysis-info">Imagens: {escape(", ".join(names))}. '
            f'Grupos aprovados: {approved}. Pixels válidos: {data["valid_pixels"]}. '
            f'Pixels sem classe: {max(0, data["valid_pixels"] - sum(g["pixels"] for g in data["groups"]))}.</p>'
            + ''.join(figures) + '<div class="legend">' + legend + '</div></section>')


def _summary_table(groups):
    headings = ['Grupo', 'Área (%)', 'Regiões medidas', 'Área mediana (px)',
                'Anisotropia média', 'Direção dominante (°)', 'Coerência direcional']
    rows = []
    for g in groups:
        values = [escape(g['label']), _format(g['area_percent']), str(g['components_measured']),
                  _format(g['component_area_median_px'], 1, missing='Sem dados'),
                  _format(g['mean_anisotropy'], 3, missing='Sem dados'),
                  _format(g['dominant_direction_deg'], 1), _format(g['direction_coherence'], 3)]
        values[0] = f'<i class="swatch" style="background:{g["hex"]}"></i> ' + values[0]
        rows.append(f'<tr data-group-id="{g["id"]}">' + ''.join(f'<td>{v}</td>' for v in values) + '</tr>')
    return ('<div class="table-scroll"><table><thead><tr>'
            + ''.join(f'<th>{h}</th>' for h in headings) + '</tr></thead><tbody>'
            + ''.join(rows) + '</tbody></table></div>')


def build_report(prepared, directory, *, stage_callback=None):
    """Compose fresh CSVs and current masks; no training, selection or display changes."""
    from .report_charts import make_charts
    data = load_metrics(directory)
    groups = data['groups']
    duplicates = Counter(g['name'] for g in groups)
    palette = prepared['review'].palette
    for g in groups:
        g['label'] = g['name'] + (f' (ID {g["id"]})' if duplicates[g['name']] > 1 else '')
        g['hex'] = '#' + ''.join(f'{int(v):02x}' for v in palette[g['id']])
    colors = {g['id']: palette[g['id']].astype(float) / 255 for g in groups}
    if stage_callback:
        stage_callback('Gerando os gráficos do relatório descritivo...')
    charts = make_charts(data, colors)
    values = {'visual_style': document_css()}
    for key, pages in charts.items():
        title = 'Gráfico ' + key[3:].upper()
        values[key] = ''.join(f'<img alt="{title}, página {i + 1}" '
                              f'src="data:{getattr(page, "mime_type", "image/png")};base64,{page}">'
                              for i, page in enumerate(pages))
    largest = sorted(groups, key=lambda g: -g['area_percent'])[:3]
    direction_groups = [g for g in groups if g['components_measured'] and np.isfinite(g['dominant_direction_deg'])]
    highest = max(direction_groups, key=lambda g: g['direction_coherence']) if direction_groups else None
    aniso_groups = [g for g in groups if np.isfinite(g['mean_anisotropy'])]
    lowest = min(aniso_groups, key=lambda g: g['mean_anisotropy']) if aniso_groups else None
    active = [groups[i] for i in data['active']]
    pairs = [(float(data['similarity'][i, j]), active[i], active[j])
             for i in range(len(active)) for j in range(i + 1, len(active))]
    pairs.sort(key=lambda p: -p[0])

    def pair_list(items):
        return '; '.join(f'{escape(a["label"])} / {escape(b["label"])} ({_format(s)})' for s, a, b in items)

    descriptor_rows = []
    for g in groups:
        descriptors = data['descriptors'][g['id']][:3]
        text = '; '.join(f'{escape(f)} ({"elevado" if v > 0 else "reduzido"}, z={_format(v, 1)})'
                         for f, v in descriptors) if descriptors else 'Sem contraste variável entre grupos'
        descriptor_rows.append(f'<tr><td>{escape(g["label"])}</td><td>{text}</td></tr>')
    info = data['info']
    values.update(area3=_format(sum(g['area_percent'] for g in largest), 1),
                  largest_groups=', '.join(f'{escape(g["label"])} ({_format(g["area_percent"], 1)}%)' for g in largest),
                  highest_coh_group_name=escape(highest['label']) if highest else 'Não definida',
                  highest_coh_direction_coherence=_format(highest['direction_coherence'], 3) if highest else 'Não definida',
                  highest_coh_dominant_direction_deg=_format(highest['dominant_direction_deg'], 1) if highest else 'Não definida',
                  lowest_aniso_group_name=escape(lowest['label']) if lowest else 'Sem dados',
                  lowest_aniso_mean_anisotropy=_format(lowest['mean_anisotropy'], 3) if lowest else 'Sem dados',
                  highest_sim_1=escape(pairs[0][1]['label']) if pairs else 'Sem pares',
                  highest_sim_2=escape(pairs[0][2]['label']) if pairs else 'para comparar',
                  highest_sim_0=_format(pairs[0][0]) if pairs else 'Não definida',
                  most_sim=pair_list(pairs[:5]) if pairs else 'Sem pares para comparar',
                  least_sim=pair_list(list(reversed(pairs[-5:]))) if pairs else 'Sem pares para comparar',
                  summary_table=_summary_table(groups), rows_html=''.join(descriptor_rows),
                  classification=_classification(prepared, data),
                  min_component_area_px=info['min_component_area_px'], connectivity=info['connectivity'],
                  max_measured_regions_per_group_image=info['max_measured_regions_per_group_image'],
                  effective_sample_per_group=info['effective_sample_per_group'])
    omitted = sum(g['components_omitted'] for g in groups)
    small = sum(g['components_below_min_area'] for g in groups)
    chart_renderer = ('SVG vetorial' if any(getattr(page, 'mime_type', '') == 'image/svg+xml'
                                           for pages in charts.values() for page in pages)
                      else 'Matplotlib')
    values['calculation_notes'] = (
        f'<p>Regiões abaixo da área mínima: {small}. Regiões acima do limite de medição: {omitted}. '
        'Contagens de pixels e áreas incluem toda a classificação; formas e distribuições usam somente as regiões medidas. '
        'As unidades são pixels e pixels de área, sem calibração física. Anisotropia é geométrica; regiões conectadas '
        'não correspondem necessariamente a grãos minerais. O diagrama de eixos não estima deformação tridimensional.</p>'
        '<p>Direções em coordenadas da imagem: 0° para a direita, 90° para baixo, orientação axial de 0° a 180°. '
        'Direções isotrópicas ou sem resultante definida permanecem sem valor. Grupos sem pixels são mostrados nas tabelas '
        'e excluídos do cálculo de similaridade e padronização. Atributos constantes não geram descritores; ausências '
        'na similaridade são preenchidas pela média antes da padronização. O mapa de calor mostra até 16 atributos ópticos mais representativos '
        'e quatro métricas geométricas. Os CSVs preservam todos os atributos.</p>'
        '<p>Similaridade usa todos os descritores variáveis do perfil; η² descreve a amostra de atributos ópticos, '
        'sem indicar importância do modelo ou identidade mineralógica. Densidades usam banda de Scott; não são desenhadas '
        'para amostras singulares. Rosetas usam anisotropia × √área; coerência e direção usam área × anisotropia.</p>'
        '<p>Métodos de referência: '
        '<a href="https://docs.scipy.org/doc/scipy/reference/generated/scipy.cluster.hierarchy.linkage.html">Ward</a> '
        'e <a href="https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.gaussian_kde.html">densidade gaussiana e banda de Scott</a>. '
        f'Cálculos adaptados em NumPy; gráficos em {chart_renderer}.</p>'
        '<p>Métricas recalculadas a partir do estado aplicado desta análise.</p>')
    template = (Path(__file__).parent / 'templates' / 'descriptive_report.html').read_text(encoding='utf-8')
    return Template(template).substitute(values)


@export_operation
def export_report(prepared, filename=None, *, stage_callback=None):
    """Refresh metrics and atomically export one portable HTML, without changing analysis."""
    if prepared['review'].has_pending:
        raise ValueError('Atualize a classificação antes de exportar o relatório.')
    if stage_callback:
        stage_callback('Calculando as métricas atuais para o relatório...')
    staged_directory(prepared['output'])
    directory = metrics.export_metrics(prepared)
    html = build_report(prepared, staged_directory(directory), stage_callback=stage_callback)
    target = Path(filename or Path(prepared['output']) / 'relatorio_analise_descritiva.html')
    target.parent.mkdir(parents=True, exist_ok=True)
    destination = staged_file(target)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode='w', encoding='utf-8', dir=destination.parent,
                                         prefix='.riima-report-', suffix='.tmp', delete=False) as handle:
            temporary = Path(handle.name)
            handle.write(html)
        os.replace(temporary, destination)
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()
    return target
