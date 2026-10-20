# Estilos visuais

`riima.visual_style` define os papéis de apresentação sem importar Tk,
Matplotlib ou Pillow. `theme.py` adapta esses papéis aos widgets existentes;
`report_plot_style.py` adapta documentos e figuras. O HTML recebe variáveis CSS
do mesmo módulo. Os renderizadores de relatório não importam o tema desktop.

## Fontes e unidades

A família principal é DejaVu Sans, seguida por Segoe UI, Noto Sans, Helvetica
Neue, Arial e Liberation Sans. Tk escolhe uma família instalada, respeitando
essa ordem; o navegador e Matplotlib usam suas listas de fallback. Na ausência
das famílias conhecidas, o backend conserva uma fonte genérica do sistema.
Não são distribuídos nem baixados arquivos de fontes.

| Papel | Tk, pt | Figuras, pt | HTML, px |
|---|---:|---:|---:|
| Título | 16 | 12 | 28 |
| Seção | 10 | 11 | 20 |
| Corpo | 9 | 10 | 16 |
| Informação secundária | 8 | 8 | 14 |
| Eixos | - | 10 | - |
| Ticks, legenda, anotação | - | 9 | 14 |
| Tabela | 9 | - | 14 |

Tamanhos positivos de fontes Tk são pontos. Espaçamentos de widgets são pixels.
O SVG converte os tamanhos de figura por `pt_to_svg`, com 96 unidades CSS por
polegada e 72 pontos por polegada. A largura responsiva da figura pode mudar
seu tamanho aparente; números iguais em unidades diferentes não indicam o
mesmo tamanho físico. Pillow usa tamanho em pixels nos painéis diagnósticos.

## Cores e apresentação

Interface e documentos compartilham superfícies claras, texto neutro e azul
institucional. O canvas escuro tem cores próprias para texto e controles.
Títulos, eixos, grades e referências usam papéis de documento; as configurações
Matplotlib são aplicadas dentro de `rc_context`, restaurando as preferências
anteriores mesmo quando o desenho falha. A impressão usa fundo claro e mantém
nomes e IDs junto às cores das legendas.

`data_colors.py` nomeia cores científicas existentes: revisão positiva/negativa,
exclusão, remoção, contornos, ausência de classe e perfis quantitativos. A paleta
categórica pertence à interpretação atual e é consultada por ID. Ela não é
derivada do azul institucional, inclusive quando editada pelo usuário.

Matplotlib mantém `RdBu_r`; SVG mantém sua interpolação histórica de três
pontos. Os dois conservam os intervalos de z-score e similaridade. Essa
diferença preexistente de interpolação não é corrigida por recoloração dos
dados. Apenas o texto de cada célula escolhe preto ou branco para maximizar
contraste sobre a cor efetivamente desenhada.

Os helpers numéricos de ticks, limites, nomes e callouts são comuns aos
renderizadores. Figuras compostas recebem tipografia e títulos; classificações,
máscaras e rasters destinados ao processamento continuam sem decoração.
