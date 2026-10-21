# RIIMA

**Really Intuitive Image Mapping by Affinity**

**Versão:** `1.0.0-rc6` (release candidate) · **Autor:** Lucas Marques Barros · **Licença:** [BSD-3-Clause](../LICENSE)

RIIMA reúne operações de processamento de imagens para apoiar a análise petrográfica digital. A interface reúne exploração de imagens, agrupamento, revisão e quantificação; o núcleo permite reutilizar essas operações em Python e no notebook. O especialista conduz a interpretação e as decisões qualitativas da análise.

O programa organiza padrões de afinidade em imagens correspondentes de um campo. Cada posição é representada pelas coordenadas CIELAB e pela heterogeneidade cromática local de todas as imagens disponíveis. K-means organiza essas posições em clusters; agrupamentos e máscaras são interpretados e corrigidos pelo usuário.

PPL, XPL e cPol são exemplos possíveis de aquisição, não requisitos do método. XPL e cPol designam configurações ópticas distintas. Toda imagem ativa contribui igualmente para a representação; a referência serve somente ao registro geométrico.

```text
imagens → representação 4N → normalização → afinidade → clusters → interpretação
```

**Cluster não é mineral.** Uma família é uma associação interpretativa de um ou mais clusters, criada pelo usuário. O software não atribui identidade mineralógica automaticamente.

## Início rápido

```bash
pip install -r requirements.txt
python -m riima
```

No Windows, `run_gui.bat` abre a janela sem console adicional. O notebook tem dependências próprias em `requirements/notebook.txt`.

1. **Imagens:** importar imagens do mesmo campo da lâmina, escolher as ativas e a referência para alinhamento.
2. **Classificação:** definir o número de grupos e clicar em **Gerar classificação**.
3. **Seleção e Agrupamento:** inspecionar máscaras, unir grupos, nomear e escolher cores.
4. **Revisão:** marcar o que manter/remover, atualizar a classificação e aprovar cada grupo.
5. **Resultados:** conferir o progresso, salvar o projeto e exportar mapas, máscaras ou tabelas.

A barra superior permite navegar entre as etapas disponíveis; **Anterior** e **Próximo** oferecem o mesmo percurso. O painel lateral mostra somente os controles da etapa atual. Imagem, classificação, zoom e seleção continuam na mesma área de trabalho. **Gerar classificação** mantém a etapa atual para permitir avaliar o resultado e ajustar K; avance com **Próximo** quando desejar.

Em **Seleção e Agrupamento**, um clique seleciona somente o grupo clicado. **Shift + clique** acrescenta ou remove grupos individuais da seleção. Vários grupos selecionados formam uma máscara conjunta para inspeção, sem modificar a classificação. **Unir** consolida a união quando desejado. Em **Máscara selecionada**, branco representa os pixels dos grupos escolhidos e preto representa os demais. **Destacar seleção** limita a sobreposição aos grupos selecionados. **Limpar seleção** deixa a máscara vazia. Ao entrar em **Revisão**, a seleção passa ao grupo em edição e acompanha a navegação até **Resultados**.

Escolher novas imagens salva o projeto da análise em andamento antes de preparar outra análise. Com **Pasta automática para novas imagens** ligada, a saída acompanha a pasta da primeira imagem selecionada, usando `riima_output` ou um sufixo livre quando essa pasta já existe. Desligue a opção para conservar a pasta escolhida. Escolher uma pasta pelo botão **…** também desliga a escolha automática. Cancelar a seleção ou uma falha ao salvar preserva a análise atual.

A função de ver classificações isoladas está integrada aos grupos, sem criar uma etapa ou um produto separado de famílias. Projetos anteriores preservam suas associações; quando existentes, **Seleções de projetos antigos** abre controles opcionais dentro de **Seleção e Agrupamento**. **Ver máscara** usa os grupos atuais dessas associações. Associações incompatíveis com uma união podem ser ajustadas ou removidas nesse painel.

**Enviar seleção para fundo** marca todas as máscaras atuais dos grupos selecionados, em todas as imagens, para exclusão. Clique em **Atualizar classificação** na Revisão para aplicar. Esses IDs deixam de disputar pixels. **Restaurar seleção do fundo** devolve as áreas registradas e os grupos à classificação na próxima atualização, inclusive depois de salvar e reabrir. Grupos aprovados precisam ser reabertos antes de mudar sua condição; os demais permanecem protegidos. Áreas que já estavam excluídas manualmente antes do envio continuam excluídas ao restaurar a seleção.

Os modos de visualização e as ferramentas de revisão usam botões com indicação da opção ativa. Em **Comparar**, os controles de Trabalho ficam ocultos e retornam com suas configurações preservadas.

## Revisão interativa de pixels

Importe imagens do mesmo campo da lâmina. O sistema alinha as imagens ativas à referência e produz uma única classificação, considerando a aparência de cada posição em todas as imagens (atributos 4N). Para analisar uma imagem isoladamente, importe somente ela; o fluxo é o mesmo e não há escolha de modo. Projetos salvos nas versões anteriores continuam podendo ser abertos e revisados com suas máscaras preservadas.

Depois de **Gerar classificação**, organize e nomeie os grupos em **Seleção e Agrupamento** e avance para **Revisão**:

1. Escolha a imagem e o grupo nos seletores. Um grupo pode conter um ou mais clusters originais.
2. Use **Manter** e **Remover** para pintar exemplos positivos e negativos sobre a imagem, a classificação ou a **Máscara selecionada**, sem mudar de visualização. **Fundo** marca áreas quebradas da lâmina e outras regiões que não devem pertencer a nenhum grupo. **Restaurar área** devolve pixels excluídos manualmente à análise sem apagar exemplos. **Apagar** limpa exemplos do grupo selecionado; **Navegar** permite arrastar a imagem. **Desfazer pincelada** recupera a pincelada anterior, incluindo seus segmentos. Zoom continua disponível. O raio do pincel é medido em pixels da imagem.
3. As marcações ficam pendentes: desenhar e trocar de imagem ou grupo não inicia treino, reclassificação ou salvamento completo. Clique em **Atualizar classificação** para aplicar todas as edições pendentes do conjunto em uma única atualização. Com exemplos de ambos os tipos, um classificador de árvores aleatórias aprende com as marcações de todas as imagens. Exemplos positivos aparecem em verde, negativos em vermelho, fundo em roxo e apagamentos pendentes em azul. Após atualizar, as marcações somem da visualização, mas permanecem no aprendizado. **Mostrar marcações** permite inspecioná-las. Com apenas um tipo, as correções pontuais são aplicadas; forneça exemplos do outro tipo para aprender a aparência do grupo.
4. Alterne entre **Imagem**, **Sobreposição**, **Classificação** e **Máscara selecionada**, sempre sobre o estado atual. **Opacidade** ajusta a sobreposição e **Contornos** mostra os limites atuais. **Comparar** oferece somente **Imagem original**, **Classificação atual** e **Máscara selecionada**; trocar a imagem atual atualiza ambos os lados.
5. Depois de aplicar as edições, clique em **Grupo pronto** para aprovar exatamente a máscara inteira exibida desse grupo em todas as imagens. A aprovação não inicia outro cálculo. Os pixels aprovados ficam protegidos e o ID do grupo deixa de disputar os pixels restantes. Uma marcação negativa exclui somente aquele grupo; o pixel permanece disponível. Uma marcação de **Fundo** exclui a região de todas as classificações seguintes, até ser restaurada e atualizada. O fundo e pixels rejeitados por todos os grupos restantes aparecem sem classificação, com ID `-1`. O pincel de fundo também respeita a proteção dos grupos prontos.
6. Continue com outro grupo. **Reabrir** libera somente o grupo escolhido para revisão, preservando as máscaras dos demais.

**Resetar grupo** limpa seus exemplos e reinicia o aprendizado na próxima atualização, preservando os outros grupos aprovados e as exclusões manuais de fundo. Se o grupo estava enviado ao fundo, também restaura suas áreas registradas. Um grupo aprovado precisa ser reaberto antes do reset. O reset e as ações de fundo em grupos encerram o histórico de pinceladas; a restauração explícita do grupo continua disponível no projeto salvo.

O ID estável de cada grupo é o menor ID dos seus clusters originais. A revisão acompanha as uniões e nomes da interpretação; os rótulos brutos e as imagens originais são preservados. Agrupe antes de marcar. Outros grupos sem revisão ainda podem ser unidos. Para alterar a composição de um grupo já revisado, reabra-o se estiver pronto, apague seus exemplos e clique em **Atualizar classificação** para descartar o modelo antigo antes de reagrupar. Projetos antigos mantêm os IDs brutos da revisão existente, sem converter suas máscaras implicitamente; nomes e famílias continuam editáveis. A revisão não atribui nomes mineralógicos automaticamente.

Ao concluir a análise inicial, o projeto completo é salvo em `projeto.riima` na pasta de resultados. Durante a revisão, use **Salvar projeto** para guardar o progresso, inclusive marcações ainda pendentes. **Salvar projeto como...** permite escolher outro arquivo, e **Abrir projeto...** recupera imagens, grupos, classificações iniciais/atuais, exemplos, modelos treinados, máscaras aprovadas e seleção atual. Também retoma etapa, visualização, opacidade e configurações do comparador. As imagens estão incluídas no arquivo, permitindo retomar a revisão mesmo quando os arquivos de origem não estão disponíveis. O fechamento normal da janela também salva o projeto; uma falha de salvamento mantém a janela aberta. O salvamento automático de agrupamentos grava somente o pequeno arquivo de interpretação. Arquivos de projeto existentes são substituídos atomicamente.

O projeto também guarda as exclusões de fundo, incluindo as ainda pendentes, e quais exemplos já foram aplicados. O histórico de **Desfazer pincelada** mantém até 50 pinceladas durante a sessão; é limpo ao aprovar, reabrir ou reagrupar e não é retomado ao reabrir o arquivo. Desfazer após uma atualização restaura os exemplos anteriores e exige outra atualização para modificar a classificação.

Ao salvar os resultados, os produtos da revisão ficam em `review/image_000/`, `review/image_001/` etc.: `labels.npy` (classificação atual), `approved.npy` (IDs protegidos, com `-1` para pixels disponíveis), `excluded.npy` (máscara booleana do fundo já aplicado), `clusters.png` e `initial.png`. `review/statistics.csv` registra contagens por imagem e grupo, incluindo os IDs dos clusters originais que o compõem. Projetos antigos com campos diferentes preservam os produtos iniciais de cada imagem em `initial/image_000/` etc.

Em **Resultados**, **Exportar máscara selecionada** salva um PNG binário da seleção na imagem atual, usando os pixels refinados. **Exportar visualização atual** salva o que está sendo mostrado. Mapas intermediários e dados históricos permanecem nos arquivos de exportação; a tela de trabalho usa uma única classificação atual.

**Exportar relatório** abre o diálogo para salvar `relatorio_analise_descritiva.html`. O arquivo reúne as seis seções de análise descritiva, gráficos, tabelas, referências e limites das métricas. A classificação final inteira e sua legenda aparecem antes do item 1. Gráficos e legenda usam os nomes e as cores atuais dos grupos, inclusive após unir, revisar, salvar e reabrir o projeto.

O relatório recalcula as métricas do estado aplicado e atualiza os CSVs na pasta `metrics`; não precisa exportá-los separadamente. Aplique marcações pendentes com **Atualizar classificação** antes de exportar. As imagens e os gráficos ficam incorporados no HTML, que pode ser aberto, compartilhado e impresso no navegador sem internet. A exportação preserva a seleção, a classificação e as imagens originais. Mapas maiores são reduzidos somente na ilustração; as métricas usam todos os pixels na resolução original.

Matplotlib, incluído em `requirements.txt`, é carregado somente durante a exportação do relatório. Se sua importação falhar por uma dependência indisponível ou bloqueada, o RIIMA gera automaticamente as 14 figuras em SVG incorporado ao HTML. Essa alternativa usa as mesmas métricas, nomes e cores dos grupos; a apresentação dos gráficos pode variar. Não é necessário instalar outro pacote gráfico para essa alternativa. Em instalações anteriores, atualize as dependências com `pip install -r requirements.txt`.

Os gráficos se adaptam à quantidade de grupos e aos nomes das imagens; grupos vazios ou isotrópicos exibem valores não definidos quando apropriado. Os cálculos são descritivos: regiões conectadas não são necessariamente grãos, anisotropia é geométrica e similaridade não atribui identidade mineralógica. A geração ocorre sob demanda e pode manter a interface ocupada por alguns segundos, conforme o conjunto.

O refinamento usa a aparência Lab/T dos exemplos. Imagens com aparências diferentes exigem exemplos representativos; padrões visualmente iguais podem precisar de mais revisão manual. O treino e a previsão ocorrem somente no comando de atualização, ainda na interface; conjuntos grandes podem mantê-la ocupada durante esse cálculo. Salvar também pode demorar porque o arquivo inclui todas as imagens. O aprendizado requer OpenCV 4.x com o módulo `ml`, conforme `requirements.txt`.

## Métricas sob demanda

Em **Resultados**, **Exportar métricas** cria a pasta `metrics` com quatro CSVs:

- `group_summary.csv`: área/proporção, contagens, estatísticas de tamanho, excentricidade, anisotropia geométrica, direção predominante e coerência por imagem e grupo.
- `regions.csv`: medidas de cada região conectada, incluindo eixos equivalentes, razão de eixos, diâmetro equivalente, perímetro na grade, compacidade e contato com a borda.
- `attribute_statistics.csv`: média, desvio padrão, mediana e quartis dos atributos Lab/T amostrados por imagem e grupo.
- `attribute_separation.csv`: ranking dos atributos pela separação estatística amostrada entre grupos (eta quadrado).

As fórmulas de forma usam momentos e eixos equivalentes documentados pelo [OpenCV](https://docs.opencv.org/4.x/d3/dc0/group__imgproc__shape.html) e [scikit-image](https://scikit-image.org/docs/stable/api/skimage.measure.html). A implementação reutiliza somente NumPy/OpenCV já instalados.

As áreas e contagens incluem todos os pixels. Para manter a leveza, a descrição de forma usa até 2.000 regiões de maior área por grupo/imagem, com mínimo de 9 pixels; os CSVs registram quantas ficaram de fora. Atributos usam até 2.000 pixels por grupo e 10.000 por imagem. As métricas só são calculadas ao exportar, após aplicar as edições, sem refazer a classificação.

`metrics_info.json` descreve unidades, limites, fórmulas e convenções: comprimentos em pixels, áreas em pixels quadrados, ângulos de 0 a 180 graus (0 para a direita, 90 para baixo), perímetro na grade de pixels incluindo furos. Regiões sem alongamento definido não recebem uma direção. A direção dominante é uma média axial ponderada por área e anisotropia, calculada separadamente para cada imagem e grupo.

As regiões conectadas não são uma segmentação validada de grãos; anisotropia geométrica não mede anisotropia óptica. O ranking de atributos é uma descrição estatística amostrada de separação entre grupos, não a importância aprendida pelo modelo nem uma identificação mineralógica.

## Representação multimagem

Para N imagens, cada pixel válido tem exatamente `D = 4N` atributos:

```text
x(p) = [L1*, a1*, b1*, T1, ..., LN*, aN*, bN*, TN]
Ti(p) = sqrt(var(Li*) + var(ai*) + var(bi*))
```

CIELAB é usado em unidades canônicas. T mede a heterogeneidade cromática numa janela local fixa, sem descrever orientação ou periodicidade. Todos os atributos são normalizados por mediana/IQR sobre a amostra válida, recebem pesos unitários e entram no K-means com distância euclidiana. A definição é a mesma para uma, duas ou mais imagens.

Cada imagem não referência é registrada diretamente à referência. A sobreposição comum limita a máscara final; uma sobreposição insuficiente pode impedir a análise. Esses limiares são salvaguardas de software, não parâmetros calibrados petrograficamente. A máscara automática usa evidência de espécime em qualquer imagem e exclui possíveis anotações. O modo de somente sobreposição comum permite inspecionar casos em que essa heurística não é adequada. Ver [método](METHOD.md).

## Três estados preservados

| Estado | Papel |
|---|---|
| Clusters brutos | Resultado imutável da análise, preservado após toda operação manual |
| Clusters interpretados | Agrupamentos manuais dos IDs brutos, com nomes e cores editáveis |
| Famílias | Associações semânticas de clusters brutos escolhidas pelo usuário |

Na etapa **Seleção e Agrupamento**, os grupos podem ser selecionados individualmente ou em conjunto, renomeados, recoloridos e unidos interpretativamente. **Limpar seleção** desmarca os grupos sem alterar a classificação ou apagar exemplos. Selecionar permite inspecionar uma máscara; unir modifica o agrupamento. Desfazer e restaurar recuperam agrupamentos sem recalcular clustering, respeitando a proteção da revisão. A API mantém operações de famílias para compatibilidade com projetos e produtos anteriores. Uma associação como clusters 2, 5 e 8 → “Plagioclásio” expressa uma decisão do usuário, não uma identificação feita pelo algoritmo.

Nomes, cores, grupos e associações ficam em `interpretation.json`; os rótulos brutos continuam disponíveis separadamente. Máscaras, contornos e contagens da área de trabalho usam a classificação atual após as correções.

## Diagnóstico de K e visualização

O diagnóstico opcional calcula inércia, silhueta amostrada, Calinski-Harabasz e Davies-Bouldin na amostra já usada na análise. O maior valor de silhueta determina `K = X (sugerido)`. Usar a sugestão apenas preenche o campo de K; é necessário clicar em **Gerar classificação** para executar outra análise. O K sugerido não representa um número comprovado de minerais.

O visualizador mantém a imagem original, a classificação atual e a máscara da seleção na mesma área de trabalho, sem uma lista de camadas de produtos. O comparador mantém duas fontes selecionáveis, divisor de 0 a 100%, linha vertical e círculo central, sem sinais laterais. Zoom e pan são compartilhados; a legenda identifica a imagem, os grupos atuais e seu progresso. Navegar e trocar a apresentação não recalcula registro nem clustering.

## Linha de comando e API

```bash
python -m riima --image imagem_1.jpg --image imagem_2.jpg \
  --reference 0 --output resultado --clusters 8
```

A forma histórica `--ppl imagem_ppl.jpg --cpol imagem_cpol.jpg` também aceita um par de imagens. `--reference` é um índice a partir de zero. Consulte `python -m riima --help` para os argumentos disponíveis; `--diagnostics` inclui diagnósticos de máscara.

```python
from riima.core import Config, run_analysis, save_session_outputs

result = run_analysis(
    ["imagem_1.jpg", "imagem_2.jpg"], "resultado",
    optical_cfg=Config(clusters=8), reference_index=0,
    diagnostics=False, mask_mode="auto",
)
raw = result["optical"]
interpretation = result["interpretation"]
save_session_outputs(result, interpretation=interpretation)
```

## Arquivos de saída

Os produtos ficam em uma única pasta. Os mapas `.npy` preservam os IDs; os `.png` permitem inspeção visual.

| Arquivo | Conteúdo |
|---|---|
| `clusters.npy`, `clusters.png` | Rótulos e mapa de clusters brutos |
| `interpreted_clusters.npy`, `interpreted_clusters.png` | Agrupamentos interpretados, separados dos rótulos brutos |
| `families.npy`, `families.png` | Famílias definidas pelo usuário; pixels não associados permanecem sem família |
| `analysis_mask.png` | Área válida da análise |
| `optical_confidence.png` | Separação relativa entre os dois centros mais próximos, não probabilidade mineralógica |
| `cluster_statistics.csv` | Contagem de pixels, fração válida, atributos/centros e confiança, com metadados interpretativos quando aplicáveis |
| `family_statistics.csv` | Clusters pertencentes, pixels, fração válida, nome e cor de cada família |
| `interpretation.json` | Estado manual de grupos, nomes, cores e associações |
| `analysis_settings.json` | Imagens, referência e configuração efetivamente usada |
| `registration.csv` | Transformação e escores de registro por imagem |

Famílias não são calculadas automaticamente. Sem associações manuais, o mapa não contém famílias atribuídas e a tabela não contém famílias criadas.

A API `load_interpretation(optical, directory)` recupera interpretações salvas sobre um resultado óptico disponível. Campos antigos que não pertencem ao modelo atual são ignorados; interpretações só são convertidas quando a correspondência com os IDs brutos é segura. Informação não conversível recebe aviso. **Abrir projeto...** aceita os novos arquivos `.riima`; pastas de exportação antigas continuam disponíveis pela API de interpretação.

## Notebook e método

O [notebook](../notebooks/RIIMA_method_walkthrough.ipynb) apresenta imagens, registro, máscara, CIELAB, T, vetor 4N, normalização, afinidade, K-means, interpretação, famílias e exportação pelo núcleo compartilhado. As entradas padrão são geradas programaticamente na pasta temporária de saída. Os padrões geométricos demonstram operações computacionais; não representam uma lâmina nem validam uma interpretação petrográfica.

Consulte [método](METHOD.md), [referências](REFERENCES.md) e [formato de projeto](PROJECT_FORMAT.md). Os testes verificam propriedades computacionais e consistência de exportação. Não há validação mineralógica independente por contagem de pontos, DRX ou referência equivalente. Registro, máscara heurística e escolhas de K exigem inspeção do usuário.

## Reprodutibilidade e dependências

A semente padrão é `13`. A mesma entrada e configuração devem reproduzir os clusters brutos no mesmo ambiente, independentemente da interpretação posterior. Pequenas diferenças de ponto flutuante podem depender da versão e do paralelismo do OpenCV; guarde os parâmetros e as versões ao comparar execuções.

O núcleo usa NumPy e OpenCV; a interface usa Tkinter e Pillow. Não requer GPU ou rede neural. A classificação inicial não exige exemplos rotulados; a revisão aprende com as marcações do usuário. O software é distribuído sob [BSD-3-Clause](../LICENSE); recursos de terceiros estão descritos em [THIRD_PARTY.md](../THIRD_PARTY.md). Cite a versão e a identificação do código utilizada: `Barros, L. RIIMA (Really Intuitive Image Mapping by Affinity), 1.0.0-rc6`. Ainda não há DOI público informado para o software.

## Testar e construir

Instale `requirements/dev.txt`. No Windows, `scripts/run_tests.ps1` executa os testes com bytecode desativado e arquivos temporários externos. Para os comandos de construção e verificação dos pacotes, consulte [SHARING.md](SHARING.md).
