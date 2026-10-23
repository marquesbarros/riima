# RIIMA

**Really Intuitive Image Mapping by Affinity**

**Versão:** `1.0.0-rc6` (release candidate) · **Autor:** Lucas Marques Barros · **Licença:** [BSD-3-Clause](../LICENSE)

RIIMA reúne operações de processamento de imagens para apoiar a análise petrográfica digital. A interface reúne exploração de imagens, agrupamento, revisão e quantificação; o núcleo permite reutilizar essas operações em Python e no notebook. O especialista conduz a interpretação e as decisões qualitativas da análise.

O programa organiza padrões de afinidade em imagens registradas do mesmo campo ou em campos independentes. No modo registrado, cada posição reúne as coordenadas CIELAB e a heterogeneidade cromática local de todas as imagens ativas. No modo independente, cada campo mantém sua geometria e compartilha a classificação óptica. K-means organiza os atributos em clusters; grupos e máscaras são interpretados e corrigidos pelo usuário.

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

A bancada reúne importação, classificação, organização e revisão sobre o mesmo Canvas. No painel contextual, importe as imagens e escolha as ativas, o modo de análise e a referência para alinhamento, quando aplicável. Defina K e clique em **Gerar classificação**. A **Máscara** é calculada pelo procedimento automático. Os grupos aparecem no painel para inspeção, criação, organização e revisão, sem trocar de tela.

A barra superior reúne **Abrir projeto**, **Salvar projeto** e **Exportar**, disponíveis durante todo o trabalho. O menu permite salvar relatório, visualização, máscara, métricas ou a árvore científica completa e abrir a pasta de resultados. As opções são habilitadas conforme a análise e a seleção disponíveis. As seções de imagens e configuração podem ser recolhidas e continuam acessíveis depois da análise. Abrir uma seção ou alterar um parâmetro prepara a próxima execução e conserva os resultados atuais. O painel pode ser redimensionado pelo divisor lateral. O botão de foco, junto ao zoom, ou **F9** recolhe temporariamente o painel, ampliando a imagem; restaurá-lo mantém seleção, grupo ativo, ferramenta, zoom, posição e revisão. Os botões compactos de edição usam ícones; passe o cursor, dê foco pelo teclado ou pressione F1 para ler sua função. Ctrl+S salva, Ctrl+O abre um projeto e Ctrl+Shift+S permite salvar como.

A lista única mostra cor, nome, ID estável e estado de cada grupo, inclusive grupos vazios e aprovados. Nomes extensos podem ser consultados pelo tooltip. Um clique seleciona somente o grupo clicado; **Shift/Ctrl + clique** acrescenta ou remove grupos individuais. A seleção múltipla forma uma máscara conjunta para inspeção, sem modificar a classificação. O **grupo ativo** é o destino do pincel e permanece independente da seleção: use **Ativar grupo**, Enter ou duplo clique para escolhê-lo. Seu nome aparece na seção de revisão. Em **Máscara selecionada**, branco representa os pixels dos grupos escolhidos e preto representa os demais. **Destacar seleção** limita a sobreposição à seleção; **Limpar seleção** deixa sua máscara vazia. Projetos históricos são retomados na mesma bancada, com os grupos e estados salvos preservados.

**Similaridade** abre uma janela compacta com os demais grupos ordenados por afinidade ao primeiro grupo selecionado, usando nomes e cores atuais. Clicar em um resultado acrescenta o grupo à seleção principal, preservando os grupos já selecionados e o grupo ativo do pincel. A união continua sendo um comando explícito. O cálculo reaproveita o perfil do relatório: sete medidas geométricas, médias de atributos ópticos e direção axial, com afinidade gaussiana após padronização. Grupos vazios e de fundo ficam fora da comparação. Aplique as marcações pendentes antes de comparar. A medida descreve semelhança dentro desta análise; não é probabilidade de identidade mineralógica. A consulta funciona em memória, sem exportar CSVs. Renomear ou recolorir atualiza a janela aberta sem recalcular medidas. Alterações de categorias ou da classificação encerram consultas obsoletas; abra Similaridade novamente para consultar o estado atual.

Escolher novas imagens prepara as entradas da próxima execução e conserva a análise existente. Ao gerar uma classificação que substituirá o trabalho atual, o programa pede confirmação e salva primeiro o projeto anterior. O novo projeto usa outro nome quando `projeto.riima` já existe. Com **Pasta automática** ligada, a saída acompanha a pasta da primeira imagem selecionada, usando `riima_output` ou um sufixo livre quando essa pasta já existe. Desligue a opção para conservar a pasta escolhida. Escolher uma pasta pelo botão **…** também desliga a escolha automática. Cancelar a substituição ou uma falha de preparação ou salvamento preserva a análise atual.

A função de ver classificações isoladas está integrada aos grupos. Projetos anteriores preservam suas associações; quando existentes, **Seleções de projetos antigos** abre controles opcionais no painel contextual. **Ver máscara** seleciona os grupos atuais vinculados aos clusters de origem da associação, preservando o grupo ativo. Associações incompatíveis com uma união podem ser ajustadas ou removidas nesse painel.

**Enviar seleção para fundo** marca todas as máscaras atuais dos grupos selecionados, em todas as imagens, para exclusão. Clique em **Atualizar classificação** para aplicar. Esses IDs deixam de disputar pixels. **Restaurar seleção do fundo** devolve as áreas registradas e os grupos à classificação na próxima atualização, inclusive depois de salvar e reabrir. Grupos aprovados precisam ser reabertos antes de mudar sua condição; os demais permanecem protegidos. Áreas que já estavam excluídas manualmente antes do envio continuam excluídas ao restaurar a seleção.

Os modos de visualização e as ferramentas de revisão usam botões com indicação da opção ativa. Em **Comparar**, os controles de Trabalho ficam ocultos e retornam com suas configurações preservadas.

## Gerenciamento dos grupos

**Criar grupo** adiciona uma categoria vazia com ID próprio e cor distinguível, sem executar K-means nem atribuir pixels. Nome e cor podem ser editados. Ative o grupo, forneça exemplos positivos com **Manter** e aplique **Atualizar classificação** para iniciar sua classificação. Um grupo vazio não possui centroide automático nem medidas ópticas inventadas.

**Unir grupos** reúne a seleção em uma categoria e conserva o menor ID selecionado, seu nome e sua cor. Os membros de origem ficam associados ao destino. Se havia exemplos ou aprendizado nos grupos unidos, a classificação fica pendente até **Atualizar classificação**. Exemplos positivos são reunidos; uma restrição negativa rejeita a união somente quando todos os grupos selecionados rejeitavam aquele pixel. Grupos aprovados precisam ser reabertos antes da união.

**Excluir grupo** elimina uma categoria. Para grupos preenchidos, aplique antes as marcações pendentes. A operação reclassifica todos os pixels válidos e editáveis entre as categorias remanescentes usando os atributos ópticos e as evidências existentes. Preserva exatamente pixels protegidos, fundo, máscara inicial, IDs e cores remanescentes. Não executa K-means novamente nem altera os labels automáticos iniciais. Se as restrições impedirem uma atribuição completa, a exclusão é cancelada integralmente. Grupos vazios são excluídos sem predição; grupos aprovados ou enviados ao fundo precisam ser reabertos ou restaurados. O último grupo não pode ser excluído.

**Desfazer grupos** restaura o estado completo anterior à edição do catálogo: grupo, nome, cor, labels, exemplos, modelos, proteções e seleção. Isso também reverte revisões e aprovações feitas depois daquela edição. É separado de **Desfazer pincelada**. O histórico guarda até dez edições na sessão, limitado a 128 MiB de arrays, preservando sempre a operação mais recente. Não é gravado no projeto. Os IDs de criações bem-sucedidas continuam reservados após Desfazer; criar novamente gera outro ID. Consulte [formato de projeto](PROJECT_FORMAT.md).

## Revisão interativa de pixels

Na seção de configuração, escolha **Registradas (mesmo campo)** para alinhar as imagens ativas à referência e classificar cada posição pelo conjunto de atributos 4N. Escolha **Independentes (campos distintos)** para compartilhar a classificação óptica entre campos, sem alinhá-los. Uma imagem isolada pode ser analisada pelo mesmo fluxo. Projetos anteriores continuam podendo ser abertos e revisados com suas máscaras preservadas.

Depois de **Gerar classificação**, os grupos e as ferramentas de revisão ficam disponíveis no painel:

1. Escolha a imagem no seletor e ative um grupo na lista única. Um grupo pode conter clusters originais ou ser uma categoria criada por exemplos, sem membros automáticos.
2. Use **Manter** e **Remover** para pintar exemplos positivos e negativos sobre a imagem, a classificação ou a **Máscara selecionada**, sem mudar de visualização. **Fundo** marca áreas quebradas da lâmina e outras regiões que não devem pertencer a nenhum grupo. **Restaurar área** devolve pixels excluídos manualmente à análise sem apagar exemplos. **Apagar marcas** limpa exemplos do grupo selecionado; **Navegar** permite arrastar a imagem. **Desfazer pincelada** recupera a pincelada anterior, incluindo seus segmentos. Zoom continua disponível. O slider ajusta o raio de 1 a 150 pixels da imagem; as setas ajustam um pixel e Home/End selecionam os limites. O contorno acompanha o cursor e o zoom, sem integrar a imagem exportada.
3. As marcações ficam pendentes: desenhar e trocar de imagem ou grupo não inicia treino, reclassificação ou salvamento completo. Clique em **Atualizar classificação** para aplicar todas as edições pendentes do conjunto em uma única atualização. Com exemplos de ambos os tipos, um classificador de árvores aleatórias aprende com as marcações de todas as imagens. Exemplos positivos aparecem em verde, negativos em vermelho, fundo em roxo e apagamentos pendentes em azul. Após atualizar, as marcações somem da visualização, mas permanecem no aprendizado. **Mostrar marcações** permite inspecioná-las. Com somente exemplos positivos ou negativos, um suporte óptico local reconhece aparências semelhantes para incluir ou rejeitar pixels não marcados, inclusive em outras imagens. Uma única aparência marcada permite somente coincidências exatas; exemplos mais variados ampliam o suporte local. Perfis com exemplos contraditórios não generalizam essas correções. Aparências iguais em contextos distintos podem exigir mais exemplos e inspeção manual; o método não compreende contexto mineralógico nem transfere aprendizado entre projetos.
4. Alterne entre **Imagem**, **Sobreposição**, **Classificação**, **Máscara selecionada** e **Máscara**, sempre sobre o estado atual. **Máscara** mostra em branco toda a área válida da análise, incluindo pixels sem grupo; **Máscara selecionada** mostra somente os grupos escolhidos. **Opacidade** ajusta a sobreposição e **Contornos** mostra os limites atuais. **Comparar** oferece **Imagem original**, **Classificação atual**, **Máscara selecionada** e **Máscara**, com divisor arrastável e zoom compartilhado; trocar a imagem atual atualiza ambos os lados. A legenda científica pode ser aberta quando necessária, sem ocupar permanentemente uma segunda lateral.
5. Depois de aplicar as edições, use **Aprovar grupo** para proteger exatamente a máscara inteira exibida desse grupo em todas as imagens. A aprovação não inicia outro cálculo. Os pixels aprovados ficam protegidos e o ID do grupo deixa de disputar os pixels restantes. Uma marcação negativa exclui somente aquele grupo; o pixel permanece disponível. Uma marcação de **Fundo** exclui a região de todas as classificações seguintes, até ser restaurada e atualizada. O fundo e pixels rejeitados por todos os grupos restantes aparecem sem classificação, com ID `-1`. O pincel de fundo também respeita a proteção dos grupos aprovados.
6. Continue com outro grupo. **Reabrir grupo** libera somente o grupo ativo para revisão, preservando as máscaras dos demais.

**Resetar grupo** limpa seus exemplos e reinicia o aprendizado na próxima atualização, preservando os outros grupos aprovados e as exclusões manuais de fundo. Se o grupo estava enviado ao fundo, também restaura suas áreas registradas. Um grupo aprovado precisa ser reaberto antes do reset. O reset e as ações de fundo em grupos encerram o histórico de pinceladas; a restauração explícita do grupo continua disponível no projeto salvo.

Os grupos históricos conservam o menor ID dos clusters de origem. Novas categorias recebem IDs crescentes e persistentes, independentes do K automático. União conserva o menor ID selecionado; exclusão não renumera os demais. A revisão acompanha o catálogo atual, enquanto rótulos brutos, centroides e imagens originais permanecem preservados. Projetos antigos mantêm suas máscaras sem conversão implícita. A revisão não atribui nomes mineralógicos automaticamente.

Ao concluir a análise inicial, somente o projeto completo é salvo em `projeto.riima` na pasta de resultados. Durante a revisão, use **Salvar projeto** para guardar o progresso, inclusive marcações ainda pendentes. **Salvar projeto como...**, no menu superior, permite escolher outro arquivo, e **Abrir projeto...** recupera imagens, grupos, classificações iniciais/atuais, exemplos, modelos treinados, máscaras aprovadas e seleção atual. Também retoma visualização, opacidade e configurações do comparador. As imagens estão incluídas no arquivo, permitindo retomar a revisão mesmo quando os arquivos de origem não estão disponíveis. O fechamento normal da janela também salva o projeto; uma falha de salvamento mantém a janela aberta. Alterações de agrupamento, nomes e cores ficam em memória com aviso de progresso não salvo. Use **Salvar projeto** para persistir essas alterações; a organização não recompacta o projeto nem cria arquivos técnicos a cada ação. Arquivos de projeto existentes são substituídos atomicamente.

O projeto também guarda as exclusões de fundo, incluindo as ainda pendentes, e quais exemplos já foram aplicados. O histórico de **Desfazer pincelada** mantém até 50 pinceladas durante a sessão; é limpo ao aprovar, reabrir ou reagrupar e não é retomado ao reabrir o arquivo. Desfazer após uma atualização restaura os exemplos anteriores e exige outra atualização para modificar a classificação.

Durante análise, treinamento, carregamento, salvamento e exportação, a barra de estado informa a operação e mostra o logotipo RIIMA girando em um espaço fixo. Os controles de edição ficam temporariamente bloqueados; aguarde a conclusão antes de fechar a janela. A animação não representa uma porcentagem de conclusão e desaparece ao terminar ou ocorrer um erro.

O treinamento e a retomada de modelos requerem `cv2.ml.RTrees_create` do OpenCV 4.x, conforme as dependências de instalação. Se essa interface estiver ausente, o RIIMA informa a versão, o módulo e o interpretador carregados para orientar a reparação nesse ambiente. Uma atualização que falha mantém os exemplos pendentes para nova tentativa.

Ao solicitar **Exportar → Exportação científica completa**, os produtos da revisão ficam em `review/image_000/`, `review/image_001/` etc.: `labels.npy` (classificação atual), `approved.npy` (IDs protegidos, com `-1` para pixels disponíveis), `excluded.npy` (máscara booleana do fundo já aplicado), `clusters.png` e `initial.png`. `review/statistics.csv` registra contagens por imagem e grupo, incluindo os IDs dos clusters originais que o compõem. Projetos antigos com campos diferentes preservam os produtos iniciais de cada imagem em `initial/image_000/` etc. Essa exportação publica os produtos de classificação, revisão, projeto e as quatro tabelas CSV com `metrics_info.json` na pasta `metrics`, dentro da mesma transação. Aplique marcações pendentes antes de exportar. Uma falha nas métricas também preserva todo o conjunto anterior. Métricas isoladas e relatório HTML continuam disponíveis em comandos próprios.

No menu superior **Exportar**, **Máscara selecionada (PNG)** salva uma máscara binária da seleção na imagem atual, usando os pixels refinados. **Visualização atual (PNG)** salva o que está sendo mostrado. Mapas intermediários e dados históricos continuam disponíveis na exportação científica completa; a tela de trabalho usa uma única classificação atual.

**Exportar relatório** abre o diálogo para salvar `relatorio_analise_descritiva.html`. O arquivo reúne as seis seções de análise descritiva, gráficos, tabelas, referências e limites das métricas. A classificação final inteira e sua legenda aparecem antes do item 1. Gráficos e legenda usam os nomes e as cores atuais dos grupos, inclusive após unir, revisar, salvar e reabrir o projeto.

O relatório recalcula as métricas do estado aplicado em memória e publica somente o HTML, sem criar ou atualizar CSVs intermediários. Solicite **Exportar → Métricas (CSV)** quando precisar das tabelas separadas. Aplique marcações pendentes com **Atualizar classificação** antes de exportar. As imagens e os gráficos ficam incorporados no HTML, que pode ser aberto, compartilhado e impresso no navegador sem internet. A exportação preserva a seleção, a classificação e as imagens originais. Mapas maiores são reduzidos somente na ilustração; as métricas usam todos os pixels na resolução original.

Matplotlib, incluído em `requirements.txt`, é carregado somente durante a exportação do relatório. Se sua importação falhar por uma dependência indisponível ou bloqueada, o RIIMA gera automaticamente as 14 figuras em SVG incorporado ao HTML. Essa alternativa usa as mesmas métricas, nomes e cores dos grupos; a apresentação dos gráficos pode variar. Não é necessário instalar outro pacote gráfico para essa alternativa. Em instalações anteriores, atualize as dependências com `pip install -r requirements.txt`.

Os gráficos se adaptam à quantidade de grupos e aos nomes das imagens; grupos vazios ou isotrópicos exibem valores não definidos quando apropriado. Os cálculos são descritivos: regiões conectadas não são necessariamente grãos, anisotropia é geométrica e similaridade não atribui identidade mineralógica. A geração ocorre sob demanda e pode manter a interface ocupada por alguns segundos, conforme o conjunto.

O refinamento usa a aparência Lab/T dos exemplos. Imagens com aparências diferentes exigem exemplos representativos; padrões visualmente iguais podem precisar de mais revisão manual. O treino e a previsão ocorrem somente no comando de atualização, fora da thread de eventos da interface. Salvar também pode demorar porque o arquivo inclui todas as imagens. Durante os cálculos, o indicador discreto junto ao estado permanece animado, sem mudar o espaço da imagem; controles incompatíveis ficam temporariamente bloqueados. O treinamento por árvores requer OpenCV 4.x com o módulo `ml`, conforme `requirements.txt`.

## Métricas sob demanda

Em **Exportar → Métricas (CSV)**, o programa cria a pasta `metrics` com quatro CSVs:

- `group_summary.csv`: área/proporção, contagens, estatísticas de tamanho, excentricidade, anisotropia geométrica, direção predominante e coerência por imagem e grupo.
- `regions.csv`: medidas de cada região conectada, incluindo eixos equivalentes, razão de eixos, diâmetro equivalente, perímetro na grade, compacidade e contato com a borda.
- `attribute_statistics.csv`: média, desvio padrão, mediana e quartis dos atributos Lab/T amostrados por imagem e grupo.
- `attribute_separation.csv`: ranking dos atributos pela separação estatística amostrada entre grupos (eta quadrado).

As fórmulas de forma usam momentos e eixos equivalentes documentados pelo [OpenCV](https://docs.opencv.org/4.x/d3/dc0/group__imgproc__shape.html) e [scikit-image](https://scikit-image.org/docs/stable/api/skimage.measure.html). A implementação reutiliza somente NumPy/OpenCV já instalados.

As áreas e contagens incluem todos os pixels. Para manter a leveza, a descrição de forma usa até 2.000 regiões de maior área por grupo/imagem, com mínimo de 9 pixels; os CSVs registram quantas ficaram de fora. Atributos usam até 2.000 pixels por grupo e 10.000 por imagem. As métricas são calculadas sob demanda na comparação de similaridade, no relatório ou na exportação das tabelas, após aplicar as edições, sem refazer a classificação.

`metrics_info.json` descreve unidades, limites, fórmulas e convenções: comprimentos em pixels, áreas em pixels quadrados, ângulos de 0 a 180 graus (0 para a direita, 90 para baixo), perímetro na grade de pixels incluindo furos. Regiões sem alongamento definido não recebem uma direção. A direção dominante é uma média axial ponderada por área e anisotropia, calculada separadamente para cada imagem e grupo.

As regiões conectadas não são uma segmentação validada de grãos; anisotropia geométrica não mede anisotropia óptica. O ranking de atributos é uma descrição estatística amostrada de separação entre grupos, não a importância aprendida pelo modelo nem uma identificação mineralógica.

## Representação multimagem

Para N imagens, cada pixel válido tem exatamente `D = 4N` atributos:

```text
x(p) = [L1*, a1*, b1*, T1, ..., LN*, aN*, bN*, TN]
Ti(p) = sqrt(var(Li*) + var(ai*) + var(bi*))
```

CIELAB é usado em unidades canônicas. T mede a heterogeneidade cromática numa janela local fixa, sem descrever orientação ou periodicidade. Todos os atributos são normalizados por mediana/IQR sobre a amostra válida, recebem pesos unitários e entram no K-means com distância euclidiana. A definição é a mesma para uma, duas ou mais imagens.

No modo registrado, cada imagem não referência é registrada diretamente à referência. A sobreposição comum limita a máscara final; uma sobreposição insuficiente pode impedir a análise. Esses limiares são salvaguardas de software, não parâmetros calibrados petrograficamente. A máscara automática usa evidência de espécime em qualquer imagem e exclui possíveis anotações. Novas análises da interface usam o procedimento `auto`, sem um seletor de métodos. Projetos antigos preservam o modo e a máscara salvos; `overlap_only` continua disponível pela CLI/API. Ver [método](METHOD.md).

## Classificações e proveniência

| Estado | Papel |
|---|---|
| Clusters brutos | Resultado imutável da análise, preservado após toda operação manual |
| Classificação atual | Categorias interpretativas com IDs estáveis, exemplos, nomes, cores e máscaras revisadas |
| Interpretação histórica | Partição dos clusters brutos preservada pela API e pelos produtos históricos |
| Famílias | Associações semânticas de clusters brutos escolhidas pelo usuário |

Na lista principal, os grupos podem ser criados, selecionados individualmente ou em conjunto, renomeados, recoloridos, unidos e excluídos. **Limpar seleção** desmarca os grupos sem alterar a classificação ou apagar exemplos. Selecionar permite inspecionar uma máscara; unir modifica o catálogo e excluir reclassifica seus pixels editáveis. A API mantém operações sobre a partição histórica e famílias para compatibilidade com projetos e produtos anteriores. Uma associação como clusters 2, 5 e 8 → “Plagioclásio” expressa uma decisão do usuário, não uma identificação feita pelo algoritmo.

Nomes, cores, grupos, associações e rótulos brutos ficam preservados no projeto `.riima`. A exportação científica completa disponibiliza a interpretação histórica em `interpretation.json`, o catálogo atual em `review/catalog.json` e seus mapas separados. Máscaras, contornos, contagens, métricas e relatório da área de trabalho usam a classificação atual após as correções.

## Diagnóstico de K e visualização

O diagnóstico opcional calcula inércia, silhueta amostrada, Calinski-Harabasz e Davies-Bouldin na amostra já usada na análise. O maior valor de silhueta determina `K = X (sugerido)`. Usar a sugestão apenas preenche o campo de K; é necessário clicar em **Gerar classificação** para executar outra análise. O K sugerido não representa um número comprovado de minerais.

O visualizador mantém a imagem original, a classificação atual, a máscara da seleção e a Máscara na mesma área de trabalho, sem uma lista de camadas de produtos. O comparador mantém duas fontes selecionáveis, divisor de 0 a 100%, linha vertical e círculo central, sem sinais laterais. Zoom e pan são compartilhados; a legenda identifica a imagem, os grupos atuais e seu progresso. Navegar e trocar a apresentação não recalcula registro nem clustering.

**Máscara** permite inspecionar a área válida no Canvas e no comparador geral, sem recalcular a análise. **Fundo** e **Restaurar área** corrigem exclusões manuais, respeitando a proteção dos grupos aprovados. A máscara inicial e as exclusões de revisão continuam armazenadas separadamente; o estado atual e as exportações consideram as correções aplicadas.

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

Na interface, o salvamento padrão produz o projeto `.riima`; o menu superior cria os demais produtos somente quando solicitados. A tabela abaixo descreve a exportação científica completa e os produtos da CLI/API, que preservam seus comportamentos anteriores. Os mapas `.npy` preservam os IDs; os `.png` permitem inspeção visual.

| Arquivo | Conteúdo |
|---|---|
| `clusters.npy`, `clusters.png` | Rótulos e mapa de clusters brutos |
| `interpreted_clusters.npy`, `interpreted_clusters.png` | Agrupamentos interpretados, separados dos rótulos brutos |
| `families.npy`, `families.png` | Famílias definidas pelo usuário; pixels não associados permanecem sem família |
| `analysis_mask.png` | Área válida da análise |
| `optical_confidence.png` | Separação relativa entre os dois centros mais próximos, não probabilidade mineralógica |
| `cluster_statistics.csv` | Contagem de pixels, fração válida, atributos/centros e confiança, com metadados interpretativos quando aplicáveis |
| `family_statistics.csv` | Clusters pertencentes, pixels, fração válida, nome e cor de cada família |
| `interpretation.json` | Partição histórica de clusters brutos, nomes, cores e associações |
| `review/catalog.json` | Categorias atuais, IDs, nomes, cores, estados, membros de origem e proveniência das removidas |
| `review/image_NNN/labels.npy` | Classificação interpretativa atual, inclusive categorias criadas e exclusões de grupos |
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
