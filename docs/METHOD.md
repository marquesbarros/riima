# Método do RIIMA

RIIMA faz mapeamento de afinidade multimagem. O resultado computacional inicial é o mapa de clusters. O usuário organiza grupos, revisa pixels sob comando e aprova máscaras, sem identificação mineralógica automática. GUI e notebook compartilham `riima/core.py`; a revisão usa `riima/review.py`. As fontes metodológicas estão na [bibliografia](REFERENCES.md).

## 1. Entrada e registro

A entrada é um conjunto de N ≥ 1 imagens do mesmo campo. PPL, XPL e cPol são exemplos de aquisição; não há caminho científico específico para qualquer par. A referência escolhida serve apenas ao registro geométrico.

`register_images` registra cada imagem diretamente à referência, nunca em cadeia. A correlação de fase sobre gradientes estruturais fornece uma translação inicial; ECC permite uma pequena rotação e translação ([Kuglin e Hines, 1975](REFERENCES.md#kuglin1975phase); [Evangelidis e Psarakis, 2008](REFERENCES.md#evangelidis2008ecc)). Um candidato só é aceito se melhorar contra identidade nos mesmos pixels suportados. ECC e fase são comparados no suporte comum e respeitam os limites geométricos. Ausência de estrutura, estimativa não informativa ou falta de melhoria preservam as coordenadas com aviso de alinhamento não verificado, inclusive em pares idênticos já alinhados.

A indicação de registro fraco considera o escore final e sua melhora. É um aviso de inspeção, não uma certificação petrográfica. Transformações e escores são salvos em `registration.csv`.

## 2. Sobreposição e máscara

A interseção das áreas registradas define a sobreposição comum O. O suporte exige contribuição integral de pixels da imagem original na interpolação bilinear; valores refletidos nas bordas não ampliam o domínio válido. Os padrões de `Config` recusam uma análise abaixo de 30% da imagem de referência e avisam entre 30% e 60%. São limites de segurança de software, sem calibração petrográfica independente.

No modo automático, a máscara é:

```text
S = união das evidências de espécime de todas as imagens
A = união das máscaras de anotação de todas as imagens
V = O AND S AND NOT A
```

`specimen_area_mask` usa brilho ou saturação como evidência de espécime e identifica fundo escuro conectado à borda. `red_annotation_mask` exige cor e evidência de forma, como alongamento ou contato com a borda, para reduzir exclusões de regiões vermelhas/rosadas reais. Essas máscaras são heurísticas; a visualização **Máscara**, na GUI, apresenta o domínio válido atual. O produto exportado `analysis_mask.png` registra a máscara inicial. Novas análises da interface usam `auto`; projetos históricos preservam seu domínio salvo. O modo `overlap_only` usa somente O e continua disponível na CLI/API.

Com diagnósticos ativados, os componentes da máscara também podem ser exportados. Nenhuma máscara atribui uma identidade mineral.

## 3. Espaço de atributos simétrico

Toda imagem ativa, inclusive a referência, contribui exatamente quatro atributos:

$$
x(p)=[L_1^*(p),a_1^*(p),b_1^*(p),T_1(p),\ldots,L_N^*(p),a_N^*(p),b_N^*(p),T_N(p)],\qquad D=4N.
$$

`iter_optical_feature_planes` define os atributos consumidos pelo clustering. Uma suavização gaussiana controlada por `Config.feature_blur_sigma` antecede a conversão de cor.

`lab_star_channels` converte os códigos Lab de 8 bits do OpenCV para unidades CIELAB canônicas ([CIE, 2019](REFERENCES.md#cie2019lab); [OpenCV](REFERENCES.md#opencv-docs)):

$$
L^*=100L_8/255,\qquad a^*=a_8-128,\qquad b^*=b_8-128.
$$

A heterogeneidade cromática local é:

$$
T_i(p)=\sqrt{\sigma^2_{L_i^*}(p)+\sigma^2_{a_i^*}(p)+\sigma^2_{b_i^*}(p)}.
$$

As variâncias usam a mesma janela fixa de 11 × 11 pixels para todas as imagens, calculada por `E[x²] - E[x]²`. T descreve dispersão local de cor; não descreve orientação, periodicidade ou arranjo espacial. Nenhuma imagem recebe peso ou vetor especial por ocupar a posição de referência.

## 4. Normalização, afinidade e K-means

Uma amostra de pixels válidos determina a mediana e o intervalo interquartil de cada atributo. Essa convenção de escala robusta é descrita na [documentação de RobustScaler](REFERENCES.md#sklearn-robustscale); aqui é implementada com NumPy:

$$
z_j(p)=\frac{x_j(p)-\operatorname{mediana}(x_j)}{\operatorname{IQR}(x_j)}.
$$

Se o IQR estiver próximo de zero, a escala é substituída por 1. Todos os atributos têm **peso unitário**. Afinidade é expressa pela distância euclidiana no espaço normalizado:

$$
d(p,q)=\sqrt{\sum_{j=1}^{4N}(z_j(p)-z_j(q))^2}.
$$

`cluster_optical_affinity` ajusta K-means na amostra, com inicialização K-means++ e semente configurada ([Lloyd, 1982](REFERENCES.md#lloyd1982kmeans); [Arthur e Vassilvitskii, 2007](REFERENCES.md#arthur2007kmeanspp)). `predict_chunked` atribui todos os pixels válidos aos centros, reutilizando a mesma mediana/IQR em blocos para limitar memória. A ordem dos clusters é a ordem retornada pelo algoritmo, sem taxonomia mineralógica.

## 5. Confiança e regularização

Com d1 e d2 iguais às duas menores **distâncias euclidianas quadráticas** aos centros, a confiança óptica é:

```text
c = clip(1 - d1 / (d2 + ε), 0, 1)
```

Ela mede separação relativa no espaço de atributos da classificação inicial, não probabilidade de acerto mineralógico ou confiança calibrada nos rótulos revisados. Não existe confiança de família inferida automaticamente.

`confidence_regularize` só reclassifica pixels válidos de baixa confiança quando a maioria local é suficiente e a borda estrutural é fraca; votos e denominador consideram somente vizinhos válidos. `cleanup_uncertain_islands` trata componentes pequenos, compactos, de baixa confiança e borda fraca; componentes alongados são preservados para evitar apagar estruturas finas. Essas operações integram a análise automática. Seu resultado final constitui o mapa bruto preservado durante a interpretação.

## 6. Diagnóstico de K

`evaluate_cluster_counts` reutiliza a amostra normalizada para inércia, silhueta amostrada, Calinski-Harabasz e Davies-Bouldin ([Rousseeuw, 1987](REFERENCES.md#rousseeuw1987silhouette); [Caliński e Harabasz, 1974](REFERENCES.md#calinski1974); [Davies e Bouldin, 1979](REFERENCES.md#davies1979)). Davies-Bouldin usa a média das distâncias euclidianas ao centro, corrigida na revisão de outubro. `recommend_cluster_count` sugere a maior silhueta no intervalo avaliado; em empate, prefere o menor K. O texto é `K = X (sugerido)`.

Selecionar a sugestão apenas modifica a configuração da próxima análise. O usuário precisa solicitar nova execução. Os índices descrevem separação estatística nessa representação, não o número correto de minerais.

## 7. Clusters brutos, grupos, revisão e famílias legadas

São estados distintos:

- **Clusters brutos:** labels produzidos pela análise automática, preservados imutáveis.
- **Clusters interpretados:** partição histórica dos IDs brutos, preservada para a rastreabilidade da análise automática e da API legada.
- **Grupos atuais:** catálogo com IDs estáveis, nomes e cores próprios. Pode incluir categorias criadas sem membros brutos; criar uma categoria vazia não acrescenta centros K-means nem pixels.
- **Classificação revisada:** labels aplicados após exemplos Manter/Remover e exclusões; IDs dos grupos são preservados e máscaras aprovadas ficam protegidas.
- **Famílias legadas:** associações de IDs brutos de projetos anteriores, disponíveis como seleções opcionais dentro do agrupamento.

Uma união interpretativa altera o mapeamento, nunca os labels originais. Desfazer/restaurar interpretação não ajusta K-means nem refaz registro. A revisão pode mudar os labels aplicados, mantendo o mapa bruto. Uma união com pendências de revisão requer resolver essas pendências primeiro. Associações legadas continuam ligadas aos IDs brutos; não equivalem automaticamente à pertinência pixel a pixel aprendida depois.

A interpretação integra o projeto `.riima`; a exportação científica também disponibiliza a partição histórica em `interpretation.json`. No retorno de `run_analysis`, `result['optical']` contém a análise bruta e `result['interpretation']` contém essa interpretação legada. `save_session_outputs(prepared, interpretation=None)` grava produtos consistentes com o estado escolhido. A classificação aplicada e o catálogo atual pertencem a `result['review']`; sua exportação em `review/` inclui `catalog.json` com IDs, nomes, cores e proveniência dos grupos ativos e retirados. Métricas, relatório e similaridade consultam os nomes atuais desse catálogo, com os nomes legados como alternativa; ver a lista de saídas no [guia de utilização](USER_GUIDE.md#arquivos-de-saída).

## 8. Estatísticas e configuração

Estatísticas de clusters descrevem contagem de pixels, fração da máscara válida, centros/atributos e confiança, com nome e cor interpretativos quando aplicáveis. É necessário distinguir centros no espaço 4N normalizado de médias ou medianas dos atributos em suas unidades originais. Estatísticas de famílias agregam pixels dos IDs associados e registram membros, fração válida, nome e cor. Frações de área usam a máscara válida como denominador. Nenhuma dessas medidas implica contagem de objetos físicos.

`analysis_settings.json` registra entradas, referência, modo de máscara, configuração e definição do espaço de atributos. Os principais padrões são K = 8, amostra de 80 000 pixels, 3 inicializações, semente 13, sigma de suavização 0,8 e blocos de predição de 220 000 pixels. A definição completa dos parâmetros permanece em `Config`; os valores efetivamente usados em cada execução ficam no arquivo de configuração exportado.

## 9. Custo e limitações

O número de atributos cresce linearmente com N. Extração custa proporcionalmente ao número de pixels e canais; ajuste de K-means depende da amostra, K, D e iterações, e predição depende dos pixels válidos, K e D. A correlação de fase inclui FFT, com custo O(P log P) por imagem para P pixels. A interpretação remapeia IDs existentes sem refazer essas etapas.

A máscara e o registro exigem revisão, os limiares de segurança não têm validação petrográfica, e medidas internas de clustering não substituem referência mineralógica independente. A confiança computacional também não representa certeza de interpretação. Os testes verificam contratos computacionais; não estabelecem acurácia mineralógica nem desempenho em outras condições de aquisição ou hardware.

## 10. Revisão sob comando e persistência

Anotações são vinculadas às imagens de origem. `ClusterReview.refine_pending` aplica as mudanças acumuladas numa predição do conjunto. Com exemplos positivos e negativos separáveis, aprende um classificador binário por grupo usando RTrees do OpenCV, baseado em florestas aleatórias ([Breiman, 2001](REFERENCES.md#breiman2001forests); [OpenCV](REFERENCES.md#opencv-docs)). O refinamento não ajusta outra vez os centros K-means.

Com apenas um tipo de exemplo, a descrição local por vizinho mais próximo usa o mesmo espaço óptico normalizado ([Tax, 2001](REFERENCES.md#tax2001oneclass)). Se `p` é o exemplo mais próximo da consulta `x` e `q` o vizinho distinto mais próximo de `p`, há suporte quando `dist(x,p) <= dist(p,q)`. O raio deriva do espaçamento observado, sem limiar de distância escolhido manualmente. Como restrição conservadora adicional, o RIIMA exige que cada atributo da consulta esteja entre o mínimo e o máximo observados nos exemplos; assim, grande variação de um atributo não autoriza extrapolação nos demais. Um único perfil distinto admite apenas correspondências exatas. O suporte usa a semente configurada e até 5.000 perfis distintos, enquanto a detecção de contradições percorre todos os exemplos válidos em blocos antes da amostragem.

Suporte de **Manter** acrescenta regiões compatíveis sem rejeitar, por ausência de exemplos, a seleção inicial. Suporte de **Remover** rejeita somente esse grupo nas regiões compatíveis; os outros grupos continuam disputando os pixels. Os demais grupos não são convertidos em negativos implícitos. Conflitos entre suportes positivos de vários grupos não forçam uma escolha. Suportes positivos não ambíguos e florestas binárias usam uma ordem comum: a atualização mais recente prevalece nas reivindicações automáticas. Assim, um novo Manter pode corrigir uma floresta anterior, e uma atualização posterior dessa floresta pode rever a decisão. Marcações explícitas e pixels aprovados mantêm prioridade. A ordem é salva no projeto; atualizações exclusivas de fundo não a alteram. Perfis idênticos marcados com sinais opostos são excluídos do treino antes da amostragem e não recebem decisões automáticas daquele modelo.

A descrição local é sensível à representatividade dos exemplos e não distingue materiais opticamente indistinguíveis ou contextos que não estão nos atributos. Exemplos heterogêneos ou isolados não estabelecem identidade mineralógica. Exemplos negativos ajudam a delimitar o grupo. O suporte é reconstruído das anotações persistidas, sem depender de arquivos de origem externos.

Remover rejeita somente o grupo em edição. Excluir da análise retira pixels da competição até restaurá-los. Aprovar exige pendências aplicadas e protege toda a máscara final, incluindo pixels escolhidos pelo algoritmo. Reabrir libera somente esse grupo. Outros aprovados permanecem protegidos pixel a pixel.

Projetos `.riima` autossuficientes preservam imagens, labels brutos/aplicados, anotações, máscaras, IDs, nomes, cores, modelos, exclusões e progresso. O esquema atual é versão 6, com leitura dos esquemas 1–5. Arrays idênticos são armazenados uma vez; representações inteiras compactas restauram o dtype original. Escrita temporária e substituição evitam trocar o projeto válido por gravação incompleta; a carga valida esquema e integridade. Desfazer mantém histórico somente na sessão.

## 11. Métricas e relatório descritivo

`riima.metrics` mede componentes conectados, não grãos identificados. A área total usa todos os pixels; momentos e geometria medem até 2.000 maiores regiões elegíveis por grupo/imagem, com mínimo de 9 pixels e conectividade 8. Perímetro é contagem de bordas na grade de quatro vizinhos. Eixos são 4 vezes a raiz dos autovalores da covariância dos centros dos pixels. Área, perímetro e eixos não têm escala física calibrada ([Russ, 2002](REFERENCES.md#russ2002); [Hu, 1962](REFERENCES.md#hu1962moments), como fundamentos de momentos, sem implementar os sete invariantes de Hu).

Orientação é axial; direção e coerência usam área × anisotropia. Rosetas usam anisotropia × raiz da área, com ponderação diferente ([Fisher, 1993](REFERENCES.md#fisher1993); [Mardia e Jupp, 1999](REFERENCES.md#mardia1999directional)). η² resume separação óptica amostrada, sem constituir importância das árvores ou acurácia ([Richardson, 2011](REFERENCES.md#richardson2011)).

Grupos vazios aparecem nas tabelas com área e contagens iguais a zero, sem medidas geométricas ou atributos inventados, e não entram na similaridade. IDs retirados deixam de aparecer nesses produtos. Categorias novas sem membros brutos e sem pixels aplicados não consomem o orçamento de amostragem dos grupos existentes; passam a compartilhá-lo quando recebem pixels. Grupos legados mantêm a regra anterior mesmo quando vazios. Renomear ou alterar a cor não modifica medidas, amostras ou afinidades.

O relatório padroniza perfis de grupos, imputa médias somente para similaridade e calcula ligação de Ward ([Ward, 1963](REFERENCES.md#ward1963)). Histogramas/KDE usam largura de banda de Scott ([Scott, 1992](REFERENCES.md#scott1992density)). O gráfico de eixos chamado tipo Flinn 2D compara geometria e escala de regiões; não calcula X/Y versus Y/Z de deformação 3D ([Flinn, 1962](REFERENCES.md#flinn1962)).

A renderização usual das figuras utiliza Matplotlib. Quando a importação dessa biblioteca falha com `ImportError` ou `OSError`, um gerador SVG do RIIMA apresenta as 14 figuras no mesmo HTML, usando as métricas já calculadas, nomes e cores dos grupos. Essa escolha altera a apresentação, que pode diferir entre os geradores, sem redefinir as medidas ou o estado aplicado. A seleção da alternativa limita-se à importação; erros posteriores de dados ou de renderização continuam sendo comunicados como falhas de exportação.

Exportação exige pendências aplicadas e inclui classificação completa, nomes/cores atuais, seis seções e imagens incorporadas ao HTML. Relatório e similaridade podem consumir as métricas em memória, sem CSVs intermediários permanentes. `export_metrics` continua exportando os dados tabulares completos; `export_report(..., export_metrics=True)` publica HTML e CSVs no mesmo lote. Mapas são reduzidos somente na ilustração; contagens usam resolução original. Parâmetros, amostragem, omissões e valores indefinidos aparecem nos metadados. Os contratos de persistência estão em [PROJECT_FORMAT.md](PROJECT_FORMAT.md), e as fontes metodológicas em [REFERENCES.md](REFERENCES.md).
