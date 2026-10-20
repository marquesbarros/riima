# Referências metodológicas

Fontes dos algoritmos, convenções e bibliotecas descritos em [METHOD.md](METHOD.md).

<a id="kuglin1975phase"></a>
### Kuglin e Hines, 1975

Kuglin, C. D.; Hines, D. C. (1975). **The phase correlation image alignment method.** *Proceedings of the IEEE International Conference on Cybernetics and Society*, p. 163-165.

Metadados corroborados pelo [catálogo J-GLOBAL da Japan Science and Technology Agency](https://jglobal.jst.go.jp/en/detail?JGLOBAL_ID=201002077897196684). O texto original não foi recuperado nesta consulta e nenhum DOI foi confirmado. A atribuição histórica da correlação de fase é preservada; a operação usada no RIIMA também é documentada em [OpenCV](#opencv-docs).

<a id="evangelidis2008ecc"></a>
### Evangelidis e Psarakis, 2008

Evangelidis, G. D.; Psarakis, E. Z. (2008). **Parametric Image Alignment Using Enhanced Correlation Coefficient Maximization.** *IEEE Transactions on Pattern Analysis and Machine Intelligence*, 30(10), p. 1858-1865. DOI: [10.1109/TPAMI.2008.113](https://doi.org/10.1109/TPAMI.2008.113). [Fonte primária: IEEE](https://ieeexplore.ieee.org/document/4515873/).

Escopo: critério ECC e otimização iterativa do alinhamento paramétrico. O modelo euclidiano, os limites de transformação e as salvaguardas de aceitação do RIIMA são descritos em [METHOD.md](METHOD.md) e usam a API [findTransformECC](#opencv-docs).

<a id="lloyd1982kmeans"></a>
### Lloyd, 1982

Lloyd, S. P. (1982). **Least Squares Quantization in PCM.** *IEEE Transactions on Information Theory*, 28(2), p. 129-137. DOI: [10.1109/TIT.1982.1056489](https://doi.org/10.1109/TIT.1982.1056489). [Artigo publicado disponibilizado em Stanford](https://web.stanford.edu/class/ee398a/handouts/papers/Lloyd%20-%20Least%20Squares%20Q%20in%20PCM.pdf).

Escopo: quantização por mínimos quadrados e alternância entre atribuição e atualização dos centros. A data citada corresponde à publicação em periódico. K, amostragem, normalização dos atributos e predição por blocos pertencem ao desenho do RIIMA.

<a id="arthur2007kmeanspp"></a>
### Arthur e Vassilvitskii, 2007

Arthur, D.; Vassilvitskii, S. (2007). **k-means++: The Advantages of Careful Seeding.** *Proceedings of the Eighteenth Annual ACM-SIAM Symposium on Discrete Algorithms (SODA)*, p. 1027-1035. [Texto dos autores em Stanford](https://theory.stanford.edu/~sergei/papers/kMeansPP-soda.pdf); [registro institucional do autor](https://research.google/pubs/k-means-the-advantages-of-careful-seeding/).

Escopo: inicialização dos centros com distribuição proporcional à distância quadrática. Nenhum DOI é atribuído nesta bibliografia, pois os textos primários consultados não o confirmam.

<a id="cie2019lab"></a>
### ISO/CIE, 2019

International Organization for Standardization; International Commission on Illumination (2019). **ISO/CIE 11664-4:2019(E): Colorimetry - Part 4: CIE 1976 L\*a\*b\* colour space.** ISO/CIE. [Registro oficial da CIE](https://cie.co.at/publications/colorimetry-part-4-cie-1976-lab-colour-space-1).

Escopo: definição de CIELAB, luminosidade, croma, matiz e diferenças de cor. A norma não valida a identidade mineralógica de grupos produzidos nesse espaço. A conversão entre os valores canônicos e a representação OpenCV em 8 bits requer considerar a escala de L* e os deslocamentos de a* e b*.

<a id="breiman2001forests"></a>
### Breiman, 2001

Breiman, L. (2001). **Random Forests.** *Machine Learning*, 45(1), p. 5-32. DOI: [10.1023/A:1010933404324](https://doi.org/10.1023/A:1010933404324). [Fonte primária: Springer](https://link.springer.com/article/10.1023/A:1010933404324).

Escopo: florestas aleatórias como fundamento do refinamento supervisionado opcional implementado com [OpenCV RTrees](#opencv-docs). Exemplos anotados, proteção de grupos aprovados e aceitação de revisões são decisões do fluxo RIIMA. A eta quadrada descritiva do relatório não é a importância de variáveis de uma floresta.

<a id="rousseeuw1987silhouette"></a>

<a id="calinski1974"></a>
### Caliński e Harabasz, 1974

Caliński, T.; Harabasz, J. (1974). **A dendrite method for cluster analysis.** *Communications in Statistics*, 3(1), p. 1-27. DOI: [10.1080/03610927408827101](https://doi.org/10.1080/03610927408827101). [Fonte primária: Taylor & Francis](https://www.tandfonline.com/doi/abs/10.1080/03610927408827101).

Escopo: critério de dispersão entre e dentro dos grupos, utilizado como índice interno de agrupamento. Valores maiores favorecem separação nesse espaço e amostra, sem estabelecer classes petrográficas corretas.

<a id="davies1979"></a>
### Davies e Bouldin, 1979

Davies, D. L.; Bouldin, D. W. (1979). **A Cluster Separation Measure.** *IEEE Transactions on Pattern Analysis and Machine Intelligence*, PAMI-1(2), p. 224-227. DOI: [10.1109/TPAMI.1979.4766909](https://doi.org/10.1109/TPAMI.1979.4766909). [Fonte primária: IEEE](https://ieeexplore.ieee.org/document/4766909/).

Escopo: razão entre dispersão intragrupo e separação dos centros; valores menores são preferidos. A implementação RIIMA utiliza distância média ao centro na dispersão, conforme a convenção explicitada na documentação [scikit-learn](#sklearn-clustering).

<a id="hu1962moments"></a>
### Hu, 1962

Hu, M.-K. (1962). **Visual pattern recognition by moment invariants.** *IRE Transactions on Information Theory*, 8(2), p. 179-187. DOI: [10.1109/TIT.1962.1057692](https://doi.org/10.1109/TIT.1962.1057692). [Fonte primária: IEEE, acervo IRE](https://ieeexplore.ieee.org/document/1057692/).

Escopo: fundamento de descritores baseados em momentos de imagem. O RIIMA calcula momentos espaciais até segunda ordem para centroide, covariância e elipse equivalente; não calcula os sete invariantes de Hu. A convenção dos eixos e da excentricidade é também documentada em [medidas regionais scikit-image](#skimage2014).

<a id="russ2002"></a>
### Russ, 2002

Russ, J. C. (2002). **The Image Processing Handbook.** 4. ed. Boca Raton: CRC Press. ISBN: 9780849311420. [Currículo bibliográfico do autor, com edição e ano](https://drjohnruss.com/downloads/JCR_longcv.pdf).

Escopo: processamento de imagens e quantificação de forma. Medidas de regiões conectadas em pixels não demonstram que cada região corresponda a um grão. Perímetro por faces expostas na grade, filtros de tamanho e ponderações usados pelo RIIMA são escolhas da implementação.

<a id="fisher1993"></a>
### Fisher, 1993

Fisher, N. I. (1993). **Statistical Analysis of Circular Data.** Cambridge: Cambridge University Press. DOI: [10.1017/CBO9780511564345](https://doi.org/10.1017/CBO9780511564345). [Fonte primária: Cambridge University Press](https://www.cambridge.org/core/books/statistical-analysis-of-circular-data/324A46F3941A5CD641ED0B0910B2C33F).

Escopo: estatística circular e tratamento de direções, incluindo dados axiais. No RIIMA, o ângulo dobrado representa a equivalência entre direções separadas por 180 graus. As ponderações por área e anisotropia são escolhas específicas do relatório.

<a id="mardia1999directional"></a>
### Mardia e Jupp, 1999/2000

Mardia, K. V.; Jupp, P. E. **Directional Statistics.** Chichester: Wiley. DOI: [10.1002/9780470316979](https://doi.org/10.1002/9780470316979). ISBN da edição impressa: 9780471953333. [Fonte primária: Wiley Online Library](https://onlinelibrary.wiley.com/doi/book/10.1002/9780470316979); [registro institucional de Jupp](https://research-portal.st-andrews.ac.uk/en/publications/directional-statistics/).

Nota de data: Wiley informa primeira publicação online em 3 de janeiro de 1999 e copyright 1999; o registro institucional do autor e citações da edição impressa usam 2000. Escopo: estatística direcional e axial, sem inferência de deformação tridimensional a partir de orientações 2D.

<a id="richardson2011"></a>
### Richardson, 2011

Richardson, J. T. E. (2011). **Eta squared and partial eta squared as measures of effect size in educational research.** *Educational Research Review*, 6(2), p. 135-147. DOI: [10.1016/j.edurev.2010.12.001](https://doi.org/10.1016/j.edurev.2010.12.001). [Fonte primária: Elsevier](https://www.sciencedirect.com/science/article/pii/S1747938X11000029).

Escopo: definição e distinção entre eta quadrada e eta quadrada parcial. O relatório RIIMA calcula eta quadrada ordinária como proporção descritiva da soma de quadrados total explicada pela separação em grupos, na amostra utilizada. Essa medida não é causal nem é importância de um modelo preditivo.

<a id="ward1963"></a>
### Ward, 1963

Ward, J. H., Jr. (1963). **Hierarchical Grouping to Optimize an Objective Function.** *Journal of the American Statistical Association*, 58(301), p. 236-244. DOI: [10.1080/01621459.1963.10500845](https://doi.org/10.1080/01621459.1963.10500845). [Fonte primária: Taylor & Francis](https://www.tandfonline.com/doi/abs/10.1080/01621459.1963.10500845).

Escopo: agrupamento hierárquico por minimização do aumento de dispersão intragrupo. O RIIMA usa distância Ward ponderada pelos tamanhos dos grupos e padronização dos perfis. A transformação posterior em afinidade gaussiana é uma escolha exploratória, sem probabilidade mineralógica associada.

<a id="scott1992density"></a>
### Scott, 1992

Scott, D. W. (1992). **Multivariate Density Estimation: Theory, Practice, and Visualization.** New York: Wiley. DOI: [10.1002/9780470316849](https://doi.org/10.1002/9780470316849). ISBN: 9780471547709. [Fonte primária: Wiley](https://onlinelibrary.wiley.com/doi/book/10.1002/9780470316849).

Escopo: estimação de densidade e regra de banda de Scott. A KDE univariada implementada em NumPy usa desvio padrão amostral multiplicado por n elevado a -1/5. A convenção também está documentada na [SciPy](#scipy-kde), sem importação dessa biblioteca.

<a id="flinn1962"></a>
### Flinn, 1962

Flinn, D. (1962). **On folding during three-dimensional progressive deformation.** *Quarterly Journal of the Geological Society of London*, 118(1-4), p. 385-428. DOI: [10.1144/gsjgs.118.1.0385](https://doi.org/10.1144/gsjgs.118.1.0385). [Registro do editor, Geological Society](https://www.lyellcollection.org/doi/10.1144/gsjgs.118.1.0385); [registro bibliográfico com metadados do editor](https://cir.nii.ac.jp/crid/1361418520013488000).

Escopo: geometria da deformação progressiva tridimensional. A apresentação inspirada em Flinn no relatório RIIMA compara eixos de elipses equivalentes 2D; não estima os três eixos de um elipsoide de deformação nem a deformação de uma rocha. A paginação 385-433 do protótipo foi corrigida para 385-428. Os metadados e o DOI foram conferidos; o texto integral do editor não foi recuperado nesta consulta.

<a id="opencv-docs"></a>
### OpenCV: documentação oficial

OpenCV contributors (sem data). **OpenCV documentation: color conversions, registration, clustering, shape descriptors and RTrees.** Documentação oficial, consulta em 6 de outubro de 2026. As páginas `4.x` consultadas redirecionavam para a versão 4.13.0.

- [Color conversions](https://docs.opencv.org/4.x/de/d25/imgproc_color_conversions.html): CIELAB, escalas e codificação de canais. Em 8 bits, L* é escalado por 255/100 e a* e b* recebem deslocamento de 128; em ponto flutuante, esses deslocamentos não são aplicados.
- [Motion Analysis and Object Tracking](https://docs.opencv.org/4.x/d7/df3/group__imgproc__motion.html): `phaseCorrelate` e sua estimativa de deslocamento translacional.
- [Object Tracking](https://docs.opencv.org/4.x/dc/d6b/group__video__track.html): `findTransformECC` e modelos de transformação.
- [Clustering](https://docs.opencv.org/4.x/d5/d38/group__core__cluster.html): `kmeans` e `KMEANS_PP_CENTERS`.
- [Structural Analysis and Shape Descriptors](https://docs.opencv.org/4.x/d3/dc0/group__imgproc__shape.html): `connectedComponentsWithStats`, conectividade, área e centroides.
- [RTrees Class Reference](https://docs.opencv.org/4.x/d0/d65/classcv_1_1ml_1_1RTrees.html): floresta aleatória e importâncias opcionais do modelo. Essas importâncias não são a eta quadrada do relatório.

<a id="sklearn-robustscale"></a>
### scikit-learn: normalização robusta

scikit-learn contributors (sem data). **RobustScaler.** *scikit-learn API reference*. [Documentação oficial](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.RobustScaler.html). Consulta em 6 de outubro de 2026; página da versão 1.9.1.

Escopo: convenção de centrar cada atributo na mediana e escalar pelo intervalo interquartil, com quartis 25 e 75 como padrão, estimados na amostra e reutilizados em novos dados. O RIIMA implementa essa transformação em NumPy e explicita seu tratamento de escalas quase nulas; scikit-learn não é dependência de execução do aplicativo. A referência documenta a convenção, sem atribuir ao RIIMA todas as opções ou comportamentos de `RobustScaler`.

<a id="scipy-kde"></a>
### SciPy: KDE e definição de Ward

SciPy contributors (sem data). **gaussian_kde** e **linkage**. *SciPy API reference*. [Documentação oficial de KDE](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.gaussian_kde.html); [documentação oficial de Ward em linkage](https://docs.scipy.org/doc/scipy/reference/generated/scipy.cluster.hierarchy.linkage.html). Consulta em 6 de outubro de 2026.

Escopo: convenções consultadas para banda de Scott e distância Ward. Os cálculos do RIIMA são implementados em NumPy; citar essas definições não implica importação de SciPy. A banda de Scott deve ser distinguida das regras de Silverman.

<a id="sklearn-clustering"></a>
### scikit-learn: índices de agrupamento

scikit-learn contributors (sem data). **Clustering: clustering performance evaluation.** *scikit-learn User Guide*. [Documentação oficial](https://scikit-learn.org/stable/modules/clustering.html#davies-bouldin-index). Consulta em 6 de outubro de 2026.

Escopo: convenções de silhueta, Calinski-Harabasz e Davies-Bouldin, com seus limites como medidas internas. A página foi consultada para conferir dispersão média ao centro no índice Davies-Bouldin; scikit-learn não é dependência de execução RIIMA.

<a id="skimage2014"></a>
### van der Walt et al., 2014: scikit-image e medidas regionais

van der Walt, S.; Schönberger, J. L.; Nunez-Iglesias, J.; Boulogne, F.; Warner, J. D.; Yager, N.; Gouillart, E.; Yu, T.; scikit-image contributors (2014). **scikit-image: image processing in Python.** *PeerJ*, 2, e453. DOI: [10.7717/peerj.453](https://doi.org/10.7717/peerj.453). [Fonte primária: PeerJ](https://peerj.com/articles/453/); [arquivo de citação oficial](https://github.com/scikit-image/scikit-image/blob/main/CITATION.cff).

scikit-image contributors (sem data). **skimage.measure: regionprops.** [Documentação oficial](https://scikit-image.org/docs/stable/api/skimage.measure.html). Consulta em 6 de outubro de 2026; página da versão 0.26.0.

Escopo: referência preservada do protótipo e convenção de centroide, eixos da elipse com os mesmos segundos momentos e excentricidade. O RIIMA adapta esses cálculos a partir dos momentos das regiões; não importa scikit-image nem reproduz todos os descritores de `regionprops`.
