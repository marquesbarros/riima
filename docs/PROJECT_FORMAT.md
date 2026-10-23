# Persistência de projetos

Projetos `.riima` são arquivos ZIP com metadados JSON e arrays NumPy sem pickle.
A leitura aceita os esquemas 1 a 7. A gravação conserva o esquema 6 enquanto o
catálogo usa o agrupamento histórico; operações explícitas de criação, exclusão,
união, nome ou cor usam o esquema 7.
O arquivo continua autossuficiente: os caminhos das
imagens originais são proveniência e não precisam existir para abrir o projeto.

No esquema 6, `array_aliases` associa um nome lógico de array a um membro físico
do ZIP. Arrays com o mesmo dtype, geometria e bytes são gravados uma única vez;
isso elimina, entre outras redundâncias, a cópia da imagem RGB de referência.
A comparação usa SHA-256 e confirma igualdade byte a byte antes de compartilhar
o armazenamento. As referências são diretas: não permitem ciclos, cadeias,
nomes científicos desconhecidos, `metadata` ou colisão com um membro físico.
Cada array reconstruído passa pelas mesmas validações científicas dos esquemas
anteriores. Máscaras, anotações e labels editáveis recebem cópias independentes
em memória, mesmo quando compartilham um membro no arquivo.

`integer_dtypes` registra o dtype original de `initial`, `labels` e `locked`
quando um inteiro de 32 ou 64 bits pode ser armazenado em `int16`: todos os
valores devem estar entre −1 e 254. A leitura restaura o dtype registrado antes
da validação e da conversão histórica dos labels de revisão para `int32`.
Não há redução de resolução, mudança das imagens `uint8`, quantização ou
conversão dos arrays científicos de ponto flutuante. Máscaras booleanas seguem
como arrays NumPy comprimidos; não há codificação adicional por bits.

No modo independente, amostras iguais entre imagens usam `sample_layout=shared`:
o membro `0_sample_X` é gravado uma vez e o carregamento compartilha a mesma
instância de array. Amostras históricas diferentes permanecem distintas e são
regravadas com `sample_layout=per_item`. O modo registrado também conserva
amostras por item. Os esquemas 1 a 4 mantêm seus membros históricos por imagem;
igualdade de conteúdo e dtype permite compartilhar a memória após a validação.

Arrays científicos obrigatórios, arrays opcionais, interpretação e workspace
são validados antes da reconstrução da interface. Apenas `feature_means`,
`overlap` e `palette` são admitidos como arrays opcionais; não podem sobrescrever
campos obrigatórios. Uma falha de abertura conserva a sessão ativa.

O esquema conserva imagens, atributos e normalização, centroides, classificações
iniciais e revisadas, locks de grupos aprovados, máscaras de fundo e recuperação,
anotações aplicadas e pendentes, modelos RTrees em XML, agrupamentos, nomes e
cores interpretativos, seleções e extensões do workspace. Nomes e caminhos de
origem, metadados de registro e associações entre grupos, imagens e interpretações
permanecem disponíveis como proveniência. A otimização não cria um classificador
entre projetos nem descarta evidências para uma possível evolução futura.

No esquema 7, `group_catalog` contém objetos `{id, members}`: `id` é a identidade
interpretativa estável e `members` contém os clusters automáticos de origem.
Uma lista vazia de membros representa um grupo criado pelo usuário, sem centroide
automático inventado. `group_names` contém nomes explícitos por ID; um nome vazio
restaura a apresentação `Grupo ID`. Sem nome explícito, `legacy_group_names` e a
interpretação histórica fornecem os nomes associados à mesma origem.
`next_group_id` é o próximo ID disponível, nunca reutilizado depois de uma criação
bem-sucedida, inclusive se ela for desfeita. A paleta tem uma linha por ID reservado,
preservando as cores dos grupos remanescentes. IDs podem ultrapassar o K automático;
os limites originais de compactação inteira do esquema 6 continuam exatos.

`retired_groups` mantém nome, cor, membros de origem e motivo de exclusão/união
por ID removido. A união indica seu destino em `reason=merged_into:ID`.
Esses registros são proveniência, sem participação nas categorias de predição,
métricas ou similaridade. A interpretação automática histórica continua separada
do catálogo atual: K, centroides, atributos e labels iniciais não são modificados.
Excluir um grupo preenchido exige aplicar antes as marcações pendentes e reabrir
o grupo se aprovado. A predição condicionada usa apenas categorias remanescentes;
uma atribuição incompleta dos pixels válidos e editáveis cancela toda a operação.
Excluir um grupo vazio dispensa predição. A união colapsa os labels atuais sem
treinar em segundo plano; evidências/modelos prévios deixam o destino pendente
até Atualizar classificação. Exemplos positivos da união são reunidos; um exemplo
negativo só rejeita a união quando todos os grupos selecionados o rejeitavam.

Desfazer edições de grupos é um histórico da sessão, separado de Desfazer pincel.
Ele restaura o estado anterior completo, incluindo eventuais marcações, atualizações
e aprovações realizadas depois da edição desfeita, além do histórico de pincel
que existia naquele momento. Uma edição de grupos inicia um novo histórico de
pincel. O histórico de grupos guarda até dez operações e descarta as mais antigas
ao ultrapassar 128 MiB de arrays; a operação mais recente permanece reversível
mesmo quando sozinha excede esse orçamento. Imagens, atributos e classificação
automática não são copiados para o histórico. O projeto persiste o estado atual
e a proveniência, sem gravar o histórico de Desfazer.

O workspace usa `step=workbench` como marcador da bancada e aceita os nomes
históricos `preparation`, `interpretation`, `images`, `classify`, `organize`,
`refine` e `results`. O display `Máscara` apresenta o domínio válido atual;
`Máscara selecionada` apresenta somente os grupos escolhidos. `selected_groups` é independente de `selected_cluster`,
que identifica o grupo ativo para o pincel. Referências de seleção removidas
são filtradas ao carregar.

A exportação explícita de revisão inclui `review/catalog.json` com IDs, nomes,
cores, origem, contagens e estados dos grupos atuais, além da proveniência dos
removidos. As colunas de `review/statistics.csv` são preservadas. O produto
`review/image_NNN/initial.png` representa os labels e a paleta automáticos de origem.

O campo opcional `refinement_order` registra a precedência dos grupos, do mais
antigo para o mais recentemente atualizado. É uma lista sem duplicatas de IDs
inteiros existentes, comum aos suportes de um sinal e às florestas binárias.
Atualizar somente o fundo não muda essa ordem; excluir temporariamente exemplos
também não apaga sua precedência. Apagar ou reiniciar um grupo sem evidências ou
modelo restantes remove sua entrada. A ordem só é publicada após uma predição
completa bem-sucedida e é restaurada se o refinamento falhar. Não há novo tipo
de modelo serializado: os suportes continuam derivados das anotações.

Na ausência de `refinement_order`, inclusive nos esquemas 1–5, a leitura conserva
a regra histórica: suportes de um sinal precedem as florestas, cuja ordem segue
a inserção dos modelos no JSON. Essa migração não executa predição nem modifica
os labels salvos ao abrir o projeto. Marcações explícitas e pixels aprovados
mantêm prioridade sobre as reivindicações automáticas.

`output_history` guarda os destinos anteriores como proveniência. Após abrir um
projeto, `output` aponta para um destino novo ao lado do arquivo, com sufixo para
evitar colisões. A leitura não cria esse diretório. Na primeira exportação, a
interface reivindica o destino de forma exclusiva e verifica a gravação; se
necessário, permite escolher outra pasta ou cancelar conservando a sessão.

No modo independente, o orçamento de amostras precisa satisfazer
`sample_pixels >= número_de_imagens * clusters`. Cada imagem recebe pelo menos
K amostras quando tem pixels válidos suficientes; imagens menores conservam
todos os seus pixels válidos. O restante segue a divisão inteira por imagem.
