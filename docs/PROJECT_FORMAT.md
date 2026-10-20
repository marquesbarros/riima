# Persistência de projetos

Projetos `.riima` são arquivos ZIP com metadados JSON e arrays NumPy sem pickle.
A leitura aceita os esquemas 1 a 5. A gravação usa o esquema 5.

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

`output_history` guarda os destinos anteriores como proveniência. Após abrir um
projeto, `output` aponta para um destino novo ao lado do arquivo, com sufixo para
evitar colisões. A leitura não cria esse diretório. Na primeira exportação, a
interface reivindica o destino de forma exclusiva e verifica a gravação; se
necessário, permite escolher outra pasta ou cancelar conservando a sessão.

No modo independente, o orçamento de amostras precisa satisfazer
`sample_pixels >= número_de_imagens * clusters`. Cada imagem recebe pelo menos
K amostras quando tem pixels válidos suficientes; imagens menores conservam
todos os seus pixels válidos. O restante segue a divisão inteira por imagem.
