# Instalação, testes e distribuição

Requer Python 3.10 ou superior, Tkinter e as dependências de `requirements.txt`. Instale com `python -m pip install .`; execute `python -m riima`. NumPy, OpenCV, Pillow e Matplotlib são instalados separadamente sob suas licenças. O wheel inclui o pacote, logo, ícone e template HTML.

O notebook é uma experiência opcional das fontes: instale `requirements/notebook.txt` e abra `notebooks/RIIMA_method_walkthrough.ipynb`. O gerador `scripts/generate_example.py --output PASTA` produz entradas geométricas sem micrografias. Escolha destinos externos para exemplos e resultados.

## Testes

Instale `python -m pip install -r requirements/dev.txt`. No Windows:

```powershell
./scripts/run_tests.ps1 -Python python
```

O executor desativa bytecode, usa arquivos temporários fora das fontes e restaura o ambiente ao terminar. `-OutputDirectory PASTA` conserva os resultados em uma pasta externa explicitamente escolhida. No Linux com GUI:

```bash
PYTHONDONTWRITEBYTECODE=1 xvfb-run -a python -B -m pytest tests -q -p no:cacheprovider --basetemp=/tmp/riima-pytest
```

Use um destino novo para cada execução. A suíte de widgets exige display real ou Xvfb. A suíte sem GUI pode usar `--ignore=tests/test_riima_gui.py`. As verificações de engenharia não substituem validação mineralógica independente ou avaliação de usabilidade.

## Construir pacotes

Instale `python -m pip install build wheel setuptools`. Execute a construção numa cópia temporária externa para conservar as fontes sem artefatos intermediários:

```bash
python -B scripts/build_share.py --output /caminho/externo/RIIMA.zip
```

Extraia o ZIP externo e, na pasta extraída, execute:

```bash
python -m build --sdist --wheel --outdir /caminho/externo/pacotes
```

`PUBLIC_FILES.json` define os caminhos de produto admitidos pelo ZIP. Arquivos desconhecidos não são incluídos automaticamente. O sdist inclui fontes, documentação, notebook e testes; o wheel inclui apenas código e recursos de execução. Os pacotes não requerem arquivos privados.

## Identificar e verificar a cópia

`python -B scripts/source_identity.py` imprime versão, caminhos, tamanhos e SHA-256 dos arquivos distribuídos. O identificador de conteúdo usa JSON UTF-8 ordenado, sem horários ou caminhos absolutos. O estado Git é medido separadamente como `clean`, `dirty` ou `unavailable`; uma pasta extraída não herda o HEAD de um repositório pai.

O ZIP tem ordem e metadados fixos e contém `SHARE_MANIFEST.json`. Cada entrada identifica os bytes efetivamente incluídos; o próprio manifesto fica fora do cálculo para evitar autorreferência. Guarde os pacotes e seus hashes junto aos resultados. Versão e HEAD isolados não identificam mudanças locais não registradas.

Para verificar instalação, CLI, relatório, recursos e widgets a partir de wheel e sdist:

```bash
python -B scripts/verify_distribution.py --artifacts /caminho/externo/pacotes --work /caminho/externo/verificacao-nova
```

No Linux, prefixe com `xvfb-run -a`. `--wheelhouse PASTA` permite instalação sem rede quando contém todos os wheels de runtime, desenvolvimento e build. O verificador também executa a suíte no sdist extraído. O workflow de CI cobre Windows/Linux e Python 3.10/3.13; uma execução local não comprova a execução remota.

`scripts/validate_workflow.py prepare --image IMAGEM --output PASTA` e `resume --output PASTA` verificam salvar, retomar, revisar e exportar em processos separados. Use imagens sintéticas e uma pasta externa nova. Os scripts de verificação recusam diretórios de trabalho dentro da árvore de fontes.
