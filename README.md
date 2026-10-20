# RIIMA

**Really Intuitive Image Mapping by Affinity**

<img src="riima/assets/branding/riima_logo.png" alt="RIIMA" width="180">

Ferramentas de petrografia digital para imagens de um mesmo campo, com interpretação sob controle do petrógrafo. A interface reúne classificação por afinidade, agrupamento, revisão de máscaras e resultados descritivos. O núcleo também pode ser utilizado em Python.

**Versão:** `1.0.0-rc6` · **Autor:** Lucas Marques Barros · **Licença:** [BSD-3-Clause](LICENSE), com [atribuições e recursos](THIRD_PARTY.md).

## Instalação e uso

Requer Python 3.10 ou superior com Tkinter. Instale o pacote na pasta das fontes:

```bash
python -m pip install .
python -m riima
```

No Windows, `run_gui.bat` abre a interface com o Python instalado. Importe imagens correspondentes do mesmo campo, gere a classificação, selecione e agrupe classes, aplique marcações **Manter/Remover** e salve ou exporte os resultados.

O [guia de utilização](docs/USER_GUIDE.md) descreve comparação, revisão, proteção de máscaras e exportação. O [método](docs/METHOD.md) explicita fórmulas e limitações; clusters não identificam automaticamente espécies minerais. Consulte também [referências](docs/REFERENCES.md) e [formato de projeto](docs/PROJECT_FORMAT.md).

## Exemplo sintético e núcleo Python

Gere um par geométrico em uma pasta de sua escolha, fora das fontes:

```bash
python scripts/generate_example.py --output /caminho/externo/riima-exemplo
python -m riima --image /caminho/externo/riima-exemplo/field_a.png --image /caminho/externo/riima-exemplo/field_b.png --clusters 4 --output /caminho/externo/riima-resultados
```

No Windows, use caminhos como `C:\Users\seu_nome\Documents\riima-exemplo`. Os padrões sintéticos demonstram operações, sem constituir dados petrográficos de referência. O [notebook](notebooks/RIIMA_method_walkthrough.ipynb) apresenta as operações em 13 etapas e gera suas entradas e saídas em pasta temporária. Instale `requirements/notebook.txt` para executá-lo.

```python
from riima import core

analysis = core.run_analysis(
    ["imagem_1.jpg", "imagem_2.jpg"],
    "/caminho/externo/resultados",
    optical_cfg=core.Config(clusters=8),
)
```

## Manutenção e distribuição

`riima/` contém o código, template e recursos visuais; `docs/` contém os contratos e o manual. `tests/`, `scripts/` e `.github/` contêm verificações, ferramentas de manutenção e CI. [PUBLIC_FILES.json](PUBLIC_FILES.json) enumera os arquivos públicos com sua finalidade técnica; `.git/` é metadado de versionamento.

Consulte [instalação, testes e distribuição](docs/SHARING.md), [contribuição](CONTRIBUTING.md) e [suporte](SUPPORT.md). Identifique a cópia utilizada com `python -B scripts/source_identity.py` e guarde o manifesto junto aos resultados.
