# Contribuir com RIIMA

Prepare um ambiente Python 3.10 ou superior com Tkinter e instale
`python -m pip install -r requirements/dev.txt`. Execute
`python -B -m pytest tests -q -p no:cacheprovider` em um display disponível; no Linux pode usar
`xvfb-run -a python -B -m pytest tests -q -p no:cacheprovider`. Os widgets reais são obrigatórios
na suíte GUI. Para processamento sem display, use
`python -B -m pytest tests -q -p no:cacheprovider --ignore=tests/test_riima_gui.py`.

Inclua em cada correção um caso mínimo que falhe antes da mudança e passe
depois. Use exemplos sintéticos; não acrescente lâminas, projetos `.riima`,
credenciais, caches ou saídas pessoais. Mudanças no método precisam declarar
o efeito nos resultados, além de passar nos testes de engenharia. Consulte
[testes e distribuição](docs/SHARING.md). Use diretórios temporários externos e mantenha os caminhos públicos em `PUBLIC_FILES.json`.

Mantenha a contribuição pequena e explique problema, comportamento esperado,
alteração e comandos executados. Gere um patch para revisão com
`git diff --binary > riima-change.patch` ou, para commits locais,
`git format-patch -1`. Confira o patch antes de encaminhá-lo. Não use
`git clean` para remover dados locais: arquivos ignorados podem ser análises
do usuário. Preserve os avisos BSD-3-Clause e documente direitos de novos recursos.

O mantenedor é Lucas Marques Barros. Não há endereço remoto ou canal público
de issues configurado nesta cópia. Entregue o patch e o relatório pelo mesmo
canal em que obteve o projeto, aos cuidados do mantenedor. Uma hospedagem
futura deverá registrar aqui o endereço efetivamente habilitado; não há
promessa de prazo de resposta. Veja [SUPPORT.md](SUPPORT.md).
