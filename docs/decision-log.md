# Decision Log — Tech Challenge Fase 2

> **O que é.** O registro de cada decisão tomada na execução do TC2: o que foi decidido, quais
> alternativas existiam e **por quê** a escolhida ganhou. É a matéria-prima do `README.md` e do
> beat *Action* do vídeo STAR — os dois entregáveis que o enunciado nomeia.
>
> **Regra da casa:** preenchido **durante** a execução, nunca depois. Na Fase 1 este arquivo chegou
> a ~3.300 linhas e foi o que permitiu escrever a documentação sem reconstruir raciocínio de
> memória.
>
> **📍 Migrado em 12/10/2026** de `pos-mle/tech-challenges/fase-02/` para cá, na abertura da
> Etapa 1.

**Fonte do escopo:** `11MLET - Tech Challenge - Fase 2.pdf` (8 páginas, lido inteiro em
28/09/2026), nesta mesma pasta. Runbook de execução: skill `tech-challenge-f2`.

---

## §0 — Enquadramento

**Problema (do enunciado):** empresa de e-commerce precisa de um sistema preditivo para identificar
a **propensão de compra** de um usuário a partir do comportamento de navegação. Classificação
binária, dados tabulares. Foco declarado: **Engenharia de ML**, não complexidade do modelo.

**Pesos:** Clean Code 20% · Reprodutibilidade 20% · MLflow+Registry 20% · Docker 15% ·
DVC+Pipeline 15% · Modelagem 10%.

### Decisões de abertura — 5 resolvidas em 28/09/2026, 3 em aberto

| # | Decisão | Resolução | Status |
|---|---|---|---|
| 0.1 | **Prazo de entrega** | **03/11/2026** — 36 dias a contar de 28/09. Não consta no PDF do enunciado; veio da trilha | ✅ |
| 0.2 | **Grupo ou solo** | **SOLO.** O enunciado diz "a princípio em grupo" e cobra *"histórico de commits que evidencie o trabalho em grupo"* ⇒ **declarar a execução individual no README** e confirmar a liberação, como na Fase 1 | ✅ |
| 0.3 | **Repositório** | **NOVO** — `~/tc-mle-fase2`, **privado no GitHub até a entrega** (aberto em 12/10/2026). Não evolui o `tc-mle-fase1`: as quatro entregas graduadas (Poetry, `dvc.yaml`, Dockerfile-via-Poetry, Registry) são construção nova; o histórico solo de outra fase atrapalha o requisito de grupo; Clean Code vale 20% e a dívida do `src/` da Fase 1 está mapeada (revisita 4, 75, 78–80); e o repo da Fase 1 é portfólio entregue e público | ✅ |
| 0.4 | **Dataset** | **Online Shoppers Purchasing Intention** (UCI 468, 12.330 × 18, CC BY 4.0) — sugestão primária do enunciado. Confirmado por ele em 28/09 depois do estudo exploratório (`estudo-dataset.md`, nesta pasta) e da contraprova com um notebook do Kaggle. CSV em `recursos/tc2/` até a Etapa 3 (`dvc add`) | ✅ |
| 0.5 | **Poetry ou uv** | **Poetry**, desde o primeiro commit. O enunciado cita "(ou uv)" uma vez, mas o **entregável** diz `poetry install` e o **critério** diz "uso correto do Poetry". Requisito nominal ganha da preferência técnica (armadilha da Etapa 8-bis da Fase 1) | ✅ 12/10 |
| 0.6 | **POO ou Funcional** | critério cobra "uso correto" de um dos dois ⇒ exige declarar qual. **Inclinação (D01-A5, 19/10):** núcleo funcional puro por estágio + `main()` imperativo fino (único I/O) + POO só onde há estado real (estimadores do sklearn, params como `frozen dataclass`). Porquê: o `dvc repro` reaproveita o cache quando `deps`/`params` não mudam, o que pressupõe transparência referencial; o contrato do estágio já mora no `dvc.yaml`, então uma classe por estágio duplicaria esse contrato. Descartado: uma classe por estágio (recomendação de 28/09). **Fechada em 20/10, depois da D01-A6**, que propõe `DataPipeline`/`ModelTrainer`/`ModelEvaluator`. As três responsabilidades viram os três estágios do DVC; como cada estágio roda em subprocesso, um objeto não atravessaria de um estágio para o outro, e o estado passa por arquivo versionado. Strategy = estimadores do sklearn intercambiáveis por `fit`/`predict`; Factory = dicionário de construtores escolhido no `params.yaml`; nenhuma herança. Não esperar a D01-A7/A8: o argumento decisivo vem da stack, e embrulhar funções puras em classes depois é refatoração mecânica | ✅ 20/10 |
| 0.7 | **Remote do DVC** | pasta local × S3 do AWS Academy Learner Lab [186698] × Google Drive | 🔴 decidir na Etapa 3 |
| 0.8 | **Backend do MLflow e como o Registry fica visível ao avaliador** | SQLite (`mlflow.db`, como na Fase 1) × servidor | 🔴 decidir na Etapa 4 |

### Decisões já tomadas

| Data | Decisão | Por quê | Alternativa descartada |
|---|---|---|---|
| 2026-09-28 | **Kubernetes fica fora da entrega** | o enunciado oficial não menciona K8s em nenhuma das 8 páginas — nenhum requisito, nenhum peso. A promessa vinha do PDF de **abertura da fase**, que não é o documento avaliável | construir manifests/EKS para a entrega. Revisitar só se aparecer errata ou orientação no Discord |
| 2026-09-28 | **Runbook em skill separada (`tech-challenge-f2`)** | a skill da Fase 1 tem 3.949 linhas sobre o pipeline de churn, com outros pesos e outra stack. O que é transferível (catálogo de armadilhas, pontos fracos) é referenciado por nome, não duplicado | estender a skill `tech-challenge` |

---

## §1 — Etapa 1 · Clean Code e Estrutura

**Aberta em 12/10/2026**, depois da D01-A1. O corte: entra agora só o que **nenhuma aula futura
derruba** (estrutura, Poetry, lint, esquema do dado). O desenho do pipeline espera a D01-A5/A6
(POO × Funcional), o Dockerfile espera a D03 e o `dvc.yaml` espera a D04.

| Data | Decisão | Por quê | Alternativa descartada |
|---|---|---|---|
| 2026-10-12 | **Layout `src/purchase_propensity/`** (pacote instalável), não `src/` como pacote | `import purchase_propensity.schema` funciona igual no teste, no `dvc repro` e no container. Na Fase 1, `packages = ["src"]` fazia todo import começar por `src.`, e o nome do pacote não dizia nada sobre o domínio | `src/` como pacote (Fase 1) |
| 2026-10-12 | **Identificadores em inglês; docstrings, comentários e docs em português** | o código convive com a API do pandas/sklearn (`fit`, `transform`, `predict`), e misturar `fit_preprocessador` com `X_train` é inconsistência de naming, que é exatamente o que o critério avalia. O texto explicativo é para uma banca brasileira | tudo em português (Fase 1) |
| 2026-10-12 | **Esquema das colunas em código** (`schema.py`), não em `configs/` | separação: **`configs/` guarda o que se ajusta sem mudar código** (hiperparâmetros, seed, split → `params.yaml` do DVC); o esquema é **contrato do dado**, e se uma coluna muda, o código muda junto. Em Python ganha lint e autocomplete, sem parser nem validação extra (KISS). Revê a sugestão do `estudo-dataset.md` §6 | YAML em `configs/` + loader |
| 2026-10-12 | **Constantes de módulo**, não `dataclass` | um único esquema, sem segundo uso: a classe seria andaime (salvaguarda KISS da D01-A1). Fica como tuplas imutáveis + `FEATURE_COLUMNS` derivado delas (DRY) | `@dataclass(frozen=True) DatasetSchema` |
| 2026-10-12 | **`select` do ruff: `E W F I N UP B SIM ANN D C90 PL ERA PD RUF`** | cada item do critério ("naming, type hints, legibilidade") tem uma regra que **falha no CI**: `N` (naming), `ANN` (hints em parâmetro e retorno), `D` google (docstrings), `C90`/`PL` (tamanho e complexidade), `ERA` (código morto comentado), `RUF100` (`noqa` inerte). **Verificado com uma função-sonda:** sem hint dá `ANN001`/`ANN202`; sem docstring em módulo público dá `D103` | `select` mínimo da Fase 1 |
| 2026-10-12 | `ignore` **`D100`/`D104`**; `tests/**` isento de **`ANN`/`D`/`PLR2004`** | docstring de módulo/pacote repetiria o nome do arquivo. Teste é lido pelo nome (`test_alvo_nao_entra_nas_features`), e hint + docstring ali só duplicariam o nome | exigir tudo em todo lugar |
| 2026-10-12 | **`.gitignore` ignora o conteúdo de `data/` e `models/`, com exceção para `*.dvc`** | o DVC (Etapa 3) precisa que os ponteiros `.dvc` entrem no Git. Ignorar a **pasta** esconderia a exceção junto, então as pastas ficam visíveis (`!/data/**/`) e só os arquivos são ignorados. **Verificado** com `git check-ignore` (CSV ignorado, `.dvc` não) | ignorar só a pasta / decidir na D04 |
| 2026-10-12 | **venv dentro do projeto** (`poetry.toml`: `virtualenvs.in-project = true`, versionado) | o caminho do venv fica previsível para `Makefile`, IDE e CI. Mesmo assim o `Makefile` usa `poetry run`, para não depender do shell (falso positivo do ruff global na Fase 1) | venv no cache global do Poetry |
| 2026-10-12 | **pandas 3** (`^3.0.6`) | projeto novo não nasce preso à major anterior. O primeiro `poetry add` saiu com `^2.2` por descuido e foi corrigido no mesmo passo | pandas 2.x |
| 2026-10-12 | **Teste de contrato esquema × CSV com `skipif`** se o arquivo não existir | o CSV não está no Git até o `dvc add`. ⚠️ **Teste pulado é afirmação não verificada:** quando o DVC entrar (Etapa 3), o CI faz `dvc pull` e o `skipif` **sai** | sem teste contra o dado real |
| 2026-10-20 | **`extend-ignore-names = ["X", "X_*"]` no pep8-naming** | a D01-A7 (pág. 8) recomenda `X_train`/`y_test`, a convenção do sklearn, e o `N` reprova (`N803` no argumento `X`, `N806` na variável `X_train`, verificado com uma sonda). A exceção é pontual: `Data` e `Xtrain` continuam reprovando | `ignore = ["N803", "N806"]` (desliga o naming de variável inteiro, que é item do critério); `x_train` minúsculo (briga com o vocabulário do sklearn e da banca) |

✅ ~~Pendente para a Etapa 2: Poetry 1.8.4 e formato `[tool.poetry]`~~ → resolvido em 20/10, ver §2.

## §2 — Etapa 2 · Ambiente e Dependências

| Data | Decisão | Por quê | Alternativa descartada |
|---|---|---|---|
| 2026-10-20 | **Poetry 2.5.1 e metadados em `[project]` (PEP 621)**; `[tool.poetry]` só com `packages` e o grupo dev; `poetry-core>=2,<3` no build-system | o avaliador instala o Poetry atual (2.x). **Testado num clone limpo com o 2.5.1 e o formato antigo:** `poetry install` funcionava, mas `poetry check --lock` dava 5 warnings de deprecação (`[tool.poetry.name] is deprecated. Use [project.name] instead`…), o que pesa num critério que diz "uso correto do Poetry". `[project]` também é o padrão aberto que qualquer ferramenta lê (D02-A2, pág. 5). Na migração **nenhuma versão de pacote mudou**, só o formato do lock (2.0 → 2.1) | manter o 1.8.4 e o `[tool.poetry]`, que a D02-A2 ensina mas que o Poetry atual marca como deprecado |
| 2026-10-20 | **Pino do Poetry no CI trocado no mesmo commit** (`pipx install poetry==2.5.1`) | o lock 2.1 não é lido pelo Poetry 1.8, então o CI quebraria se ficasse no pino antigo. O Dockerfile (Etapa 3) usa o mesmo pino | deixar o CI sem pino (o formato do lock mudaria sem ninguém perceber) |
| 2026-10-20 | Não citar `poetry shell` na documentação | saiu no Poetry 2.0 (a D02-A2 ainda ensina). O substituto é `poetry env activate`, e o README já usa `poetry run`/`make`, que não dependem de ativar o venv | — |

## §3 — Etapa 3 · Containerização e Versionamento

*(a preencher)*

## §4 — Etapa 4 · Modelagem, Registry e Entrega

*(a preencher)*
