# Propensão de compra em e-commerce

Tech Challenge da Fase 2 da pós em Machine Learning Engineering (FIAP + Alura).
Pipeline de classificação que estima a probabilidade de uma sessão de navegação terminar em
compra, com o dataset [Online Shoppers Purchasing Intention](https://archive.ics.uci.edu/dataset/468/online+shoppers+purchasing+intention+dataset)
(UCI 468: 12.330 sessões, 15,5% com compra).

Stack: scikit-learn, DVC (dado e pipeline), MLflow (tracking e Model Registry), Poetry e Docker.

**Vídeo (STAR, 5 min):** _link a incluir_

> Execução individual: todos os commits são de um único autor.

## Requisitos

- Python **3.12.5** (o patch está em `.python-version`). O Poetry não instala Python, e quem
  garante a versão exata é o Docker.
- Poetry **2.5.1**. O `poetry.lock` está no formato 2.1, que o Poetry 1.x não lê.
- Git e acesso à internet: o dataset é baixado da UCI.
- Docker, só para o caminho pelo container.

## Como rodar

Do clone até o modelo em produção no Registry, na máquina local:

```bash
poetry install                     # ambiente exato do poetry.lock (deps de produção + dev)

# 1. Dado e pipeline
poetry run dvc update data/raw/online_shoppers_intention.zip.dvc   # baixa o zip da UCI
poetry run dvc repro               # extract → prepare → train → evaluate

# 2. Runs comparáveis no MLflow: um por candidato, mesmo commit e mesmo dado
poetry run dvc exp run -f -S configs/params.yaml:train.model=dummy
poetry run dvc exp run -f -S configs/params.yaml:train.model=logistic_regression
poetry run dvc exp run -f -S configs/params.yaml:train.model=hist_gradient_boosting

# 3. Promoção ao Registry e predição pelo champion
poetry run promote                 # aplica o critério do params.yaml e registra o vencedor
poetry run predict                 # pontua o test.csv com o modelo @champion
```

Quatro pontos que não são óbvios:

- **`dvc update`, e não `dvc pull`.** O dataset não fica num remote do DVC: ele é importado da
  UCI por `dvc import-url`, e o `.dvc` guarda a URL de origem e o MD5. Num clone novo o
  `dvc pull` falha (*"No remote provided"*). O import é `frozen`, então o dado só muda com um
  `dvc update` explícito, e uma mudança na UCI aparece no `git diff` do `.dvc`.
- **`-f` em cada `dvc exp run`.** O run do MLflow é um efeito que o DVC não enxerga. Quando
  uma combinação de parâmetros já rodou, o run-cache do DVC restaura as saídas sem executar o
  `train`, e o MLflow fica sem o run, sem aviso nenhum. O `-f` obriga a execução.
- **O `promote` roda fora do `dvc.yaml`.** Ele escolhe entre runs que já existem, e o resultado
  dele (um alias no Registry) não é um arquivo que o DVC possa versionar. Quando promove, ele
  imprime o `dvc exp apply <experimento>` do vencedor: depois de vários `dvc exp run`, o
  `params.yaml` e o `dvc.lock` do workspace são os do **último** experimento, e o commit
  precisa ser o do campeão. Na ordem acima o último já é o campeão.
- **`configs/params.yaml:` no `-S`.** Sem o nome do arquivo, o DVC procura um `params.yaml`
  na raiz e recusa com *"Could not override 'train.model'"*.

`make ci` roda o lint (ruff) e os testes, a mesma sequência do CI.

Se a UCI estiver fora do ar, baixe o zip
[da página do dataset](https://archive.ics.uci.edu/dataset/468/online+shoppers+purchasing+intention+dataset)
para `data/raw/online_shoppers_intention.zip` e siga a partir do `dvc repro`. O CSV extraído
tem de ter MD5 `cc6ec1db03b4f10f8de52c56ff48b085`.

## Como verificar o Registry

O MLflow usa um banco SQLite local (`mlflow.db`), sem servidor, e o banco fica fora do Git:
ele guarda caminhos absolutos da máquina que treinou. Os passos 2 e 3 acima o recriam.

```bash
poetry run mlflow ui --backend-store-uri sqlite:///mlflow.db    # http://127.0.0.1:5000
```

- **Experiments → `purchase-propensity`:** um run por treino, com os parâmetros dos blocos `prepare` e
  `train`, as métricas da validação cruzada (`cv_pr_auc_mean`, `cv_pr_auc_std`, ROC-AUC), o
  modelo logado e as tags `git.commit`, `git.dirty`, `data.md5` e `dvc.exp`.
- **Models → `purchase-propensity`:** a versão com o alias `champion`. As tags da versão
  registram a promoção: `criterion` (os números que decidiram), `approved_by`, `promoted_at`
  e `previous_champion`.

O `predict` também confere o Registry: ele carrega `models:/purchase-propensity@champion` e
imprime a versão que usou.

## Rodar no Docker

A imagem traz o ambiente (Python 3.12.5 e as dependências do `poetry.lock`, instaladas pelo
Poetry no estágio de build). O projeto entra por bind mount do repositório em `/app`, e por
isso as saídas do pipeline (`dvc.lock`, `models/`, `metrics/`) ficam no host.

```bash
docker build --build-arg UID="$(id -u)" -t tc-mle-fase2 .

docker run --rm -v "$PWD":/app tc-mle-fase2 dvc update data/raw/online_shoppers_intention.zip.dvc
docker run --rm -v "$PWD":/app tc-mle-fase2                  # CMD = dvc repro
docker run --rm -v "$PWD":/app tc-mle-fase2 dvc exp run -f -S configs/params.yaml:train.model=dummy
docker run --rm -v "$PWD":/app tc-mle-fase2 dvc exp run -f -S configs/params.yaml:train.model=logistic_regression
docker run --rm -v "$PWD":/app tc-mle-fase2 dvc exp run -f -S configs/params.yaml:train.model=hist_gradient_boosting
docker run --rm -v "$PWD":/app -e USER="$(whoami)" tc-mle-fase2 promote
docker run --rm -v "$PWD":/app tc-mle-fase2 predict
```

- **`--build-arg UID`:** o usuário da imagem (sem privilégio) precisa ser o dono do
  repositório montado para escrever nele. O padrão é 1000. Trocar só o `--user` do
  `docker run` não serve, porque o `dvc exp run` falha num UID sem entrada no `/etc/passwd`.
- **`-e USER`:** sem ele, o `approved_by` da promoção registraria o usuário da imagem.
- **Banco próprio do container:** dentro da imagem, `MLFLOW_TRACKING_URI` aponta para
  `mlflow-docker.db`, e não para o `mlflow.db` do host. O MLflow grava o caminho absoluto de
  cada artefato, e um banco usado pelos dois lados quebra nos dois. Para ver o Registry criado
  pelo container: `poetry run mlflow ui --backend-store-uri sqlite:///mlflow-docker.db`.

O CI builda a imagem em x86_64 e roda o pipeline dentro dela sobre uma amostra de 200 linhas
do dataset (`tests/fixtures/`), sem baixar nada da UCI.

## Pipeline

```
+--------------------------------------------+
| data/raw/online_shoppers_intention.zip.dvc |
+--------------------------------------------+
                       *
                  +---------+
                  | extract |
                  +---------+
                       *
                  +---------+
                  | prepare |
                  +---------+
                  *         *
         +-------+           *
         | train |           *
         +-------+           *
                  *         *
                 +----------+
                 | evaluate |
                 +----------+
```

- **`extract`**: grava o CSV de dentro do zip, byte a byte. É em Python porque a imagem
  `python:*-slim` não tem `unzip`.
- **`prepare`**: seleciona as colunas do esquema, converte o alvo `Revenue` para 0/1 e separa
  treino e teste (80/20, estratificado). Não ajusta nada: o one-hot e o escalonamento ficam
  dentro do `Pipeline` do scikit-learn e são ajustados só no treino, em cada dobra da CV. Por
  isso o estágio não se chama `preprocess`.
- **`train`**: validação cruzada estratificada de 5 dobras no treino, ajuste final, modelo em
  `models/model.joblib`, métricas em `metrics/cv.json` e um run no MLflow. Não lê o
  `test.csv`, e o `dvc.yaml` mostra isso nas `deps`.
- **`evaluate`**: mede o modelo no teste, uma vez. Lê o `model.joblib`, que é `dep`, e não o
  alias do Registry: mover o alias não invalidaria o cache do DVC.

Fora do DVC ficam o `promote` e o `predict`, que trabalham sobre o Registry.

Todo número que muda um resultado está em `configs/params.yaml`, e cada estágio declara no
`dvc.yaml` o bloco que lê e todos os módulos que importa (um teste confere os imports). O
critério de promoção também mora no `params.yaml`, commitado antes dos runs.

## Resultados

| Modelo | PR-AUC (CV, média ± desvio) | ROC-AUC (CV) |
|---|---|---|
| `dummy` (piso) | 0,155 | 0,500 |
| Regressão logística | 0,646 ± 0,017 | 0,894 |
| **HistGradientBoosting (champion v1)** | **0,747 ± 0,012** | **0,931** |

No teste: **PR-AUC 0,716** e ROC-AUC 0,926, com prevalência de 0,155, que é o PR-AUC de um
modelo sem informação. A métrica principal é o PR-AUC porque a classe positiva é minoritária.
Não há métrica com limiar (precisão, recall, F1): escolher o limiar olhando o teste seria usar
o teste para decidir, e transformar a probabilidade em sim/não é decisão de negócio.

**Critério de promoção:** entram só runs do commit atual, com o mesmo `data.md5` e código sem
alteração. O melhor candidato precisa superar o `dummy` e o champion vigente por mais de 1
desvio da sua própria CV. Uma diferença dentro do ruído não promove.

## Decisões de projeto

### Funcional nos estágios, objetos onde há estado

Cada estágio tem um núcleo de funções puras, testável sem arquivo, e um `main()` curto que lê
os parâmetros, carrega os dados, chama o núcleo e grava o resultado. I/O e MLflow ficam só no
`main()`.

A divisão em pipeline de dados, treino e avaliação é a mesma de um desenho com classes
`DataPipeline`, `ModelTrainer` e `ModelEvaluator`. Muda o veículo: o DVC roda cada estágio num
processo próprio, então um objeto não sobreviveria de um estágio para o outro, e o estado passa
por arquivos versionados (`outs` → `deps`).

Orientação a objetos entra onde o estado é real:

- o modelo é um `Pipeline` do scikit-learn, com o pré-processamento ajustado dentro dele;
- os estimadores são intercambiáveis por `fit`/`predict` (Strategy), escolhidos por nome no
  `params.yaml` a partir de um dicionário de construtores (Factory);
- os parâmetros são carregados uma vez para uma `dataclass` imutável e validados na carga.

Não há herança no projeto.

### Reprodutibilidade

Um treino depende do código, do dado, dos parâmetros (as seeds estão neles) e do ambiente. No
repositório, são o `src/`, o `.dvc` do dataset, o `configs/params.yaml` e o
`poetry.lock` + `.python-version`. Com os quatro fixos, o `dvc repro -f` gera um `dvc.lock`
byte a byte igual, inclusive o hash do modelo, e o `dvc.lock` gerado no Linux do container é
o mesmo do macOS em que foi commitado.

Cada run do MLflow leva o commit do código (`git.commit`, mais `git.dirty` sobre `src/`,
`pyproject.toml`, `poetry.lock` e `dvc.yaml`), o MD5 do `train.csv` (o mesmo do `dvc.lock`) e o
nome do experimento do DVC.

### Convenções

- Identificadores em inglês; docstrings, comentários e documentação em português.
- Dados por estágio: `data/raw/` (origem), `data/processed/` (`train.csv`, `test.csv`).
- Hiperparâmetros e seeds: só em `configs/params.yaml`. O `.env` guarda o que muda onde o
  projeto roda (`MLFLOW_TRACKING_URI`), nunca o que muda o modelo, porque o DVC não rastreia
  o `.env`. Copie `.env.example` para `.env` se quiser outro destino para o MLflow.
- Modelos: o do pipeline em `models/model.joblib` (versionado pelo DVC); os candidatos e o
  champion no MLflow, nomeados pelo alias, nunca por arquivo.
- Lint com ruff (naming, type hints, docstrings, tamanho e complexidade de função), no CI.

O registro completo das decisões, com as alternativas descartadas, está em
[`docs/decision-log.md`](docs/decision-log.md).

## Limitações

- **Teste abaixo da CV** (0,716 contra 0,747 ± 0,012). O desvio da CV mede a variação entre
  dobras, não o otimismo de escolher o melhor de três candidatos pela mesma CV, e o teste é
  um único split de 2.466 sessões. O modelo não foi trocado por causa do teste, o que seria
  escolher pelo teste.
- **Sem desempate por simplicidade.** Dois candidatos dentro de 1 desvio não promovem um ao
  outro, mas não há regra que prefira o mais simples quando ambos superam o champion.
- **Um ano de dados, sem split temporal.** A conversão tem pico em novembro, e o modelo não
  foi testado num período que não viu.
- **`PageValues`** depende de como o Google Analytics atribui valor às páginas, então a relação
  com a compra pode mudar sem nenhuma mudança de código. Ele fica no modelo; o experimento sem
  ele é `dvc exp run -f -S configs/params.yaml:train.use_page_values=false`.
- **Registry local.** O champion existe no `mlflow.db` de quem roda os comandos; quem clona
  recria os runs e a promoção, não recebe os do autor.
- O pipeline depende de a UCI estar no ar para baixar o dado (caminho manual acima).

## Estrutura

```
configs/params.yaml        parâmetros do pipeline e critério de promoção
dvc.yaml, dvc.lock         pipeline e estado reproduzível
data/raw/*.dvc             ponteiro do dataset (import da UCI)
src/purchase_propensity/   pacote: um módulo por estágio + models, params, schema,
                           tracking, promotion (critério puro), promote, predict
tests/                     testes unitários, contra MLflow em SQLite temporário e smoke
                           do pipeline sobre tests/fixtures/
Dockerfile, .dockerignore  imagem multi-stage via Poetry
docs/decision-log.md       decisões, alternativas e porquês
```
