# Imagem = ambiente (Python + dependências do poetry.lock). O projeto (código, dvc.yaml,
# dados, mlflow.db) entra por bind mount do repositório em /app, para que as saídas do
# pipeline fiquem no host e no `git status` (decision log §3).
#
#   docker build -t tc-mle-fase2:1.0.0 .
#   docker run --rm -v "$PWD":/app tc-mle-fase2:1.0.0

# Mesmo patch do .python-version: a imagem não pode ter outro Python que o lock.
ARG PYTHON_IMAGE=python:3.12.5-slim

# --- build: o Poetry só existe aqui ----------------------------------------------
FROM ${PYTHON_IMAGE} AS build
# Mesma versão do Poetry da máquina de desenvolvimento e do CI: o formato do lock depende dela.
RUN pip install --no-cache-dir poetry==2.5.1 && python -m venv /opt/venv
# Com um venv ativo, o Poetry instala nele em vez de criar o seu. O venv fica fora de /app
# porque o bind mount do repositório cobriria um /app/.venv.
ENV VIRTUAL_ENV=/opt/venv \
    PATH=/opt/venv/bin:$PATH
WORKDIR /app
# Lock antes do código: mudar o código não reinstala as dependências.
COPY pyproject.toml poetry.lock README.md ./
RUN poetry install --only main --no-root --no-interaction
COPY src/ ./src/
# Instala só o próprio pacote, em modo editável: os executáveis (`train`, `promote`, ...)
# importam de /app/src, que com o bind mount é o código do host, o mesmo que o DVC rastreia.
RUN poetry install --only main --no-interaction

# --- runtime: venv + código, sem Poetry, usuário sem privilégio --------------------------
FROM ${PYTHON_IMAGE} AS runtime
# UID 1000 = o primeiro usuário na maioria dos Linux: com o bind mount, os arquivos gerados
# saem com o dono de quem rodou, e não como root.
RUN adduser --system --group --uid 1000 mluser \
    # O `dvc exp` usa o libgit2, que recusa repositório cujo dono não é o usuário do
    # processo. No bind mount o dono é o do host (no Docker Desktop, /app aparece como
    # root), e o `dvc exp run` falhava com "repository path '/app' is not owned by current
    # user". A exceção vale só para /app, e mora na config de sistema, que o libgit2 lê
    # sem o binário `git`. O `dvc exp run` também grava cada experimento como commit numa
    # ref própria (refs/exps), e sem autor ele falha: a identidade abaixo assina só essas
    # refs. O commit do projeto continua sendo feito no host, com o autor de quem commita.
    && printf '[safe]\n\tdirectory = /app\n[user]\n\tname = tc-mle-fase2 container\n\temail = container@tc-mle-fase2.local\n' \
        > /etc/gitconfig
# O venv mantém o caminho: os scripts em bin/ têm o Python de /opt/venv no shebang.
COPY --from=build --chown=mluser:mluser /opt/venv /opt/venv
WORKDIR /app
# O código também vai na imagem, para que ela rode sem o mount (`docker run img train --help`).
COPY --chown=mluser:mluser src/ ./src/
# MLFLOW_TRACKING_URI: banco próprio do container. O MLflow grava no banco o caminho
# ABSOLUTO dos artefatos (/app/mlruns/... aqui, /Users/... no host), então um banco usado
# pelos dois lados quebra: o `train` no container não consegue criar /Users, e o `predict`
# no host não acha /app. Com bancos separados, cada lado só lê o que ele mesmo gravou. O
# `.env` não sobrescreve este valor (o `load_dotenv` roda sem `override`).
# GIT_PYTHON_REFRESH: o MLflow tenta o GitPython, que procura o binário `git` (ausente na
# slim) e imprime um aviso de 20 linhas por run. O commit do run vem do `tracking.py`.
ENV PATH=/opt/venv/bin:$PATH \
    MLFLOW_TRACKING_URI=sqlite:///mlflow-docker.db \
    GIT_PYTHON_REFRESH=quiet \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1
USER mluser
CMD ["dvc", "repro"]
