"""Registro de cada execução do `train` no MLflow.

O DVC decide **se** o estágio roda (memoização por hash); o MLflow guarda **o que** cada
execução deu. Este módulo só registra: nenhum modelo é registrado no Registry aqui, porque
quem escolhe o campeão é o `promote`, depois de comparar os runs (decision log §4).

As tags amarram o run às três entradas que o definem além dos params: o código
(`git.commit` + `git.dirty`), o dado (`data.md5`) e o experimento do DVC (`dvc.exp`).
"""

import hashlib
import os
import warnings
from collections.abc import Mapping
from dataclasses import asdict
from pathlib import Path
from typing import Any

import mlflow
import mlflow.sklearn
import pandas as pd
from dulwich import porcelain
from dulwich.errors import NotGitRepository
from dulwich.repo import Repo
from sklearn.pipeline import Pipeline

# Relativo à pasta de execução, como os caminhos do schema.py. O `load_dotenv()` sem
# argumento procura a partir do arquivo .py que o chama, e numa instalação que não seja
# editável (o pacote dentro de site-packages) não acharia o `.env` do projeto.
ENV_PATH = Path(".env")
EXPERIMENT_NAME = "purchase-propensity"
UNAVAILABLE = "unavailable"
# O que conta como "código" para o `git.dirty`. O `params.yaml` fica de fora de propósito:
# o `dvc exp run -S` o reescreve antes de rodar, e os params já vão para o run como params
# (D04-A6 §3). O dado também: vai como `data.md5`.
CODE_PATHS = ("src/", "pyproject.toml", "poetry.lock", "dvc.yaml")
INPUT_EXAMPLE_ROWS = 5


def file_md5(path: Path) -> str:
    """Calcula o MD5 de um arquivo, o mesmo hash que o DVC grava no `dvc.lock`.

    Args:
        path: Arquivo a ler.

    Returns:
        O hash em hexadecimal.
    """
    return hashlib.md5(path.read_bytes(), usedforsecurity=False).hexdigest()


def _changed_paths(status: porcelain.GitStatus) -> list[str]:
    staged = [path for paths in status.staged.values() for path in paths]
    return [path.decode() for path in (*staged, *status.unstaged, *status.untracked)]


def git_state(repo_path: Path = Path()) -> tuple[str, str]:
    """Lê o commit atual e se o código difere dele, sem depender do binário `git`.

    Usa o `dulwich` (git em Python puro), porque a imagem slim não tem o `git`. Sem
    repositório, por exemplo uma imagem rodada sem o projeto montado, os dois valores
    saem `unavailable`: o run é registrado, mas não pode ser promovido.

    Args:
        repo_path: Pasta dentro do repositório.

    Returns:
        O hash do commit e `"true"`/`"false"` para mudança não commitada em `CODE_PATHS`.
    """
    try:
        repo = Repo.discover(str(repo_path))
    except NotGitRepository:
        return UNAVAILABLE, UNAVAILABLE
    with repo:
        commit = repo.head().decode()
        changed = _changed_paths(porcelain.status(repo))
    dirty = any(path.startswith(CODE_PATHS) for path in changed)
    return commit, str(dirty).lower()


def run_tags(data_path: Path) -> dict[str, str]:
    """Monta as tags que identificam código, dado e experimento do run.

    Args:
        data_path: O arquivo de treino que o estágio leu.

    Returns:
        As tags `git.commit`, `git.dirty`, `data.md5` e `dvc.exp`. A última fica vazia num
        `dvc repro` simples: o DVC só exporta `DVC_EXP_NAME` dentro de um `dvc exp run`.
    """
    commit, dirty = git_state()
    return {
        "git.commit": commit,
        "git.dirty": dirty,
        "data.md5": file_md5(data_path),
        "dvc.exp": os.environ.get("DVC_EXP_NAME", ""),
    }


def flatten_params(blocks: Mapping[str, Any]) -> dict[str, Any]:
    """Achata os blocos do `params.yaml` em chaves `bloco.chave`.

    Args:
        blocks: Nome do bloco → dataclass de parâmetros já validada.

    Returns:
        Um dicionário plano, no formato que o `mlflow.log_params` espera.
    """
    return {
        f"{block}.{key}": value
        for block, params in blocks.items()
        for key, value in asdict(params).items()
    }


def log_training_run(
    pipeline: Pipeline,
    X: pd.DataFrame,
    params: Mapping[str, Any],
    metrics: Mapping[str, float],
    tags: Mapping[str, str],
) -> str:
    """Registra params, métricas da CV, tags e o pipeline ajustado num run novo.

    O destino vem de `MLFLOW_TRACKING_URI`. Sem a variável, o MLflow usa o padrão dele,
    `sqlite:///mlflow.db` na pasta de execução.

    Args:
        pipeline: Pipeline já ajustado no treino inteiro.
        X: Features do treino. As primeiras linhas viram o exemplo de entrada, de onde o
            MLflow infere a assinatura do modelo.
        params: Parâmetros achatados (`flatten_params`).
        metrics: Métricas da validação cruzada.
        tags: Tags do run (`run_tags`).

    Returns:
        O id do run criado.
    """
    mlflow.set_experiment(EXPERIMENT_NAME)
    with mlflow.start_run() as run:
        mlflow.log_params(params)
        mlflow.log_metrics({f"cv_{name}": value for name, value in metrics.items()})
        mlflow.set_tags(tags)
        with warnings.catch_warnings():
            # O aviso diz que coluna inteira não aceita valor faltante. É o contrato
            # pretendido: o dataset não tem faltantes e o pipeline não imputa.
            warnings.filterwarnings(
                "ignore", message="Hint: Inferred schema contains integer"
            )
            mlflow.sklearn.log_model(
                pipeline, name="model", input_example=X.head(INPUT_EXAMPLE_ROWS)
            )
    return run.info.run_id
