"""Comando `predict`: pontua um CSV com o modelo que o alias do Registry aponta.

É a predição em batch que o enunciado pede no lugar de uma API (decision log §0). Fica
fora do `dvc.yaml` porque lê o Registry: o alias não é `dep`, e o DVC não saberia que o
modelo mudou. O resultado é uma probabilidade, a propensão de compra; transformá-la em
sim/não exige um limiar, decisão de negócio que não sai do teste (estudo do dataset, §9).
"""

import argparse
from pathlib import Path

import mlflow
import pandas as pd
from dotenv import load_dotenv
from mlflow.tracking import MlflowClient
from sklearn.pipeline import Pipeline

from purchase_propensity.params import load_promotion_params
from purchase_propensity.schema import FEATURE_COLUMNS, TEST_DATA_PATH
from purchase_propensity.tracking import ENV_PATH

PREDICTIONS_PATH = Path("predictions/predictions.csv")
SCORE_COLUMN = "purchase_probability"


def score(pipeline: Pipeline, data: pd.DataFrame) -> pd.DataFrame:
    """Acrescenta a probabilidade de compra a cada linha.

    Args:
        pipeline: Pipeline já ajustado.
        data: Linhas a pontuar, com as colunas do esquema. Colunas a mais (o alvo, um id)
            são mantidas na saída e não entram no modelo.

    Returns:
        Uma cópia de `data` com a coluna `purchase_probability`.

    Raises:
        ValueError: Se faltar alguma coluna do esquema.
    """
    missing = [col for col in FEATURE_COLUMNS if col not in data.columns]
    if missing:
        msg = f"colunas ausentes no CSV de entrada: {missing}"
        raise ValueError(msg)
    probability = pipeline.predict_proba(data.loc[:, FEATURE_COLUMNS])[:, 1]
    return data.assign(**{SCORE_COLUMN: probability})


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Lê os caminhos de entrada e saída da linha de comando.

    Args:
        argv: Argumentos; `None` usa os do processo.

    Returns:
        `input` e `output` como `Path`.
    """
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--input", type=Path, default=TEST_DATA_PATH)
    parser.add_argument("--output", type=Path, default=PREDICTIONS_PATH)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    """Carrega o champion do Registry, pontua o CSV e grava o resultado.

    Args:
        argv: Argumentos da linha de comando; `None` usa os do processo.
    """
    load_dotenv(ENV_PATH)
    args = parse_args(argv)
    params = load_promotion_params()
    version = MlflowClient().get_model_version_by_alias(params.model_name, params.alias)
    pipeline = mlflow.sklearn.load_model(f"models:/{params.model_name}@{params.alias}")

    scored = score(pipeline, pd.read_csv(args.input))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    scored.to_csv(args.output, index=False)
    print(
        f"{len(scored)} linhas pontuadas com {params.model_name} v{version.version} "
        f"(@{params.alias}) → {args.output}"
    )
