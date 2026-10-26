"""Estágio `train`: ajusta o pipeline no treino e mede o desempenho por validação cruzada.

Lê só `data/processed/train.csv`. O conjunto de teste fica para um estágio de avaliação
que roda uma única vez, e não aparece nas `deps` deste estágio (decision log §3).

Todo o pré-processamento mora dentro do `Pipeline`, então ele é reajustado em cada dobra
da validação cruzada, só com as linhas de treino daquela dobra, e o mesmo objeto que é
avaliado é o que vai para `models/`.
"""

import json

import joblib
import numpy as np
import pandas as pd
from dotenv import load_dotenv
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from purchase_propensity.models import build_model
from purchase_propensity.params import (
    TrainParams,
    load_prepare_params,
    load_train_params,
)
from purchase_propensity.schema import (
    BOOLEAN_FEATURES,
    CATEGORICAL_FEATURES,
    CV_METRICS_PATH,
    MODEL_PATH,
    NUMERIC_FEATURES,
    TARGET_COLUMN,
    TRAIN_DATA_PATH,
)
from purchase_propensity.tracking import (
    ENV_PATH,
    flatten_params,
    log_training_run,
    run_tags,
)

PAGE_VALUES = "PageValues"
# Nome no relatório → scorer do scikit-learn. PR-AUC é a métrica principal (estudo do
# dataset, D4): com 15,5% de positivos, o ROC-AUC fica alto mesmo com ranking medíocre.
SCORERS = {"pr_auc": "average_precision", "roc_auc": "roc_auc"}


def numeric_features(use_page_values: bool) -> list[str]:
    """Lista as features numéricas, com ou sem `PageValues`.

    Args:
        use_page_values: Se `False`, remove o quase-vazamento (estudo do dataset, §4.1).

    Returns:
        Os nomes das colunas numéricas, na ordem do esquema.
    """
    return [c for c in NUMERIC_FEATURES if use_page_values or c != PAGE_VALUES]


def build_pipeline(params: TrainParams) -> Pipeline:
    """Monta pré-processamento + estimador num único objeto ajustável.

    Args:
        params: Parâmetros do estágio.

    Returns:
        O `Pipeline` ainda não ajustado. Colunas fora do esquema são descartadas.
    """
    encoder = OneHotEncoder(
        handle_unknown="infrequent_if_exist",
        min_frequency=params.min_frequency,
        sparse_output=False,
    )
    preprocessor = ColumnTransformer(
        [
            ("numeric", StandardScaler(), numeric_features(params.use_page_values)),
            ("categorical", encoder, list(CATEGORICAL_FEATURES)),
            ("boolean", "passthrough", list(BOOLEAN_FEATURES)),
        ],
        remainder="drop",
    )
    model = build_model(params.model, params.seed)
    return Pipeline([("preprocess", preprocessor), ("model", model)])


def cross_validate_pipeline(
    pipeline: Pipeline, X: pd.DataFrame, y: pd.Series, params: TrainParams
) -> dict[str, float]:
    """Mede o pipeline por validação cruzada estratificada.

    Devolve média e desvio-padrão de cada métrica: a comparação entre candidatos olha a
    dispersão entre as dobras, não só a média.

    Args:
        pipeline: Pipeline ainda não ajustado.
        X: Features do treino.
        y: Alvo do treino.
        params: Parâmetros do estágio (dobras e semente).

    Returns:
        Um dicionário `{métrica}_mean` / `{métrica}_std` com floats arredondados.
    """
    folds = StratifiedKFold(
        n_splits=params.cv_folds, shuffle=True, random_state=params.seed
    )
    scores = cross_validate(pipeline, X, y, cv=folds, scoring=SCORERS)
    return {
        f"{name}_{stat}": round(float(fn(scores[f"test_{name}"])), 6)
        for name in SCORERS
        for stat, fn in (("mean", np.mean), ("std", np.std))
    }


def main() -> None:
    """Valida o pipeline por CV, ajusta no treino inteiro, grava e registra no MLflow.

    O modelo em `models/` e o `cv.json` são as saídas que o DVC versiona; o run no MLflow
    é o histórico que permite comparar execuções e escolher o campeão.
    """
    # Sem `override`: uma variável já definida no shell ou no container prevalece.
    load_dotenv(ENV_PATH)
    params = load_train_params()
    train = pd.read_csv(TRAIN_DATA_PATH)
    X, y = train.drop(columns=TARGET_COLUMN), train[TARGET_COLUMN]
    pipeline = build_pipeline(params)
    metrics = cross_validate_pipeline(pipeline, X, y, params)
    pipeline.fit(X, y)
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, MODEL_PATH)
    CV_METRICS_PATH.parent.mkdir(parents=True, exist_ok=True)
    CV_METRICS_PATH.write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
    log_training_run(
        pipeline,
        X,
        params=flatten_params({"prepare": load_prepare_params(), "train": params}),
        metrics=metrics,
        tags=run_tags(TRAIN_DATA_PATH),
    )
