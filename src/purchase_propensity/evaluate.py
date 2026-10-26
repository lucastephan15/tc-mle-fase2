"""Estágio `evaluate`: mede o modelo do `dvc.lock` no conjunto de teste.

O teste não escolhe nada. O modelo já foi escolhido pela CV no treino e pelo `promote`;
aqui só se mede quanto ele vale em linhas que nenhuma etapa viu antes (decision log §3).
Por isso as métricas são só de ranking: escolher um limiar de decisão olhando o teste
seria usá-lo para decidir (estudo do dataset, §9).

O modelo vem de `models/model.joblib`, que é `dep` do estágio, e não do alias do
Registry: mover o alias não invalida o cache do DVC, e o `dvc repro` diria "up to date"
sobre o resultado do modelo antigo.
"""

import json

import joblib
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.pipeline import Pipeline

from purchase_propensity.schema import (
    MODEL_PATH,
    TARGET_COLUMN,
    TEST_DATA_PATH,
    TEST_METRICS_PATH,
)

# Os mesmos nomes do `cv.json`: `average_precision` é o PR-AUC que a CV mede no `train`.
METRICS = {"pr_auc": average_precision_score, "roc_auc": roc_auc_score}


def evaluate_model(pipeline: Pipeline, test: pd.DataFrame) -> dict[str, float]:
    """Calcula as métricas de ranking do modelo no teste.

    Args:
        pipeline: Pipeline já ajustado.
        test: Conjunto de teste, com o alvo.

    Returns:
        PR-AUC e ROC-AUC, mais a prevalência: é o PR-AUC de um modelo sem informação, a
        referência para ler o PR-AUC.
    """
    X, y = test.drop(columns=TARGET_COLUMN), test[TARGET_COLUMN]
    scores = pipeline.predict_proba(X)[:, 1]
    metrics = {name: metric(y, scores) for name, metric in METRICS.items()}
    metrics["prevalence"] = y.mean()
    return {name: round(float(value), 6) for name, value in metrics.items()}


def main() -> None:
    """Lê o modelo e o teste e grava as métricas em `metrics/test.json`."""
    metrics = evaluate_model(joblib.load(MODEL_PATH), pd.read_csv(TEST_DATA_PATH))
    TEST_METRICS_PATH.parent.mkdir(parents=True, exist_ok=True)
    TEST_METRICS_PATH.write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
