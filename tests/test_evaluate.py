import pytest

from purchase_propensity import train as train_stage
from purchase_propensity.evaluate import evaluate_model
from purchase_propensity.params import TrainParams
from purchase_propensity.schema import TARGET_COLUMN


def test_metricas_de_ranking_e_prevalencia(fitted, train):
    metrics = evaluate_model(fitted, train)
    assert set(metrics) == {"pr_auc", "roc_auc", "prevalence"}
    assert metrics["prevalence"] == pytest.approx(train[TARGET_COLUMN].mean())


def test_modelo_sem_informacao_tem_pr_auc_igual_a_prevalencia(train):
    dummy = train_stage.build_pipeline(TrainParams("dummy", True, 10, 5, 7))
    dummy.fit(train.drop(columns=TARGET_COLUMN), train[TARGET_COLUMN])
    metrics = evaluate_model(dummy, train)
    assert metrics["pr_auc"] == pytest.approx(metrics["prevalence"])
    assert metrics["roc_auc"] == pytest.approx(0.5)


def test_evaluate_nao_altera_a_entrada(fitted, train):
    before = train.copy()
    evaluate_model(fitted, train)
    assert train.equals(before)
