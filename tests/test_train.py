import pandas as pd
import pytest

from purchase_propensity.models import MODEL_FACTORIES
from purchase_propensity.params import TrainParams
from purchase_propensity.schema import (
    NUMERIC_FEATURES,
    TARGET_COLUMN,
)
from purchase_propensity.train import (
    PAGE_VALUES,
    build_pipeline,
    cross_validate_pipeline,
    numeric_features,
)


def make_params(**overrides) -> TrainParams:
    defaults = {
        "model": "logistic_regression",
        "use_page_values": True,
        "min_frequency": 5,
        "cv_folds": 3,
        "seed": 42,
    }
    return TrainParams(**(defaults | overrides))


def split_xy(data: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    return data.drop(columns=TARGET_COLUMN), data[TARGET_COLUMN]


def test_sem_page_values_remove_so_essa_coluna():
    assert PAGE_VALUES in numeric_features(use_page_values=True)
    without = numeric_features(use_page_values=False)
    assert PAGE_VALUES not in without
    assert len(without) == len(NUMERIC_FEATURES) - 1


@pytest.mark.parametrize("model", sorted(MODEL_FACTORIES))
def test_todo_candidato_ajusta_e_devolve_probabilidade(train, model):
    X, y = split_xy(train)
    proba = build_pipeline(make_params(model=model)).fit(X, y).predict_proba(X)
    assert proba.shape == (len(train), 2)


def test_page_values_nao_chega_ao_modelo_quando_desligado(train):
    X, y = split_xy(train)
    pipeline = build_pipeline(make_params(use_page_values=False)).fit(X, y)
    names = pipeline.named_steps["preprocess"].get_feature_names_out()
    assert not any(PAGE_VALUES in name for name in names)


def test_categoria_nunca_vista_nao_quebra_a_predicao(train):
    X, y = split_xy(train)
    pipeline = build_pipeline(make_params()).fit(X, y)
    unseen = X.head(3).assign(Month="Jan", Region=99)
    assert pipeline.predict_proba(unseen).shape == (3, 2)


def test_coluna_codigo_vira_one_hot_e_nao_grandeza(train):
    X, y = split_xy(train)
    pipeline = build_pipeline(make_params()).fit(X, y)
    names = pipeline.named_steps["preprocess"].get_feature_names_out()
    assert "numeric__Region" not in names
    assert any(name.startswith("categorical__Region_") for name in names)


def test_cv_devolve_media_e_desvio_de_cada_metrica(train):
    X, y = split_xy(train)
    metrics = cross_validate_pipeline(build_pipeline(make_params()), X, y, make_params())
    assert set(metrics) == {"pr_auc_mean", "pr_auc_std", "roc_auc_mean", "roc_auc_std"}
    assert all(0 <= value <= 1 for value in metrics.values())


def test_mesma_seed_mesmas_metricas(train):
    X, y = split_xy(train)
    params = make_params(model="random_forest")
    first = cross_validate_pipeline(build_pipeline(params), X, y, params)
    second = cross_validate_pipeline(build_pipeline(params), X, y, params)
    assert first == second


def test_treino_nao_altera_a_entrada(train):
    X, y = split_xy(train)
    before = X.copy()
    build_pipeline(make_params()).fit(X, y)
    pd.testing.assert_frame_equal(X, before)
