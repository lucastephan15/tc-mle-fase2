import pytest

from purchase_propensity.params import (
    PARAMS_PATH,
    PrepareParams,
    TrainParams,
    load_prepare_params,
    load_train_params,
)


def test_params_do_repositorio_carregam():
    assert isinstance(load_prepare_params(PARAMS_PATH), PrepareParams)


@pytest.mark.parametrize("test_size", [0, 1, -0.1, 1.5])
def test_test_size_fora_da_faixa_falha_na_carga(test_size):
    with pytest.raises(ValueError, match="test_size"):
        PrepareParams(test_size=test_size, seed=42)


def test_seed_negativa_falha_na_carga():
    with pytest.raises(ValueError, match="seed"):
        PrepareParams(test_size=0.2, seed=-1)


def test_chave_digitada_errada_falha_na_carga(tmp_path):
    path = tmp_path / "params.yaml"
    path.write_text("prepare:\n  test_sise: 0.2\n  seed: 42\n", encoding="utf-8")
    with pytest.raises(TypeError):
        load_prepare_params(path)


TRAIN_DEFAULTS = {
    "model": "logistic_regression",
    "use_page_values": True,
    "min_frequency": 10,
    "cv_folds": 5,
    "seed": 42,
}


def test_params_de_train_do_repositorio_carregam():
    assert isinstance(load_train_params(PARAMS_PATH), TrainParams)


@pytest.mark.parametrize(
    ("key", "value"),
    [("model", "xgboost"), ("min_frequency", 0), ("cv_folds", 1), ("seed", -1)],
)
def test_train_params_invalidos_falham_na_carga(key, value):
    with pytest.raises(ValueError, match=key):
        TrainParams(**(TRAIN_DEFAULTS | {key: value}))
