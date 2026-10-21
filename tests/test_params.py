import pytest

from purchase_propensity.params import PARAMS_PATH, PrepareParams, load_prepare_params


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
