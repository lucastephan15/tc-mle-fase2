import numpy as np
import pytest

from purchase_propensity.predict import SCORE_COLUMN, parse_args, score
from purchase_propensity.schema import TARGET_COLUMN


def test_acrescenta_a_probabilidade_e_mantem_as_colunas(fitted, train):
    data = train.assign(session_id=range(len(train)))
    scored = score(fitted, data)
    assert list(scored.columns) == [*data.columns, SCORE_COLUMN]
    assert scored[SCORE_COLUMN].between(0, 1).all()


def test_coluna_a_mais_nao_muda_a_probabilidade(fitted, train):
    plain = score(fitted, train.drop(columns=TARGET_COLUMN))[SCORE_COLUMN]
    extra = score(fitted, train.assign(session_id=1))[SCORE_COLUMN]
    np.testing.assert_allclose(plain, extra)


def test_coluna_do_esquema_ausente_falha_com_o_nome(fitted, train):
    with pytest.raises(ValueError, match="PageValues"):
        score(fitted, train.drop(columns="PageValues"))


def test_caminhos_padrao_e_da_linha_de_comando():
    assert parse_args([]).input.name == "test.csv"
    assert str(parse_args(["--input", "novo.csv"]).input) == "novo.csv"
