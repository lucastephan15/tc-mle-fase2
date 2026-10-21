import numpy as np
import pandas as pd
import pytest

from purchase_propensity.prepare import select_and_cast, split_train_test
from purchase_propensity.schema import (
    BOOLEAN_FEATURES,
    CATEGORICAL_FEATURES,
    FEATURE_COLUMNS,
    NUMERIC_FEATURES,
    TARGET_COLUMN,
)

N_ROWS = 200
POSITIVE_RATE = 0.15


@pytest.fixture
def raw() -> pd.DataFrame:
    rng = np.random.default_rng(0)
    data = {col: rng.random(N_ROWS) for col in NUMERIC_FEATURES}
    data |= {col: rng.integers(1, 5, N_ROWS) for col in CATEGORICAL_FEATURES}
    data |= {col: rng.random(N_ROWS) < 0.5 for col in BOOLEAN_FEATURES}
    data[TARGET_COLUMN] = np.arange(N_ROWS) < N_ROWS * POSITIVE_RATE
    data["ExtraColumn"] = 0
    return pd.DataFrame(data)


def test_alvo_sai_como_inteiro_zero_ou_um(raw):
    target = select_and_cast(raw)[TARGET_COLUMN]
    assert target.dtype == "int64"
    assert set(target.unique()) == {0, 1}


def test_mantem_so_as_colunas_do_esquema_na_ordem(raw):
    assert list(select_and_cast(raw).columns) == [*FEATURE_COLUMNS, TARGET_COLUMN]


def test_select_and_cast_nao_altera_a_entrada(raw):
    before = raw.copy()
    select_and_cast(raw)
    pd.testing.assert_frame_equal(raw, before)


def test_split_nao_perde_nem_repete_linhas(raw):
    train, test = split_train_test(select_and_cast(raw), test_size=0.2, seed=42)
    assert len(train) + len(test) == N_ROWS
    assert train.index.intersection(test.index).empty


def test_split_preserva_a_prevalencia(raw):
    train, test = split_train_test(select_and_cast(raw), test_size=0.2, seed=42)
    assert train[TARGET_COLUMN].mean() == pytest.approx(POSITIVE_RATE, abs=0.01)
    assert test[TARGET_COLUMN].mean() == pytest.approx(POSITIVE_RATE, abs=0.01)


def test_mesma_seed_mesmo_split(raw):
    data = select_and_cast(raw)
    first, _ = split_train_test(data, test_size=0.2, seed=42)
    second, _ = split_train_test(data, test_size=0.2, seed=42)
    pd.testing.assert_frame_equal(first, second)
