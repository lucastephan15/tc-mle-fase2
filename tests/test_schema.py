import pandas as pd
import pytest

from purchase_propensity.schema import (
    BOOLEAN_FEATURES,
    CATEGORICAL_FEATURES,
    FEATURE_COLUMNS,
    NUMERIC_FEATURES,
    RAW_DATA_PATH,
    TARGET_COLUMN,
)


def test_cada_feature_pertence_a_um_unico_grupo():
    groups = [set(NUMERIC_FEATURES), set(CATEGORICAL_FEATURES), set(BOOLEAN_FEATURES)]
    assert sum(len(g) for g in groups) == len(set().union(*groups))


def test_feature_columns_nao_tem_repeticao():
    assert len(FEATURE_COLUMNS) == len(set(FEATURE_COLUMNS))


def test_alvo_nao_entra_nas_features():
    assert TARGET_COLUMN not in FEATURE_COLUMNS


@pytest.mark.skipif(
    not RAW_DATA_PATH.exists(),
    reason="CSV bruto ausente; passa a ser obrigatório quando o DVC entrar (Etapa 3)",
)
def test_esquema_cobre_exatamente_as_colunas_do_csv():
    header = pd.read_csv(RAW_DATA_PATH, nrows=0).columns
    assert set(header) == {*FEATURE_COLUMNS, TARGET_COLUMN}
