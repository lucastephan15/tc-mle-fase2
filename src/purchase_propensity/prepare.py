"""Estágio `prepare`: do CSV bruto aos conjuntos de treino e teste.

Só converte tipos e separa as linhas. Nada é ajustado aqui: one-hot e escalonamento ficam
dentro do `Pipeline` do scikit-learn, no estágio `train`, para que nenhuma estatística do
teste vaze para o treino. Por isso o nome é `prepare`, e não `preprocess`
(decision log §3).
"""

import pandas as pd
from sklearn.model_selection import train_test_split

from purchase_propensity.params import load_prepare_params
from purchase_propensity.schema import (
    FEATURE_COLUMNS,
    RAW_DATA_PATH,
    TARGET_COLUMN,
    TEST_DATA_PATH,
    TRAIN_DATA_PATH,
)


def select_and_cast(raw: pd.DataFrame) -> pd.DataFrame:
    """Mantém as colunas do esquema e converte o alvo para 0/1.

    O alvo chega como booleano (`TRUE`/`FALSE` no CSV). A conversão fica em código
    porque a saída deste estágio é versionada pelo DVC. As duplicadas são mantidas: são
    sessões distintas com o mesmo comportamento trivial (estudo do dataset, §4.5).

    Args:
        raw: DataFrame lido do CSV bruto. Não é modificado.

    Returns:
        Novo DataFrame com as features na ordem do esquema e o alvo como `int`.
    """
    selected = raw.loc[:, [*FEATURE_COLUMNS, TARGET_COLUMN]]
    return selected.assign(**{TARGET_COLUMN: selected[TARGET_COLUMN].astype(int)})


def split_train_test(
    data: pd.DataFrame, test_size: float, seed: int
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Separa treino e teste, estratificando pelo alvo.

    Com 15,5% de positivos, um split sem estratificação poderia deixar o teste com uma
    prevalência diferente da do treino e distorcer o PR-AUC.

    Args:
        data: DataFrame com features e alvo.
        test_size: Fração das linhas que vai para o teste.
        seed: Semente do sorteio.

    Returns:
        Os conjuntos de treino e de teste, nesta ordem.
    """
    train, test = train_test_split(
        data, test_size=test_size, random_state=seed, stratify=data[TARGET_COLUMN]
    )
    return train, test


def main() -> None:
    """Lê o CSV bruto e grava treino e teste em `data/processed/`."""
    params = load_prepare_params()
    data = select_and_cast(pd.read_csv(RAW_DATA_PATH))
    train, test = split_train_test(data, params.test_size, params.seed)
    TRAIN_DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
    train.to_csv(TRAIN_DATA_PATH, index=False)
    test.to_csv(TEST_DATA_PATH, index=False)
