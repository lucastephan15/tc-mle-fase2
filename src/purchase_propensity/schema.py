"""Contrato das colunas do dataset Online Shoppers Purchasing Intention (UCI 468).

Única fonte de verdade sobre o papel de cada coluna. Pré-processamento, treino e testes
importam daqui; nenhuma lista de colunas é redigitada em outro módulo.
"""

from pathlib import Path

RAW_DATA_PATH = Path("data/raw/online_shoppers_intention.csv")
TRAIN_DATA_PATH = Path("data/processed/train.csv")
TEST_DATA_PATH = Path("data/processed/test.csv")

TARGET_COLUMN = "Revenue"

NUMERIC_FEATURES: tuple[str, ...] = (
    "Administrative",
    "Administrative_Duration",
    "Informational",
    "Informational_Duration",
    "ProductRelated",
    "ProductRelated_Duration",
    "BounceRates",
    "ExitRates",
    # Quase-vazamento: o Google Analytics atribui valor à página olhando transações que
    # vêm depois dela. Fica no modelo por decisão registrada, com experimento com/sem.
    "PageValues",
    "SpecialDay",
)

# Os quatro primeiros chegam como int, mas são CÓDIGOS (região 9 não é 9x a região 1).
# Tratados como categoria, nunca como grandeza.
CATEGORICAL_FEATURES: tuple[str, ...] = (
    "OperatingSystems",
    "Browser",
    "Region",
    "TrafficType",
    "Month",
    "VisitorType",
)

BOOLEAN_FEATURES: tuple[str, ...] = ("Weekend",)

FEATURE_COLUMNS: tuple[str, ...] = (
    NUMERIC_FEATURES + CATEGORICAL_FEATURES + BOOLEAN_FEATURES
)
