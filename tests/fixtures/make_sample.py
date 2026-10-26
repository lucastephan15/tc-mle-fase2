"""Gera `online_shoppers_sample.csv`, a amostra do dataset real usada no smoke test.

Rodar uma vez, da raiz do repo, depois do `dvc repro` (lê o CSV extraído do zip):

    poetry run python tests/fixtures/make_sample.py

A amostra fica no Git (~20 KB) para o CI rodar o pipeline sem baixar o dado da UCI.
"""

from pathlib import Path

import pandas as pd

from purchase_propensity.schema import RAW_DATA_PATH, TARGET_COLUMN

SAMPLE_PATH = Path(__file__).parent / "online_shoppers_sample.csv"
N_ROWS = 200
SEED = 42
# Categorias raras (VisitorType "Other" é 0,7% do dataset) sumiriam de uma amostra
# aleatória de 200 linhas. Uma linha de cada entra primeiro, e o resto é sorteado.
COVERED_COLUMNS = ("Month", "VisitorType")


def main() -> None:
    """Sorteia uma linha por categoria e completa com amostra estratificada pelo alvo."""
    data = pd.read_csv(RAW_DATA_PATH)
    seeds = pd.concat(
        data.groupby(col).sample(1, random_state=SEED) for col in COVERED_COLUMNS
    )
    rest = data.drop(seeds.index)
    rate = (N_ROWS - len(seeds)) / len(rest)
    filler = rest.groupby(TARGET_COLUMN).sample(frac=rate, random_state=SEED)
    sample = pd.concat([seeds, filler]).sort_index()
    sample.to_csv(SAMPLE_PATH, index=False)


if __name__ == "__main__":
    main()
