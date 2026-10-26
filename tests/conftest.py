"""Fixtures compartilhadas pelos testes de `train` e `tracking`."""

import numpy as np
import pandas as pd
import pytest

from purchase_propensity.schema import (
    BOOLEAN_FEATURES,
    CATEGORICAL_FEATURES,
    NUMERIC_FEATURES,
    TARGET_COLUMN,
)

N_ROWS = 200


@pytest.fixture
def train() -> pd.DataFrame:
    rng = np.random.default_rng(0)
    data = {col: rng.random(N_ROWS) for col in NUMERIC_FEATURES}
    data |= {col: rng.integers(1, 5, N_ROWS) for col in CATEGORICAL_FEATURES[:4]}
    data["Month"] = rng.choice(["Feb", "Mar", "May", "Nov"], N_ROWS)
    data["VisitorType"] = rng.choice(["Returning_Visitor", "New_Visitor"], N_ROWS)
    data |= {col: rng.random(N_ROWS) < 0.5 for col in BOOLEAN_FEATURES}
    data[TARGET_COLUMN] = (rng.random(N_ROWS) < 0.3).astype(int)
    return pd.DataFrame(data)
