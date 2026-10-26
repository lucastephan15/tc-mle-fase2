"""Smoke test: `prepare` → `train` → `evaluate` sobre a amostra do dataset real.

Os outros testes usam dado sintético e chamam o núcleo de cada estágio. Este chama os
`main()` em sequência, lendo e gravando nos caminhos do projeto, como o `dvc repro` faz,
e prova no CI que as peças se encaixam sem baixar o dado da UCI.
"""

import json
import math
import shutil
from pathlib import Path

import pandas as pd
import pytest
from mlflow.tracking import MlflowClient

from purchase_propensity import evaluate, prepare
from purchase_propensity import train as train_stage
from purchase_propensity.params import PARAMS_PATH
from purchase_propensity.schema import (
    CV_METRICS_PATH,
    MODEL_PATH,
    RAW_DATA_PATH,
    TARGET_COLUMN,
    TEST_METRICS_PATH,
)
from purchase_propensity.tracking import EXPERIMENT_NAME

REPO = Path(__file__).parents[1]
SAMPLE = Path(__file__).parent / "fixtures" / "online_shoppers_sample.csv"
MONTHS = {"Feb", "Mar", "May", "June", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"}
VISITOR_TYPES = {"Returning_Visitor", "New_Visitor", "Other"}


@pytest.fixture
def project(tmp_path, monkeypatch):
    """Pasta de projeto com o params.yaml do repo e a amostra no lugar do dado bruto."""
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr("mlflow.tracking._tracking_service.utils._tracking_uri", None)
    monkeypatch.setenv("MLFLOW_TRACKING_URI", f"sqlite:///{tmp_path / 'mlflow.db'}")
    monkeypatch.delenv("DVC_EXP_NAME", raising=False)
    for source, target in ((REPO / PARAMS_PATH, PARAMS_PATH), (SAMPLE, RAW_DATA_PATH)):
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, target)
    return tmp_path


def test_amostra_tem_as_duas_classes_e_todas_as_categorias():
    sample = pd.read_csv(SAMPLE)
    assert set(sample[TARGET_COLUMN]) == {True, False}
    assert set(sample["Month"]) == MONTHS
    assert set(sample["VisitorType"]) == VISITOR_TYPES


def test_pipeline_roda_de_ponta_a_ponta_na_amostra(project):
    prepare.main()
    train_stage.main()
    evaluate.main()

    assert MODEL_PATH.exists()
    for path in (CV_METRICS_PATH, TEST_METRICS_PATH):
        metrics = json.loads(path.read_text(encoding="utf-8"))
        assert metrics
        assert all(math.isfinite(value) for value in metrics.values())
    client = MlflowClient()
    experiment = client.get_experiment_by_name(EXPERIMENT_NAME)
    assert len(client.search_runs([experiment.experiment_id])) == 1
