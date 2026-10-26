"""O `promote` contra um MLflow de verdade, num banco SQLite temporário."""

import mlflow
import numpy as np
import pandas as pd
import pytest
from mlflow.tracking import MlflowClient
from sklearn.dummy import DummyClassifier

from purchase_propensity.promote import NO_CHAMPION, promote
from purchase_propensity.tracking import EXPERIMENT_NAME
from tests.test_promotion import COMMIT, DATA_MD5, PARAMS

NAME, ALIAS = PARAMS.model_name, PARAMS.alias


@pytest.fixture
def client(tmp_path, monkeypatch) -> MlflowClient:
    # Só a variável de ambiente, como no `main()`. O URI global do `mlflow` é zerado
    # porque outro teste pode tê-lo fixado; o monkeypatch o devolve ao estado anterior.
    monkeypatch.setattr("mlflow.tracking._tracking_service.utils._tracking_uri", None)
    monkeypatch.setenv("MLFLOW_TRACKING_URI", f"sqlite:///{tmp_path / 'mlflow.db'}")
    return MlflowClient()


def log_run(model: str, mean: float, std: float = 0.01, commit: str = COMMIT) -> str:
    X, y = pd.DataFrame({"x": np.arange(10.0)}), np.arange(10) % 2
    mlflow.set_experiment(EXPERIMENT_NAME)
    with mlflow.start_run() as run:
        mlflow.log_param("train.model", model)
        mlflow.log_metrics({"cv_pr_auc_mean": mean, "cv_pr_auc_std": std})
        tags = {"git.commit": commit, "git.dirty": "false", "data.md5": DATA_MD5}
        mlflow.set_tags(tags | {"dvc.exp": f"exp-{model}"})
        mlflow.sklearn.log_model(DummyClassifier().fit(X, y), name="model")
    return run.info.run_id


def run_promote(client: MlflowClient) -> str | None:
    return promote(client, PARAMS, COMMIT, DATA_MD5)[1]


def test_primeira_promocao_registra_e_aponta_o_alias(client):
    log_run("dummy", 0.16)
    lr = log_run("logistic_regression", 0.64)

    assert run_promote(client) == "1"

    champion = client.get_model_version_by_alias(NAME, ALIAS)
    assert champion.run_id == lr
    assert champion.tags["previous_champion"] == NO_CHAMPION
    assert {"approved_by", "promoted_at", "criterion"} <= champion.tags.keys()
    assert "piso" in champion.tags["criterion"]
    assert mlflow.sklearn.load_model(f"models:/{NAME}@{ALIAS}") is not None


def test_rodar_de_novo_nao_cria_versao(client):
    log_run("dummy", 0.16)
    log_run("logistic_regression", 0.64)
    run_promote(client)

    assert run_promote(client) is None
    assert len(client.search_model_versions(f"name = '{NAME}'")) == 1


def test_novo_champion_guarda_o_anterior(client):
    log_run("dummy", 0.16)
    log_run("logistic_regression", 0.60)
    run_promote(client)
    hgb = log_run("hist_gradient_boosting", 0.65, std=0.02)

    assert run_promote(client) == "2"
    champion = client.get_model_version_by_alias(NAME, ALIAS)
    assert champion.run_id == hgb
    assert champion.tags["previous_champion"] == "1"


def test_melhora_dentro_do_ruido_mantem_o_champion(client):
    log_run("dummy", 0.16)
    lr = log_run("logistic_regression", 0.640)
    run_promote(client)
    log_run("hist_gradient_boosting", 0.650, std=0.017)

    assert run_promote(client) is None
    assert client.get_model_version_by_alias(NAME, ALIAS).run_id == lr


def test_modelo_registrado_sem_alias_conta_como_primeira_promocao(client):
    # Neste estado o `get_model_version_by_alias` levanta INVALID_PARAMETER_VALUE, e não
    # RESOURCE_DOES_NOT_EXIST: tratar pelo código de erro faria o `promote` quebrar aqui.
    client.create_registered_model(NAME)
    log_run("dummy", 0.16)
    log_run("logistic_regression", 0.64)
    assert run_promote(client) == "1"


def test_run_ja_registrado_e_reaproveitado(client):
    log_run("dummy", 0.16)
    lr = log_run("logistic_regression", 0.64)
    (output,) = client.get_run(lr).outputs.model_outputs
    mlflow.register_model(f"models:/{output.model_id}", NAME)

    assert run_promote(client) == "1"
    assert len(client.search_model_versions(f"name = '{NAME}'")) == 1


def test_runs_de_outro_commit_nao_sao_promovidos(client):
    log_run("dummy", 0.16, commit="b" * 40)
    log_run("logistic_regression", 0.64, commit="b" * 40)
    decision, version = promote(client, PARAMS, COMMIT, DATA_MD5)
    assert version is None
    assert decision.candidate is None
