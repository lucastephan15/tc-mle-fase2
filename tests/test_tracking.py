import hashlib
from pathlib import Path

import joblib
import mlflow
import numpy as np
import pytest
from dulwich import porcelain
from mlflow.tracking import MlflowClient

from purchase_propensity import train as train_stage
from purchase_propensity.params import PARAMS_PATH, PrepareParams, TrainParams
from purchase_propensity.schema import MODEL_PATH, TARGET_COLUMN, TRAIN_DATA_PATH
from purchase_propensity.tracking import (
    EXPERIMENT_NAME,
    UNAVAILABLE,
    file_md5,
    flatten_params,
    git_state,
    run_tags,
)

REPO_PARAMS = Path(__file__).parents[1] / PARAMS_PATH
IDENTITY = b"Teste <teste@example.com>"


@pytest.fixture
def isolated_env(monkeypatch):
    # setenv antes do delenv faz o monkeypatch lembrar do estado original e restaurá-lo,
    # inclusive se o load_dotenv do main() definir a variável durante o teste.
    for name in ("MLFLOW_TRACKING_URI", "DVC_EXP_NAME"):
        monkeypatch.setenv(name, "")
        monkeypatch.delenv(name)


def commit_all(repo_path: Path) -> None:
    porcelain.add(str(repo_path), paths=[str(p) for p in repo_path.rglob("*.*")])
    porcelain.commit(str(repo_path), message=b"base", author=IDENTITY, committer=IDENTITY)


@pytest.fixture
def repo(tmp_path) -> Path:
    porcelain.init(str(tmp_path))
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "stage.py").write_text("x = 1\n", encoding="utf-8")
    (tmp_path / "params.yaml").write_text("seed: 1\n", encoding="utf-8")
    commit_all(tmp_path)
    return tmp_path


def test_md5_e_o_mesmo_que_o_dvc_grava(tmp_path):
    path = tmp_path / "dado.csv"
    path.write_bytes(b"a,b\n1,2\n")
    assert file_md5(path) == hashlib.md5(b"a,b\n1,2\n").hexdigest()


def test_fora_de_repositorio_o_estado_git_sai_indisponivel(tmp_path):
    assert git_state(tmp_path) == (UNAVAILABLE, UNAVAILABLE)


def test_repositorio_limpo_tem_commit_e_nao_esta_sujo(repo):
    commit, dirty = git_state(repo)
    assert len(commit) == 40
    assert dirty == "false"


def test_mudanca_so_no_params_nao_suja_o_codigo(repo):
    # É o que o `dvc exp run -S` faz antes de todo run.
    (repo / "params.yaml").write_text("seed: 2\n", encoding="utf-8")
    assert git_state(repo)[1] == "false"


@pytest.mark.parametrize("change", ["modified", "untracked"])
def test_mudanca_no_codigo_suja(repo, change):
    target = repo / "src" / ("stage.py" if change == "modified" else "novo.py")
    target.write_text("x = 2\n", encoding="utf-8")
    assert git_state(repo)[1] == "true"


def test_tag_do_experimento_vem_do_dvc(tmp_path, monkeypatch, isolated_env):
    data = tmp_path / "train.csv"
    data.write_bytes(b"x\n")
    assert run_tags(data)["dvc.exp"] == ""
    monkeypatch.setenv("DVC_EXP_NAME", "north-mats")
    assert run_tags(data)["dvc.exp"] == "north-mats"


def test_params_achatados_levam_o_nome_do_bloco():
    blocks = {
        "prepare": PrepareParams(test_size=0.2, seed=42),
        "train": TrainParams("dummy", True, 10, 5, 7),
    }
    flat = flatten_params(blocks)
    assert flat["prepare.seed"] == 42
    assert flat["train.seed"] == 7
    assert flat["train.model"] == "dummy"


def test_main_registra_o_run_no_destino_do_env(
    train, tmp_path, monkeypatch, isolated_env
):
    """O `.env` aponta para um banco diferente do padrão: o run tem de cair nele."""
    monkeypatch.chdir(tmp_path)
    PARAMS_PATH.parent.mkdir(parents=True)
    PARAMS_PATH.write_text(REPO_PARAMS.read_text(encoding="utf-8"), encoding="utf-8")
    TRAIN_DATA_PATH.parent.mkdir(parents=True)
    train.to_csv(TRAIN_DATA_PATH, index=False)
    Path(".env").write_text("MLFLOW_TRACKING_URI=sqlite:///outro.db\n", encoding="utf-8")

    train_stage.main()

    assert not Path("mlflow.db").exists()
    client = MlflowClient(tracking_uri="sqlite:///outro.db")
    experiment = client.get_experiment_by_name(EXPERIMENT_NAME)
    (run,) = client.search_runs([experiment.experiment_id])
    assert run.data.tags["data.md5"] == file_md5(TRAIN_DATA_PATH)
    assert run.data.tags["git.commit"] == UNAVAILABLE
    assert {"prepare.seed", "train.model"} <= run.data.params.keys()
    assert {"cv_pr_auc_mean", "cv_pr_auc_std"} <= run.data.metrics.keys()

    # O modelo do run é o mesmo que o DVC versiona em models/.
    mlflow.set_tracking_uri("sqlite:///outro.db")
    logged = mlflow.sklearn.load_model(f"runs:/{run.info.run_id}/model")
    X = train.drop(columns=TARGET_COLUMN)
    np.testing.assert_allclose(
        logged.predict_proba(X), joblib.load(MODEL_PATH).predict_proba(X)
    )
