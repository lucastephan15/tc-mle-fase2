import pytest

from purchase_propensity.params import PromotionParams
from purchase_propensity.promotion import RunSummary, decide_promotion, eligible_runs

COMMIT = "a" * 40
DATA_MD5 = "d" * 32
PARAMS = PromotionParams(
    model_name="purchase-propensity",
    alias="champion",
    metric="cv_pr_auc_mean",
    baseline_model="dummy",
    margin_in_stds=1.0,
)


def summary(
    run_id: str, model: str, mean: float, std: float = 0.01, **tags
) -> RunSummary:
    fields = {"git_commit": COMMIT, "git_dirty": "false", "data_md5": DATA_MD5}
    return RunSummary(run_id, model, mean, std, **(fields | tags), dvc_exp="")


DUMMY = summary("dummy", "dummy", 0.16)


@pytest.mark.parametrize(
    "tags",
    [{"git_commit": "b" * 40}, {"git_dirty": "true"}, {"data_md5": "e" * 32}],
    ids=["outro commit", "codigo sujo", "outro dado"],
)
def test_run_de_outro_codigo_ou_dado_nao_concorre(tags):
    runs = [
        summary("ok", "logistic_regression", 0.6),
        summary("x", "random_forest", 0.9, **tags),
    ]
    assert [r.run_id for r in eligible_runs(runs, COMMIT, DATA_MD5)] == ["ok"]


def test_run_sem_estado_git_nao_concorre():
    # Um run treinado fora do repositório tem a tag `unavailable`, que nunca é um commit.
    run = summary("x", "logistic_regression", 0.6, git_commit="unavailable")
    assert eligible_runs([run], COMMIT, DATA_MD5) == []


def test_primeira_promocao_compara_so_com_o_piso():
    decision = decide_promotion(
        [DUMMY, summary("lr", "logistic_regression", 0.6)], None, PARAMS
    )
    assert decision.promote
    assert decision.candidate.run_id == "lr"


def test_candidato_e_o_melhor_fora_o_piso():
    runs = [
        DUMMY,
        summary("lr", "logistic_regression", 0.6),
        summary("hgb", "hist_gradient_boosting", 0.65),
    ]
    assert decide_promotion(runs, None, PARAMS).candidate.run_id == "hgb"


def test_sem_run_do_piso_nao_promove():
    decision = decide_promotion([summary("lr", "logistic_regression", 0.6)], None, PARAMS)
    assert not decision.promote
    assert "piso" in decision.reasons[0]


def test_so_o_piso_nao_tem_candidato():
    decision = decide_promotion([DUMMY], None, PARAMS)
    assert decision.candidate is None
    assert not decision.promote


def test_diferenca_dentro_do_ruido_nao_promove():
    champion = summary("old", "logistic_regression", 0.640)
    candidate = summary("new", "hist_gradient_boosting", 0.650, std=0.017)
    decision = decide_promotion([DUMMY, candidate], champion, PARAMS)
    assert not decision.promote
    assert any("NÃO supera o champion" in r for r in decision.reasons)


def test_supera_o_champion_por_mais_que_a_margem_promove():
    champion = summary("old", "logistic_regression", 0.62)
    candidate = summary("new", "hist_gradient_boosting", 0.65, std=0.017)
    assert decide_promotion([DUMMY, candidate], champion, PARAMS).promote


def test_margem_e_em_desvios_do_candidato():
    champion = summary("old", "logistic_regression", 0.62)
    candidate = summary("new", "hist_gradient_boosting", 0.65, std=0.017)
    strict = PromotionParams(**{**PARAMS.__dict__, "margin_in_stds": 2.0})
    assert not decide_promotion([DUMMY, candidate], champion, strict).promote


def test_candidato_que_ja_e_champion_nao_promove_de_novo():
    candidate = summary("lr", "logistic_regression", 0.6)
    decision = decide_promotion([DUMMY, candidate], candidate, PARAMS)
    assert not decision.promote
    assert "já é o champion" in decision.reasons[0]
