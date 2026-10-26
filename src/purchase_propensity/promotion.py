"""Critério de promoção ao Registry: quem concorre, quem é o candidato e se ele passa.

Módulo puro, sem MLflow: recebe os runs já resumidos e devolve a decisão com os motivos.
Quem lê o MLflow e registra é o `promote.py` (decision log §4).

A regra está no bloco `promotion` do `params.yaml`, commitado antes dos runs da entrega:
concorrem só os runs do commit atual, com o código limpo e o mesmo dado; o candidato é o
de maior média fora o piso; ele é promovido se superar o piso e o champion atual por mais
que `margin_in_stds` desvios da própria CV. Uma diferença dentro do ruído não promove.
"""

from collections.abc import Iterable
from dataclasses import dataclass

from purchase_propensity.params import PromotionParams


@dataclass(frozen=True)
class RunSummary:
    """O que o critério precisa saber de um run do `train`.

    Attributes:
        run_id: Id do run no MLflow.
        model: Nome do candidato (`train.model`).
        mean: Média da métrica na CV.
        std: Desvio-padrão da métrica entre as dobras.
        git_commit: Commit do código que gerou o run.
        git_dirty: `"true"` se o código diferia do commit.
        data_md5: MD5 do `train.csv` lido.
        dvc_exp: Nome do experimento do DVC (vazio num `dvc repro`).
    """

    run_id: str
    model: str
    mean: float
    std: float
    git_commit: str
    git_dirty: str
    data_md5: str
    dvc_exp: str


@dataclass(frozen=True)
class PromotionDecision:
    """Resultado do critério, com os motivos para imprimir e gravar na versão.

    Attributes:
        candidate: Melhor run elegível fora o piso, ou `None` se não houver.
        promote: Se o candidato deve virar o champion.
        reasons: Uma frase por verificação feita, na ordem.
    """

    candidate: RunSummary | None
    promote: bool
    reasons: tuple[str, ...]


def eligible_runs(
    runs: Iterable[RunSummary], commit: str, data_md5: str
) -> list[RunSummary]:
    """Filtra os runs que podem ser comparados entre si.

    Comparar runs de commits ou dados diferentes não é comparação: a diferença de métrica
    poderia vir do código ou do dado, e não do modelo.

    Args:
        runs: Todos os runs do experimento.
        commit: Commit atual do repositório.
        data_md5: MD5 do `train.csv` atual.

    Returns:
        Os runs do mesmo commit e dado, com o código limpo.
    """
    return [
        run
        for run in runs
        if run.git_commit == commit
        and run.git_dirty == "false"
        and run.data_md5 == data_md5
    ]


def _best(runs: Iterable[RunSummary]) -> RunSummary | None:
    return max(runs, key=lambda run: run.mean, default=None)


def _describe(run: RunSummary) -> str:
    return f"{run.model} ({run.mean:.4f} ± {run.std:.4f}, run {run.run_id[:8]})"


def decide_promotion(
    runs: Iterable[RunSummary],
    champion: RunSummary | None,
    params: PromotionParams,
) -> PromotionDecision:
    """Escolhe o candidato entre os runs elegíveis e aplica o gate.

    Args:
        runs: Runs já filtrados por `eligible_runs`.
        champion: Run da versão que o alias aponta hoje, ou `None` na primeira promoção.
        params: Critério lido do `params.yaml`.

    Returns:
        A decisão. Sem candidato ou sem run do piso, não promove.
    """
    runs = list(runs)
    baseline = _best(r for r in runs if r.model == params.baseline_model)
    candidate = _best(r for r in runs if r.model != params.baseline_model)
    if candidate is None:
        return PromotionDecision(None, False, ("nenhum run elegível fora o piso",))
    if baseline is None:
        reason = f"sem run do piso ({params.baseline_model}) no mesmo commit e dado"
        return PromotionDecision(candidate, False, (reason,))
    if champion is not None and champion.run_id == candidate.run_id:
        return PromotionDecision(candidate, False, ("o candidato já é o champion",))

    margin = params.margin_in_stds * candidate.std
    rivals = [("piso", baseline)] + ([("champion", champion)] if champion else [])
    reasons = [f"candidato: {_describe(candidate)}; margem {margin:.4f}"]
    passed = True
    for label, rival in rivals:
        beats = candidate.mean > rival.mean + margin
        passed = passed and beats
        verdict = "supera" if beats else "NÃO supera"
        reasons.append(f"{verdict} o {label} {_describe(rival)} + margem")
    if champion is None:
        reasons.append("sem champion: primeira promoção, só o piso conta")
    return PromotionDecision(candidate, passed, tuple(reasons))
