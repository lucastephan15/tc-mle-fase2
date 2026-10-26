"""Comando `promote`: escolhe o melhor run, aplica o critério e registra o champion.

Roda fora do `dvc.yaml`, depois dos `dvc exp run` da entrega. "Registrar o melhor" é uma
decisão sobre um conjunto de runs, e não o produto de um run só (decision log §4). A
decisão é do `promotion.py`; aqui fica só a conversa com o MLflow.

O alias não guarda histórico no MLflow: reapontá-lo apaga para onde ele apontava. A
trilha de auditoria são as tags gravadas na versão promovida (`previous_champion`,
`promoted_at`, `approved_by`, `criterion`).
"""

import getpass
from datetime import UTC, datetime

import mlflow
from dotenv import load_dotenv
from mlflow.entities import Run, RunStatus
from mlflow.entities.model_registry import ModelVersion
from mlflow.exceptions import MlflowException
from mlflow.protos.databricks_pb2 import RESOURCE_DOES_NOT_EXIST, ErrorCode
from mlflow.tracking import MlflowClient

from purchase_propensity.params import PromotionParams, load_promotion_params
from purchase_propensity.promotion import (
    PromotionDecision,
    RunSummary,
    decide_promotion,
    eligible_runs,
)
from purchase_propensity.schema import TRAIN_DATA_PATH
from purchase_propensity.tracking import (
    ENV_PATH,
    EXPERIMENT_NAME,
    UNAVAILABLE,
    file_md5,
    git_state,
)

NO_CHAMPION = "none"


def summarize(run: Run, params: PromotionParams) -> RunSummary | None:
    """Resume um run do MLflow no que o critério lê.

    Args:
        run: Run do experimento.
        params: Critério, de onde vêm os nomes da métrica e do desvio.

    Returns:
        O resumo, ou `None` se o run não terminou ou não tem a métrica (um `train` que
        falhou no meio, por exemplo).
    """
    metrics, tags = run.data.metrics, run.data.tags
    finished = run.info.status == RunStatus.to_string(RunStatus.FINISHED)
    if not finished or not {params.metric, params.std_metric} <= metrics.keys():
        return None
    return RunSummary(
        run_id=run.info.run_id,
        model=run.data.params.get("train.model", ""),
        mean=metrics[params.metric],
        std=metrics[params.std_metric],
        git_commit=tags.get("git.commit", UNAVAILABLE),
        git_dirty=tags.get("git.dirty", UNAVAILABLE),
        data_md5=tags.get("data.md5", ""),
        dvc_exp=tags.get("dvc.exp", ""),
    )


def current_champion(
    client: MlflowClient, params: PromotionParams
) -> ModelVersion | None:
    """Lê a versão para onde o alias aponta.

    Pelo `aliases` do modelo registrado, e não pelo `get_model_version_by_alias`: com o
    modelo registrado e sem o alias, este levanta `INVALID_PARAMETER_VALUE`, o código
    de um parâmetro errado (testado no MLflow 3.17.2).

    Args:
        client: Cliente apontado para o destino do `MLFLOW_TRACKING_URI`.
        params: Critério, com o nome do modelo e o alias.

    Returns:
        A versão champion, ou `None` se o modelo ou o alias ainda não existem.
    """
    try:
        registered = client.get_registered_model(params.model_name)
    except MlflowException as error:
        # Só "não existe" quer dizer primeira promoção. Qualquer outro erro (banco
        # inacessível, por exemplo) sobe, em vez de virar uma promoção sem comparação.
        if error.error_code != ErrorCode.Name(RESOURCE_DOES_NOT_EXIST):
            raise
        return None
    version = registered.aliases.get(params.alias)
    return (
        None if version is None else client.get_model_version(params.model_name, version)
    )


def _version_for(client: MlflowClient, run_id: str, params: PromotionParams) -> str:
    # Rodar de novo não duplica: se o run já tem versão (um rollback anterior, ou uma
    # execução interrompida antes do alias), a versão existente é reaproveitada.
    existing = client.search_model_versions(
        f"name = '{params.model_name}' and run_id = '{run_id}'"
    )
    if existing:
        return str(existing[0].version)
    # O `models:/m-<id>` é resolvido pela URI global do `mlflow`, não pela do client:
    # os dois precisam apontar para o mesmo banco (testado no 3.17.2).
    (output,) = client.get_run(run_id).outputs.model_outputs
    return str(
        mlflow.register_model(f"models:/{output.model_id}", params.model_name).version
    )


def register_champion(
    client: MlflowClient,
    decision: PromotionDecision,
    previous: ModelVersion | None,
    params: PromotionParams,
) -> str:
    """Registra o candidato (se ainda não estiver), grava a auditoria e move o alias.

    Args:
        client: Cliente apontado para o destino do `MLFLOW_TRACKING_URI`.
        decision: Decisão com `promote=True`.
        previous: Champion atual, ou `None`.
        params: Critério, com o nome do modelo e o alias.

    Returns:
        O número da versão que passou a ser o champion.
    """
    version = _version_for(client, decision.candidate.run_id, params)
    audit = {
        "approved_by": getpass.getuser(),
        "promoted_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "previous_champion": previous.version if previous else NO_CHAMPION,
        "criterion": " | ".join(decision.reasons),
    }
    for key, value in audit.items():
        client.set_model_version_tag(params.model_name, version, key, str(value))
    client.set_registered_model_alias(params.model_name, params.alias, version)
    return str(version)


def promote(
    client: MlflowClient, params: PromotionParams, commit: str, data_md5: str
) -> tuple[PromotionDecision, str | None]:
    """Compara os runs elegíveis com o champion e registra se o critério passar.

    Args:
        client: Cliente apontado para o destino do `MLFLOW_TRACKING_URI`.
        params: Critério lido do `params.yaml`.
        commit: Commit atual do repositório.
        data_md5: MD5 do `train.csv` atual.

    Returns:
        A decisão e a versão promovida (`None` se não promoveu).
    """
    experiment = client.get_experiment_by_name(EXPERIMENT_NAME)
    runs = client.search_runs([experiment.experiment_id]) if experiment else []
    summaries = [s for s in (summarize(run, params) for run in runs) if s]

    previous = current_champion(client, params)
    champion = summarize(client.get_run(previous.run_id), params) if previous else None
    decision = decide_promotion(
        eligible_runs(summaries, commit, data_md5), champion, params
    )
    if not decision.promote:
        return decision, None
    return decision, register_champion(client, decision, previous, params)


def main() -> None:
    """Executa a promoção e imprime os motivos e o passo seguinte.

    Sai com código 1 quando nada é promovido, para que um script que encadeie o
    `promote` não siga adiante achando que há champion novo.
    """
    load_dotenv(ENV_PATH)
    params = load_promotion_params()
    commit, _ = git_state()
    if commit == UNAVAILABLE:
        raise SystemExit("promote precisa do repositório git: o critério compara commits")

    decision, version = promote(MlflowClient(), params, commit, file_md5(TRAIN_DATA_PATH))
    for reason in decision.reasons:
        print(f"- {reason}")
    if version is None:
        raise SystemExit(f"nada promovido; {params.alias} continua onde estava")

    print(f"{params.model_name} v{version} agora é @{params.alias}")
    if decision.candidate.dvc_exp:
        # O workspace tem os params e o dvc.lock do último experimento, não do campeão.
        print(f"antes de commitar: dvc exp apply {decision.candidate.dvc_exp}")
