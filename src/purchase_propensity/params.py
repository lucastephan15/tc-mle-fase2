"""Carga e validação do `configs/params.yaml`.

O arquivo é lido uma vez, no `main()` de cada estágio, e vira um objeto imutável
passado por parâmetro. Chave digitada errada ou valor fora da faixa falham na carga,
antes de qualquer dado ser lido (decision log §1).
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from purchase_propensity.models import MODEL_FACTORIES

PARAMS_PATH = Path("configs/params.yaml")
MIN_CV_FOLDS = 2
METRIC_SUFFIX = "_mean"


@dataclass(frozen=True)
class PrepareParams:
    """Parâmetros do estágio `prepare`.

    Attributes:
        test_size: Fração das linhas reservada ao conjunto de teste.
        seed: Semente do split estratificado.
    """

    test_size: float
    seed: int

    def __post_init__(self) -> None:
        """Rejeita valores que o `train_test_split` aceitaria com outro sentido."""
        if not 0 < self.test_size < 1:
            msg = f"test_size deve estar entre 0 e 1 (exclusivos): {self.test_size}"
            raise ValueError(msg)
        _check_seed(self.seed)


@dataclass(frozen=True)
class TrainParams:
    """Parâmetros do estágio `train`.

    Attributes:
        model: Nome do candidato em `MODEL_FACTORIES`.
        use_page_values: Se `PageValues` entra nas features.
        min_frequency: Contagem mínima para uma categoria ganhar coluna própria.
        cv_folds: Número de dobras da validação cruzada estratificada.
        seed: Semente das dobras e do estimador.
    """

    model: str
    use_page_values: bool
    min_frequency: int
    cv_folds: int
    seed: int

    def __post_init__(self) -> None:
        """Rejeita modelo desconhecido e valores que não fazem sentido."""
        if self.model not in MODEL_FACTORIES:
            msg = f"model deve ser um de {sorted(MODEL_FACTORIES)}: {self.model!r}"
            raise ValueError(msg)
        if self.min_frequency < 1:
            msg = f"min_frequency deve ser ao menos 1: {self.min_frequency}"
            raise ValueError(msg)
        if self.cv_folds < MIN_CV_FOLDS:
            msg = f"cv_folds deve ser ao menos {MIN_CV_FOLDS}: {self.cv_folds}"
            raise ValueError(msg)
        _check_seed(self.seed)


@dataclass(frozen=True)
class PromotionParams:
    """Critério de promoção ao Registry, lido pelo comando `promote`.

    Attributes:
        model_name: Nome do modelo no Registry.
        alias: Alias que aponta para a versão em uso.
        metric: Média da CV usada na comparação (`cv_<métrica>_mean`).
        baseline_model: Candidato que serve de piso, rodado no mesmo commit e dado.
        margin_in_stds: Quantos desvios da CV do candidato ele precisa superar.
    """

    model_name: str
    alias: str
    metric: str
    baseline_model: str
    margin_in_stds: float

    def __post_init__(self) -> None:
        """Rejeita nome vazio, métrica sem desvio correspondente e margem negativa."""
        for field_name in ("model_name", "alias"):
            if not getattr(self, field_name).strip():
                msg = f"{field_name} não pode ser vazio"
                raise ValueError(msg)
        if not (self.metric.startswith("cv_") and self.metric.endswith(METRIC_SUFFIX)):
            msg = (
                f"metric deve ser uma média da CV (cv_*{METRIC_SUFFIX}): {self.metric!r}"
            )
            raise ValueError(msg)
        if self.baseline_model not in MODEL_FACTORIES:
            msg = (
                f"baseline_model deve ser um de {sorted(MODEL_FACTORIES)}: "
                f"{self.baseline_model!r}"
            )
            raise ValueError(msg)
        if self.margin_in_stds < 0:
            msg = f"margin_in_stds não pode ser negativa: {self.margin_in_stds}"
            raise ValueError(msg)

    @property
    def std_metric(self) -> str:
        """Nome do desvio-padrão que acompanha a métrica no run."""
        return self.metric.removesuffix(METRIC_SUFFIX) + "_std"


def _check_seed(seed: int) -> None:
    if seed < 0:
        msg = f"seed deve ser não negativa, recebido {seed}"
        raise ValueError(msg)


def _read_block(path: Path, block: str) -> dict[str, Any]:
    return yaml.safe_load(path.read_text(encoding="utf-8"))[block]


def load_prepare_params(path: Path = PARAMS_PATH) -> PrepareParams:
    """Lê o bloco `prepare` do arquivo de parâmetros.

    Args:
        path: Caminho do YAML de parâmetros.

    Returns:
        Os parâmetros validados. Uma chave a mais ou a menos levanta `TypeError`.
    """
    return PrepareParams(**_read_block(path, "prepare"))


def load_train_params(path: Path = PARAMS_PATH) -> TrainParams:
    """Lê o bloco `train` do arquivo de parâmetros.

    Args:
        path: Caminho do YAML de parâmetros.

    Returns:
        Os parâmetros validados. Uma chave a mais ou a menos levanta `TypeError`.
    """
    return TrainParams(**_read_block(path, "train"))


def load_promotion_params(path: Path = PARAMS_PATH) -> PromotionParams:
    """Lê o bloco `promotion` do arquivo de parâmetros.

    Args:
        path: Caminho do YAML de parâmetros.

    Returns:
        Os parâmetros validados. Uma chave a mais ou a menos levanta `TypeError`.
    """
    return PromotionParams(**_read_block(path, "promotion"))
