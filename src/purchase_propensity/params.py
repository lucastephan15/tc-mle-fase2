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
