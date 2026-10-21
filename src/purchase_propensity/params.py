"""Carga e validação do `configs/params.yaml`.

O arquivo é lido uma vez, no `main()` de cada estágio, e vira um objeto imutável
passado por parâmetro. Chave digitada errada ou valor fora da faixa falham na carga,
antes de qualquer dado ser lido (decision log §1).
"""

from dataclasses import dataclass
from pathlib import Path

import yaml

PARAMS_PATH = Path("configs/params.yaml")


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
        if self.seed < 0:
            msg = f"seed deve ser não negativa, recebido {self.seed}"
            raise ValueError(msg)


def load_prepare_params(path: Path = PARAMS_PATH) -> PrepareParams:
    """Lê o bloco `prepare` do arquivo de parâmetros.

    Args:
        path: Caminho do YAML de parâmetros.

    Returns:
        Os parâmetros validados. Uma chave a mais ou a menos levanta `TypeError`.
    """
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    return PrepareParams(**raw["prepare"])
