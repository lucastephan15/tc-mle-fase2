"""Candidatos a modelo: o único lugar do projeto onde um estimador é instanciado.

O `params.yaml` escolhe o candidato pelo nome (`train.model`). É um dicionário de
funções, e não uma hierarquia de classes: cada candidato é uma linha, e acrescentar um
não exige mexer em nenhum outro módulo (decision log §3).

Os hiperparâmetros são os padrões do scikit-learn. Modelagem vale 10% da nota, e o
ajuste fino, se vier, entra pelo `params.yaml` na Etapa 4.
"""

from collections.abc import Callable

from sklearn.base import ClassifierMixin
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression

ModelFactory = Callable[[int], ClassifierMixin]

MODEL_FACTORIES: dict[str, ModelFactory] = {
    # Piso: prevê a prevalência para todo mundo. Qualquer candidato tem de superá-lo.
    "dummy": lambda seed: DummyClassifier(strategy="prior", random_state=seed),
    "logistic_regression": lambda seed: LogisticRegression(random_state=seed),
    "random_forest": lambda seed: RandomForestClassifier(random_state=seed, n_jobs=-1),
    "hist_gradient_boosting": lambda seed: HistGradientBoostingClassifier(
        random_state=seed
    ),
}


def build_model(name: str, seed: int) -> ClassifierMixin:
    """Instancia o candidato pelo nome declarado no `params.yaml`.

    Args:
        name: Chave de `MODEL_FACTORIES`.
        seed: Semente repassada ao estimador.

    Returns:
        O estimador ainda não ajustado.
    """
    return MODEL_FACTORIES[name](seed)
