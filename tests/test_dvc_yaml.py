"""A regra do topo do `dvc.yaml`: todo módulo que um estágio importa está nas `deps`."""

import ast
import tomllib
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).parents[1]
PACKAGE = "purchase_propensity"
PACKAGE_DIR = Path("src") / PACKAGE
STAGES = yaml.safe_load((ROOT / "dvc.yaml").read_text(encoding="utf-8"))["stages"]
SCRIPTS = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))["project"][
    "scripts"
]


def local_imports(module: str) -> set[str]:
    tree = ast.parse((ROOT / PACKAGE_DIR / f"{module}.py").read_text(encoding="utf-8"))
    return {
        node.module.removeprefix(f"{PACKAGE}.")
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom)
        and node.module
        and node.module.startswith(f"{PACKAGE}.")
    }


def transitive_imports(module: str) -> set[str]:
    seen, pending = {module}, [module]
    while pending:
        for imported in local_imports(pending.pop()) - seen:
            seen.add(imported)
            pending.append(imported)
    return seen


@pytest.mark.parametrize("stage", sorted(STAGES))
def test_todo_modulo_importado_esta_nas_deps(stage):
    entry_point = SCRIPTS[STAGES[stage]["cmd"]]
    module = entry_point.split(":")[0].removeprefix(f"{PACKAGE}.")
    expected = {str(PACKAGE_DIR / f"{name}.py") for name in transitive_imports(module)}
    assert expected <= set(STAGES[stage]["deps"])


def test_evaluate_nao_depende_do_registry():
    # O alias não é `dep`: um estágio que o lesse ficaria "up to date" com modelo novo.
    assert "tracking" not in transitive_imports("evaluate")
    assert "mlflow" not in (ROOT / PACKAGE_DIR / "evaluate.py").read_text(
        encoding="utf-8"
    )
