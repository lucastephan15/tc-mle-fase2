# Comandos do projeto: a mesma sequência que o CI roda.
# `poetry run` em todos os alvos: o resultado não pode depender de qual venv está
# ativado no shell (lição da Fase 1, em que um ruff global deu falso positivo).

.PHONY: setup lint format test ci

setup:  ## Instala o ambiente exatamente como travado no poetry.lock
	poetry install

lint:  ## Lint + checagem de formatação
	poetry run ruff check .
	poetry run ruff format --check .

format:  ## Aplica a formatação
	poetry run ruff format .

test:  ## Suíte de testes
	poetry run pytest

ci: lint test  ## O que o CI roda
