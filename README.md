# Propensão de compra em e-commerce

Tech Challenge da Fase 2 da pós em Machine Learning Engineering (FIAP + Alura).
Pipeline de classificação que estima a propensão de compra de uma sessão a partir do
comportamento de navegação ([Online Shoppers Purchasing Intention](https://archive.ics.uci.edu/dataset/468/online+shoppers+purchasing+intention+dataset), UCI 468).

> 🚧 Em construção. Execução individual.

## Paradigma: funcional nos estágios, objetos onde há estado

Cada estágio do pipeline (`prepare`, `train`, `evaluate`) tem um núcleo de funções puras, testável
sem arquivo, e um `main()` curto que lê os parâmetros, carrega os dados, chama o núcleo e grava o
resultado. I/O e MLflow ficam só no `main()`.

A divisão em pipeline de dados, treino e avaliação é a mesma de um desenho com classes
`DataPipeline`, `ModelTrainer` e `ModelEvaluator`. A diferença é o veículo: o DVC roda cada estágio
num processo próprio, então um objeto não sobreviveria de um estágio para o outro, e o estado passa
por arquivos versionados (`outs` → `deps`).

Orientação a objetos entra onde o estado é real:

- o modelo é um `Pipeline` do scikit-learn, com o pré-processamento ajustado dentro dele;
- os estimadores são intercambiáveis por `fit`/`predict` (Strategy), escolhidos por nome no
  `params.yaml` a partir de um dicionário de construtores (Factory);
- os parâmetros são carregados uma vez para uma `dataclass` imutável e validados na carga.

Não há herança no projeto.

## Como rodar

```bash
make setup             # poetry install, a partir do poetry.lock
make ci                # lint + testes
poetry run dvc repro   # executa o pipeline
```

> O remote do DVC ainda não está configurado, então o `dvc pull` não funciona num clone novo.

## Pipeline

```
+--------------------------------------------+
| data/raw/online_shoppers_intention.csv.dvc |
+--------------------------------------------+
                       *
                  +---------+
                  | prepare |
                  +---------+
```

- **`prepare`**: seleciona as colunas do esquema, converte o alvo `Revenue` para 0/1 e separa
  treino e teste (80/20, estratificado, seed em `configs/params.yaml`). Saída em
  `data/processed/`. O enunciado usa `preprocess` como exemplo, mas este estágio não ajusta
  nada: o one-hot e o escalonamento ficam dentro do `Pipeline` do scikit-learn, no `train`, e
  são ajustados só no treino.
