# Estudo do dataset — Online Shoppers Purchasing Intention

Reconhecimento do dataset feito antes do pipeline: o que cada coluna é, o que os números
mostram e onde estão as armadilhas. Os números das §3 e §4 vêm de uma análise exploratória
sobre o arquivo completo (28/09/2026), fora do pipeline. Os resultados do modelo da entrega
estão no README.

## 1. Identidade

| | |
|---|---|
| **Fonte** | UCI ML Repository, id 468 (`archive.ics.uci.edu/static/public/468/online+shoppers+purchasing+intention+dataset.zip`) |
| **Artigo de origem** | Sakar, Polat, Katircioglu & Kastro (2018), *Real-time prediction of online shoppers' purchasing intention using multilayer perceptron and LSTM recurrent neural networks*, Neural Computing and Applications |
| **Arquivo** | `online_shoppers_intention.csv`, 1,07 MB, 12.330 linhas × 18 colunas, MD5 `cc6ec1db03b4f10f8de52c56ff48b085` |
| **Licença** | CC BY 4.0 |

Uma linha é uma sessão de navegação numa loja online, ao longo de um ano. Os autores montaram o
dataset de modo que cada sessão pertence a um usuário diferente, para não enviesar para uma
campanha, data ou perfil. Não há ID de usuário, e por isso não existe vazamento por grupo (o
mesmo cliente no treino e no teste).

As métricas vêm do Google Analytics da loja. Algumas colunas não foram medidas na sessão: são
médias por página, calculadas pelo GA sobre o ano inteiro e atribuídas à sessão.

## 2. Dicionário

**Comportamento na sessão**

| Coluna | Tipo | O que é |
|---|---|---|
| `Administrative` | int (0–27) | páginas administrativas visitadas (conta, login, cadastro) |
| `Administrative_Duration` | float (s) | tempo total nessas páginas |
| `Informational` | int (0–24) | páginas informativas (sobre, contato, políticas) |
| `Informational_Duration` | float (s) | tempo total nelas |
| `ProductRelated` | int (0–705) | páginas de produto |
| `ProductRelated_Duration` | float (s) | tempo total nelas, até 63.973 s (~17,8 h) numa sessão |

**Métricas do Google Analytics**

| Coluna | Tipo | O que é |
|---|---|---|
| `BounceRates` | float (0–0,2) | média da taxa de rejeição das páginas visitadas |
| `ExitRates` | float (0–0,2) | média da taxa de saída das páginas visitadas |
| `PageValues` | float (0–361) | média do "valor de página" das páginas visitadas: quanto a página costuma aparecer antes de uma transação (ver §4.1) |

**Contexto**

| Coluna | Tipo | O que é |
|---|---|---|
| `SpecialDay` | float (0; 0,2…1,0) | proximidade de uma data comemorativa (1,0 = no dia). Só aparece em fevereiro e maio |
| `Month` | texto | mês da sessão. 10 meses, sem janeiro e abril; `June` por extenso, os outros abreviados |
| `OperatingSystems` | int 1–8 | código do sistema operacional (categoria) |
| `Browser` | int 1–13 | código do navegador (categoria) |
| `Region` | int 1–9 | código da região (categoria) |
| `TrafficType` | int 1–20 | código da origem do tráfego (categoria). O significado dos códigos não foi publicado |
| `VisitorType` | texto | `Returning_Visitor`, `New_Visitor` ou `Other` (85 linhas) |
| `Weekend` | booleano | sessão no fim de semana, como `TRUE`/`FALSE` no CSV |

**Alvo**

| Coluna | Tipo | O que é |
|---|---|---|
| `Revenue` | booleano | a sessão terminou em compra. 15,47% positivos (1.908 de 12.330) |

## 3. O que os números mostram

**Qualidade.** Nenhum nulo, nenhuma duração negativa, nenhuma sessão com 0 páginas e tempo
positivo.

**Desbalanceamento.** 15,5% de compra. Acurácia não serve: prever sempre "não" acerta 84,5%.

**`PageValues` domina.** Correlação de Spearman com o alvo de 0,63; a segunda maior é
`ExitRates`, com −0,25.

| | Sessões | Taxa de compra |
|---|---|---|
| `PageValues = 0` | 9.600 (78%) | 3,9% |
| `PageValues > 0` | 2.730 (22%) | 56,3% |

Sozinha, usada como score, a coluna dá ROC-AUC 0,863 e PR-AUC 0,62.

**Caudas longas.** Contagens e durações têm assimetria entre 2 e 7,6. `PageValues`,
`Informational*` e `SpecialDay` são zero na maioria das linhas.

**Redundância.** `BounceRates` × `ExitRates`: correlação de 0,91. `ProductRelated` ×
`ProductRelated_Duration`: 0,88 (Spearman).

**Sazonalidade.**

| Mês | Sessões | Compra |
|---|---|---|
| Fev | 184 | 1,6% |
| Mar | 1.907 | 10,1% |
| Mai | 3.364 | 10,9% |
| Jun–Out | 288–549 cada | 10–21% |
| Nov | 2.998 | 25,4% |
| Dez | 1.727 | 12,5% |

Março, maio, novembro e dezembro concentram 81% das sessões.

**`SpecialDay`.** Perto de uma data comemorativa a compra cai (16,5% em dias comuns, 3–8% com
`SpecialDay > 0`). Como a coluna só existe em fevereiro e maio, o efeito se confunde com o mês.

**Visitante novo compra mais que o recorrente** (24,9% contra 13,9%).

**`TrafficType` tem sinal** (de 5,8% no tipo 13 a 27,7% no tipo 8); `Region` e `Browser`
quase nenhum (região entre 12,9% e 16,8%).

## 4. Armadilhas e como o pipeline trata cada uma

### 4.1 `PageValues` é quase-vazamento

O valor de página do GA é calculado a partir das transações, sobre o ano inteiro, incluindo a
própria sessão que se quer prever. Na prática, é um *target encoding* das páginas feito sem
separar treino e teste.

Não é vazamento total: 19% dos compradores têm `PageValues = 0`, e 11% dos não compradores têm
valor positivo. Em produção, o valor de cada página pode ser calculado só com o histórico
passado e estar disponível durante a sessão, que é o cenário do artigo.

Efeito medido na análise exploratória (CV estratificada de 5 dobras, seed 42):

| Modelo | PR-AUC com `PageValues` | PR-AUC sem `PageValues` |
|---|---|---|
| Dummy | 0,155 | 0,155 |
| Regressão logística | 0,667 | 0,333 |
| Random Forest | 0,730 | 0,379 |
| HistGradientBoosting | 0,745 | 0,363 |

Sem a coluna, o PR-AUC cai pela metade, e o ROC-AUC vai de ~0,93 para ~0,78.

**No pipeline:** a coluna fica no modelo, e a limitação está no README. O parâmetro
`train.use_page_values` permite rodar o experimento sem ela
(`dvc exp run -f -S configs/params.yaml:train.use_page_values=false`).

### 4.2 Códigos inteiros que são categorias

`OperatingSystems`, `Browser`, `Region` e `TrafficType` vêm como inteiros. Tratados como
número, a região 9 valeria nove vezes a região 1. Há categorias com uma linha só
(`Browser=9`, `TrafficType=12` e `17`).

**No pipeline:** one-hot, com as categorias raras agrupadas (`min_frequency` no
`params.yaml`) e categoria nunca vista tratada como infrequente, dentro do `Pipeline` do
scikit-learn.

### 4.3 `Month`: meses ausentes e deslocamento temporal

Janeiro e abril não existem, e o arquivo não está em ordem cronológica. Treinando com
fevereiro a outubro e testando em novembro e dezembro, o PR-AUC fica em 0,65 e o ROC-AUC em
0,82 (contra ~0,93 no split aleatório), com a prevalência subindo de 12,3% para 20,7%.

**No pipeline:** split estratificado aleatório com seed fixa (não há data, só mês). O encoder
aceita meses que não viu. A queda no split temporal está nas limitações do README.

### 4.4 Tempo zero não é erro

717 sessões têm páginas de produto com duração 0 (135 nas administrativas, 226 nas
informativas). O GA mede o tempo de uma página pela abertura da seguinte, então a última
página da sessão conta 0, e uma sessão de página única tem tempo total 0.

**No pipeline:** sem imputação nem remoção.

### 4.5 Sessões de rejeição e duplicadas

700 sessões têm `BounceRates = ExitRates = 0,2`, o teto: sessões de uma página, com 0,4% de
compra. As 125 linhas duplicadas são todas desse tipo (uma página, 0 s, sem compra): sessões
diferentes com o mesmo comportamento trivial, não registros repetidos.

**No pipeline:** mantidas. Removê-las mudaria a prevalência.

### 4.6 Booleanos como texto

`Weekend` e `Revenue` chegam como `TRUE`/`FALSE`.

**No pipeline:** o `prepare` converte o alvo para 0/1 em código, e a amostra de teste em
`tests/fixtures/` é do arquivo real, com o mesmo formato.

### 4.7 O que o dataset não tem

Não há timestamp (só mês), ID de usuário, valor da compra nem produto. Não é possível fazer
split temporal fino nem medir o resultado em dinheiro.

## 5. O que esperar do modelo

Na análise exploratória, Random Forest e HistGradientBoosting chegam a PR-AUC ~0,73–0,75 e
ROC-AUC ~0,93, e a distância para a regressão logística é de ~0,08 de PR-AUC. Esse é o teto
prático do dataset com as colunas disponíveis, e por isso o projeto compara poucos candidatos
com os hiperparâmetros padrão, sem tuning.

A métrica principal é o PR-AUC, por causa do desbalanceamento e do interesse na classe
positiva. Um limiar para transformar a probabilidade em decisão (por exemplo, as X% sessões
mais prováveis recebem um cupom) depende de uma regra de negócio que o dataset não traz.
