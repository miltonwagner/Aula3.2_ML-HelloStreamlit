# 🚗 Desafio do Oráculo Preditivo — Previsão de Preços de Veículos Usados

Projeto acadêmico da disciplina de **Aprendizagem de Máquina** — Fatec Indaiatuba Dr. Archimedes Lammoglia.

Comparação de três modelos de regressão (Regressão Linear Múltipla, Random Forest e SVR) para prever o preço de venda de **carros usados**, com validação cruzada, análise de resíduos e um dashboard interativo em **Streamlit**.

> ⚠️ **Unidade dos preços:** os valores do dataset estão em **lakhs de rupias indianas** (1 lakh = ₹ 100.000), e não em dólar. Os dados são de aproximadamente 2018.

---

## Sumário

- [Dataset](#dataset)
- [Metodologia](#metodologia)
- [Resultados](#resultados)
- [Análise do outlier (Land Cruiser)](#análise-do-outlier-land-cruiser)
- [Veredito do grupo](#veredito-do-grupo)
- [Limitações](#limitações)
- [Dashboard (Streamlit)](#dashboard-streamlit)
- [Estrutura do repositório](#estrutura-do-repositório)
- [Como executar](#como-executar)
- [Tecnologias](#tecnologias)
- [Equipe](#equipe)
- [Referências](#referências)

---

## Dataset

- **Fonte:** Car Price Prediction (CarDekho), Kaggle — arquivo `Vehicle-DS.csv`.
- **Bruto:** 301 linhas × 9 colunas, sem valores nulos e com 2 duplicatas.
- **Após limpeza:** 299 linhas = **198 carros + 101 motos**.
- **Escopo dos modelos:** apenas os **198 carros**. O campo `Tipo` (carro/moto) foi criado por regex de marcas de moto; trata-se de uma heurística **não validada linha a linha**.
- **Divisão treino/teste:** 80/20 com `random_state=42` → **158 treino / 40 teste**.

**Variáveis**

| Variável | Descrição |
|---|---|
| `Selling_Price` | Preço de venda (alvo), em lakhs |
| `Present_Price` | Preço do modelo novo, em lakhs |
| `Kms_Driven` | Quilometragem rodada |
| `Owner` | Número de proprietários anteriores |
| `Car_Age` | Idade do veículo (`2019 − Year`), feature derivada |
| `Fuel_Type` | Combustível (Petrol, Diesel, CNG) |
| `Seller_Type` | Tipo de vendedor (Dealer, Individual) |
| `Transmission` | Transmissão (Manual, Automatic) |

**Distribuição dos carros:** Seller 193 Dealer / 5 Individual; Fuel 138 Petrol / 58 Diesel / 2 CNG; Transmission 168 Manual / 30 Automatic.

**`Selling_Price` (carros):** média 6,60; desvio 5,04; mediana 5,25; máximo 35,00; assimetria ≈ 2,7 (cauda longa à direita). Outliers pelo critério 1,5×IQR: 16 / 11 / 6 nas três variáveis analisadas. **Decisão: mantê-los.**

**Correlação com `Selling_Price`:** `Present_Price` +0,82; `Car_Age` −0,35; `Kms_Driven` −0,11; `Owner` −0,08. Correlação entre `Kms_Driven` e `Car_Age`: +0,71. **VIF:** Present_Price 1,10; Kms_Driven 2,23; Owner 1,04; Car_Age 2,07 (sem multicolinearidade severa).

---

## Metodologia

**Pré-processamento (dentro de `Pipeline` / `ColumnTransformer`)**
- Numéricas: `Present_Price`, `Kms_Driven`, `Owner`, `Car_Age`.
- Categóricas: `Fuel_Type`, `Seller_Type`, `Transmission` com `OneHotEncoder(drop="first", handle_unknown="ignore")`.
- `StandardScaler` apenas em **RLM** e **SVR**. O **RFR** não usa escala.
- Outliers mantidos; por isso `StandardScaler` foi preferido ao `MinMaxScaler` (o MinMax não foi testado).

**Modelos**
1. **Regressão Linear Múltipla (RLM)**: baseline interpretável.
2. **Random Forest Regressor (RFR)**: ensemble de árvores (bagging).
3. **Support Vector Regressor (SVR)**: kernels linear e RBF comparados; o RBF foi o escolhido pela busca.

**Ajuste de hiperparâmetros:** `GridSearchCV` com 5 folds e `scoring` MAE, **somente nos dados de treino**.
- RFR: `n_estimators=200`, `max_depth=None`, `min_samples_leaf=1`.
- SVR: `kernel=rbf`, `C=300` (**maior valor da grade**, portanto o ótimo pode estar além dela), `epsilon=0,1`, `gamma=0,01`.

**Validação cruzada final:** `RepeatedKFold` 5×3 (`random_state=42`) sobre os 198 carros.

---

## Resultados

Valores em **lakhs**, obtidos na execução do notebook. Gap = R² treino − R² teste.

| Modelo | R² teste | MAE | MAPE (%) | RMSE | R² treino | R² CV (±) | MAE CV | Gap |
|---|---|---|---|---|---|---|---|---|
| RLM | 0,799 | 1,395 | 24,470 | 2,605 | 0,884 | 0,823 ± 0,043 | 1,360 | 0,085 |
| RFR | 0,790 | 1,122 | 16,081 | 2,663 | 0,991 | 0,869 ± 0,066 | 0,940 | 0,202 |
| SVR | 0,587 | 1,309 | 16,971 | 3,733 | 0,974 | 0,842 ± 0,150 | 0,918 | 0,387 |

**Leitura dos resultados**
- No **teste**, a RLM tem o maior R² e o menor RMSE, enquanto o RFR tem o menor MAE e o menor MAPE.
- Na **validação cruzada**, o RFR tem o maior R² e os três modelos ficam próximos em MAE; o SVR é o mais instável (± 0,150).
- O **gap** cresce da RLM para o RFR e o SVR, o que indica mais sobreajuste nos dois últimos.

**Coeficientes da RLM** (intercepto 6,324): Present_Price +3,757; Car_Age −1,182; Kms_Driven −1,158; Owner −0,418; Transmission_Manual −0,637; Fuel_Petrol +0,295; Fuel_Diesel +2,088; Seller_Individual +2,075.
Os coeficientes de combustível são relativos à categoria CNG (apenas 2 carros) e o de Seller_Individual se apoia em 5 carros; **não devem ser interpretados** como efeito causal.

**Importância no RFR:** Present_Price 0,799; Car_Age 0,144; Kms_Driven 0,050; demais ≈ 0.

---

## Análise do outlier (Land Cruiser)

Um Land Cruiser 2010 no conjunto de teste (preço real 35,0; `Present_Price` 92,6) domina o erro quadrático:

| Modelo | Previsão (real = 35,0) | % do erro quadrático total |
|---|---|---|
| RLM | 48,96 | 71,8% |
| RFR | 19,60 | 83,5% |
| SVR | 12,11 | 94,0% |

**Métricas sem esse carro (39 linhas de teste):**

| Modelo | R² | MAE | RMSE |
|---|---|---|---|
| RLM | 0,849 | 1,073 | 1,402 |
| RFR | 0,908 | 0,756 | 1,095 |
| SVR | 0,934 | 0,756 | 0,925 |

O **ranking de R² se inverte** sem esse carro: o resultado do teste depende fortemente de uma única observação. A menor previsão da RLM foi −0,17 (preço negativo, impossível), consequência de a regressão linear não ter limite inferior.

---

## Veredito do grupo

**Não há vencedor absoluto.** A escolha depende do critério:

- **RFR** — apoio à precificação: menor MAE e MAPE no teste, melhor R² em validação cruzada.
- **RLM** — baseline explicativo: mais simples, interpretável e com o menor gap.
- **SVR** — **descartado**: pior R² no teste, maior gap e maior instabilidade entre dobras.

---

## Limitações

- **Poucos dados:** 198 carros (40 no teste), com categorias muito desbalanceadas (CNG: 2; Individual: 5).
- **Teste sensível a outlier:** ver seção do Land Cruiser.
- **Validação cruzada levemente otimista:** as 40 linhas de teste participam das dobras do `RepeatedKFold`.
- **Cauda alta:** os resíduos indicam padrão na faixa de preços altos; os modelos não capturaram toda a informação ali.
- **SVR:** `C=300` está no limite da grade de busca.
- **Heurística carro/moto** não validada linha a linha.
- **Custo de predição não foi medido** (tempo de inferência de RLM, RFR e SVR).
- **Conversão monetária:** os preços são nominais, de ~2018. Qualquer conversão para R$ ou US$ usa câmbio **aproximado e não confirmado** e serve só como ordem de grandeza.

---

## Dashboard (Streamlit)

O arquivo `app-Aula32.py` reúne, em abas: exploração dos dados, gráficos (distribuição de preços, correlação, resíduos, reais × preditos), tabela comparativa dos modelos com validação cruzada 5×3, MAPE e gap, sensibilidade do Land Cruiser, coeficientes da RLM, importância do RFR e simulador de predição.

---

## Estrutura do repositório

> **[CONFIRMAR]** Ajustar com a listagem real da raiz do projeto antes do commit.

```
Aula3.2_ML-HelloStreamlit/
├── app-Aula32.py                  # Dashboard Streamlit
├── Vehicle-DS.csv                 # Dataset
├── requirements.txt               # Dependências
├── README.md                      # Este arquivo
├── README.txt                     # Instruções rápidas para o cliente
├── Relatorio_Machine_Learn.pdf    # Relatório técnico  [CONFIRMAR nome atual]
├── <notebook>.ipynb               # Notebook de análise  [CONFIRMAR nome]
└── Entrega_Cliente/               # Pacote de entrega  [CONFIRMAR conteúdo]
```

---

## Como executar

**Pré-requisitos:** Python 3.x e Git.

1. Clone o repositório:
   ```bash
   git clone https://github.com/miltonwagner/Aula3.2_ML-HelloStreamlit.git
   cd Aula3.2_ML-HelloStreamlit
   ```

2. Crie o ambiente virtual:
   ```bash
   python -m venv .venv-puro
   ```

3. Ative o ambiente.

   Windows (PowerShell):
   ```powershell
   Set-ExecutionPolicy -Scope Process -ExecutionPolicy RemoteSigned
   .\.venv-puro\Scripts\Activate.ps1
   ```
   Linux/macOS:
   ```bash
   source .venv-puro/bin/activate
   ```

4. Instale as dependências:
   ```bash
   pip install -r requirements.txt
   ```

5. Execute o dashboard (a partir da raiz do projeto):
   ```bash
   streamlit run app-Aula32.py
   ```

6. Abra `http://localhost:8501` no navegador.

---

## Tecnologias

Python 3 · Streamlit · Pandas · NumPy · Scikit-Learn (Pipeline, ColumnTransformer, RLM, Random Forest, SVR, GridSearchCV, RepeatedKFold) · Plotly / Matplotlib *(confirmar quais bibliotecas o app usa)*

---

## Equipe

- Gustavo Molina
- João Emanoel
- Milton Wagner
- Murilo

**Instituição:** Faculdade de Tecnologia de Indaiatuba Dr. Archimedes Lammoglia (Fatec Indaiatuba) — Indaiatuba, 2026.

---

## Referências

- Dataset: Car Price Prediction (CarDekho), Kaggle — *[colar o link do PDF do professor]*.
- Material didático da disciplina de Aprendizagem de Máquina — Fatec Indaiatuba.
- Documentação do Scikit-Learn: Pipeline, ColumnTransformer, modelos lineares, Random Forest e SVR.
- Documentação do Streamlit, Pandas e Plotly.

---

🔗 [github.com/miltonwagner/Aula3.2_ML-HelloStreamlit](https://github.com/miltonwagner/Aula3.2_ML-HelloStreamlit)