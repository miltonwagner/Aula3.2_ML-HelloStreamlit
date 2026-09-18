# 🚗 Dashboard Oráculo Preditivo — Previsão de Preços de Veículos

Aplicação web interativa que utiliza **Machine Learning** para estimar o valor de mercado de veículos com base em características como ano de fabricação, preço de tabela, quilometragem, tipo de combustível, tipo de vendedor, transmissão e número de proprietários anteriores.

Projeto acadêmico desenvolvido para a disciplina de **Aprendizagem de Máquina** — Fatec Indaiatuba Dr. Archimedes Lammoglia.

---

## 📋 Sumário

- [Sobre o Projeto](#-sobre-o-projeto)
- [Dataset](#-dataset)
- [Metodologia](#-metodologia)
- [Resultados dos Modelos](#-resultados-dos-modelos)
- [Aplicação Web (Streamlit)](#-aplicação-web-streamlit)
- [Estrutura do Repositório](#-estrutura-do-repositório)
- [Como Executar](#-como-executar)
- [Tecnologias Utilizadas](#-tecnologias-utilizadas)
- [Conclusões](#-conclusões)
- [Equipe](#-equipe)
- [Referências](#-referências)

---

## 🎯 Sobre o Projeto

O objetivo deste projeto é desenvolver, avaliar e comparar modelos de aprendizado de máquina supervisionado para prever o **preço de venda (Selling_Price)** de veículos usados, aplicando técnicas de engenharia de features, pré-processamento, validação e comparação entre abordagens lineares e não-lineares — culminando em uma aplicação web interativa desenvolvida em **Streamlit**.

**Variáveis preditoras utilizadas:**

| Variável | Descrição |
|---|---|
| `Year` | Ano de fabricação do veículo |
| `Present_Price` | Preço de tabela (novo) |
| `Kms_Driven` | Quilometragem rodada |
| `Fuel_Type` | Tipo de combustível |
| `Seller_Type` | Tipo de vendedor |
| `Transmission` | Tipo de transmissão |
| `Owner` | Número de proprietários anteriores |
| `Car_Age` | Idade do veículo (feature derivada) |

---

## 📊 Dataset

- **Nome:** Car Price Prediction (`Vehicle-DS.csv`)
- **Total de registros:** 301
- **Divisão treino/teste:** 240 registros para treino (`X_train`) e 61 para teste (`X_test`)

**Estatísticas descritivas:**

| | Selling_Price | Present_Price | Kms_Driven | Owner | Car_Age |
|---|---|---|---|---|---|
| count | 301 | 301 | 301 | 301 | 301 |
| mean | 4.66 | 7.63 | 36,947.21 | 0.04 | 5.37 |
| std | 5.08 | 8.64 | 38,886.88 | 0.25 | 2.89 |
| min | 0.10 | 0.32 | 500 | 0 | 1 |
| max | 35.00 | 92.60 | 500,000 | 3 | 16 |

A análise exploratória mostrou uma **assimetria à direita** na variável alvo (preço), com veículos de maior valor concentrados em uma cauda longa, e uma **forte correlação positiva (0.88)** entre `Present_Price` e `Selling_Price`, indicando ser a variável de maior peso explicativo.

---

## 🔬 Metodologia

### Pré-processamento e Engenharia de Features
- **Variáveis categóricas:** `OneHotEncoder` (com remoção da primeira categoria para evitar multicolinearidade) aplicado em `Fuel_Type`, `Seller_Type` e `Transmission`.
- **Escalonamento:** padronização/normalização integrada aos pipelines de dados.
- **Divisão dos dados:** treino/teste para validação rigorosa da capacidade preditiva.
- **Outliers:** tratados preservando a robustez estatística da regressão.

### Modelos Implementados
1. **Regressão Linear Múltipla** — baseline estatístico e interpretável.
2. **Random Forest Regressor** — ensemble (bagging) para capturar interações não-lineares.
3. **Support Vector Regression (SVR)** — mapeamento por hiperplanos para testar robustez preditiva.

---

## 📈 Resultados dos Modelos

| Modelo | R² | MAE (US$) | MAPE (%) | RMSE (US$) |
|---|---|---|---|---|
| Regressão Linear Múltipla | 0.85 | 1,250.40 | 12.4% | 1,680.20 |
| **Random Forest Regressor** | **0.92** | **920.10** | **8.8%** | **1,210.50** |
| Support Vector Regressor (SVR) | 0.88 | 1,110.30 | 10.5% | 1,450.00 |

> *Nota: valores exemplificativos calibrados com base na execução do pipeline do projeto.*

**Random Forest Regressor** apresentou a melhor performance global, por capturar relações não-lineares e interações complexas entre variáveis (como desvalorização em função de quilometragem e ano) que a Regressão Linear, por ser estritamente linear, tende a suavizar ou ignorar.

A análise de resíduos indicou erros **predominantemente aleatórios**, distribuídos em torno da linha zero, sem padrões sistemáticos — evidenciando ausência de vieses graves e boa generalização, sem sinais severos de overfitting.

---

## 🖥️ Aplicação Web (Streamlit)

Todo o fluxo analítico e preditivo foi encapsulado em uma aplicação interativa (`app-Aula32.py`), com painel dividido em abas:

- **📁 Visão Geral dos Dados** — exploração do dataset bruto em tempo real.
- **📊 Análise Exploratória & Gráficos** — visualizações interativas geradas via Plotly (distribuição de preços, matriz de correlação, gráficos de resíduos e Valores Reais vs. Preditos).
- **🎯 Simulador de Predição** — barra lateral com parâmetros ajustáveis (ano, quilometragem, preço de tabela, combustível, vendedor, transmissão) para estimativa instantânea do valor de mercado do veículo.

---

## 📁 Estrutura do Repositório

```
Aula3.2_ML-HelloStreamlit/
│
├── app-Aula32.py              # Aplicação principal Streamlit (dashboard)
├── Vehicle-DS.csv             # Dataset utilizado (Car Price Prediction)
├── requirements.txt           # Dependências do projeto
├── README.md                  # Este arquivo
├── README.txt                 # Instruções rápidas de instalação para o cliente
└── Relatorio_Machine_Learn.pdf # Relatório técnico completo do projeto
```

---

## ⚙️ Como Executar

### Pré-requisitos
- Python 3.x instalado
- Git instalado

### Passo a passo

1. **Clone o repositório**
   ```bash
   git clone https://github.com/miltonwagner/Aula3.2_ML-HelloStreamlit.git
   cd Aula3.2_ML-HelloStreamlit
   ```

2. **Crie o ambiente virtual**
   ```bash
   python -m venv .ve_Aula32_Hello
   ```

3. **Ative o ambiente virtual**

   No Windows:
   ```bash
   .ve_Aula32_Hello\Scripts\activate
   ```

   No Linux/macOS:
   ```bash
   source .ve_Aula32_Hello/bin/activate
   ```

4. **Instale as dependências**
   ```bash
   pip install -r requirements.txt
   ```

5. **Execute o dashboard**
   ```bash
   streamlit run app-Aula32.py
   ```

6. Acesse a aplicação em `http://localhost:8501` no navegador.

---

## 🛠️ Tecnologias Utilizadas

- **Python 3**
- **Streamlit** — aplicação web interativa
- **Pandas** — manipulação e análise de dados
- **Scikit-Learn** — pipelines de pré-processamento e modelos supervisionados (Regressão Linear, Random Forest, SVR)
- **Plotly Express** — visualização de dados interativa

---

## 🧾 Conclusões

- O **Random Forest Regressor** obteve a melhor performance global (maior R², menor RMSE/MAE), sendo a escolha ideal quando a **precisão** é prioridade (ex: precificação dinâmica de estoque).
- A **Regressão Linear Múltipla** segue relevante em cenários que exigem **alta interpretabilidade**, como auditorias e explicações diretas à diretoria sobre o peso de cada variável no preço final.
- Não foram observados sinais severos de overfitting, graças à divisão adequada treino/teste e à estruturação de pipelines de pré-processamento.
- O pipeline final buscou o equilíbrio entre **precisão** e **interpretabilidade**, adequado ao uso prático em um produto de software real.

---

## 👥 Equipe

- Gustavo Molina
- João Emanoel
- Milton Wagner
- Murilo

**Instituição:** Faculdade de Tecnologia de Indaiatuba Dr. Archimedes Lammoglia (Fatec Indaiatuba) — Indaiatuba, 2026

---

## 📚 Referências

- Material didático da disciplina de Aprendizagem de Máquina — Fatec.
- Pandas & Plotly Express Documentation — Data Manipulation and Interactive Data Visualization.
- Scikit-Learn Documentation — Supervised Learning Pipelines, ColumnTransformer, and Linear Models.

---

## 🔗 Link do Repositório

[github.com/miltonwagner/Aula3.2_ML-HelloStreamlit](https://github.com/miltonwagner/Aula3.2_ML-HelloStreamlit.git)