"""
Dashboard Oráculo Preditivo — Preço de veículos usados (Índia)
Fatec Indaiatuba — Aprendizagem de Máquina

Rodar (a partir da pasta do projeto, com o ambiente virtual ativo):
    streamlit run app-Aula32.py
ou  python -m streamlit run app-Aula32.py

Arquivo de dados esperado na mesma pasta: Vehicle-DS.csv

IMPORTANTE (unidade): Selling_Price e Present_Price estão em LAKHS de rupias
indianas (1 lakh = ₹100.000), e NÃO em dólares.
"""
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import (mean_absolute_error, mean_absolute_percentage_error,
                             mean_squared_error, r2_score)
from sklearn.model_selection import RepeatedKFold, cross_validate, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.svm import SVR

st.set_page_config(page_title="Oráculo Preditivo — Veículos Usados (Índia)",
                   page_icon="🚗", layout="wide")

# ----------------------------------------------------------------------------
# CONSTANTES (iguais às do notebook analise_veiculos.ipynb)
# ----------------------------------------------------------------------------
ARQUIVO = "Vehicle-DS.csv"
ANO_REFERENCIA = 2019          # Car_Age = 2019 - Year (último ano do dataset = 2018)
LAKH = 100_000                 # 1 lakh = ₹100.000
NUMERICAS = ["Present_Price", "Kms_Driven", "Owner", "Car_Age"]
CATEGORICAS = ["Fuel_Type", "Seller_Type", "Transmission"]
REGEX_MOTO = r"^(bajaj|hero|honda|yamaha|tvs|ktm|royal|hyosung|mahindra|suzuki|um|activa)\b"
SEMENTE = 42
CV_FOLDS, CV_REPETICOES = 5, 3  # RepeatedKFold 5x3, como no notebook

# Hiperparâmetros encontrados com GridSearchCV no notebook (só no treino)
PARAMS_RFR = dict(n_estimators=200, max_depth=None, min_samples_leaf=1, random_state=SEMENTE)
PARAMS_SVR = dict(kernel="rbf", C=300, epsilon=0.1, gamma=0.01)


# ----------------------------------------------------------------------------
# DADOS
# ----------------------------------------------------------------------------
@st.cache_data
def carregar_dados():
    bruto = pd.read_csv(ARQUIVO)

    auditoria = {
        "linhas_brutas": len(bruto),
        "nulos": int(bruto.isna().sum().sum()),
        "duplicadas": int(bruto.duplicated().sum()),
        "nomes_espaco_duplo": int(bruto["Car_Name"].str.contains(r"\s{2,}").sum()),
        "nomes_espaco_borda": int((bruto["Car_Name"] != bruto["Car_Name"].str.strip()).sum()),
        "nomes_maiusculas_mistas": int((bruto["Car_Name"] != bruto["Car_Name"].str.lower()).sum()),
    }

    df = bruto.drop_duplicates().reset_index(drop=True)
    df["Car_Name"] = (df["Car_Name"].str.strip().str.lower()
                      .str.replace(r"\s+", " ", regex=True))
    df["Tipo"] = np.where(df["Car_Name"].str.match(REGEX_MOTO), "Moto", "Carro")
    df["Car_Age"] = ANO_REFERENCIA - df["Year"]
    df["Retencao"] = df["Selling_Price"] / df["Present_Price"]   # preço de venda / preço novo
    auditoria["linhas_limpas"] = len(df)
    return df, auditoria


df, auditoria = carregar_dados()
carros = df[df["Tipo"] == "Carro"].reset_index(drop=True)


# ----------------------------------------------------------------------------
# MODELOS (somente carros)
# ----------------------------------------------------------------------------
def montar_pipeline(modelo, escalar):
    pre = ColumnTransformer([
        ("cat", OneHotEncoder(drop="first", handle_unknown="ignore"), CATEGORICAS),
        ("num", StandardScaler() if escalar else "passthrough", NUMERICAS),
    ])
    return Pipeline([("pre", pre), ("modelo", modelo)])


def criar_modelos():
    return {
        "Regressão Linear Múltipla": montar_pipeline(LinearRegression(), True),
        "Random Forest": montar_pipeline(RandomForestRegressor(**PARAMS_RFR), False),
        "SVR": montar_pipeline(SVR(**PARAMS_SVR), True),
    }


def metricas(real, pred):
    return {
        "R²": r2_score(real, pred),
        "MAE": mean_absolute_error(real, pred),
        "MAPE (%)": mean_absolute_percentage_error(real, pred) * 100,
        "RMSE": float(np.sqrt(mean_squared_error(real, pred))),
    }


@st.cache_resource
def treinar_e_avaliar(_carros, assinatura):
    """assinatura serve só para o cache invalidar se os dados mudarem."""
    X, y = _carros[NUMERICAS + CATEGORICAS], _carros["Selling_Price"]
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=SEMENTE)
    cv = RepeatedKFold(n_splits=CV_FOLDS, n_repeats=CV_REPETICOES, random_state=SEMENTE)

    # carro de maior preço real no teste (análise de sensibilidade, Bloco 13)
    pos_extremo = int(np.argmax(y_te.values))
    nome_extremo = _carros.loc[y_te.index[pos_extremo], "Car_Name"]
    mascara = np.ones(len(y_te), dtype=bool)
    mascara[pos_extremo] = False

    linhas, previsoes, finais, treino_split, sens = [], {}, {}, {}, []
    for nome, pipe in criar_modelos().items():
        pipe.fit(X_tr, y_tr)
        treino_split[nome] = pipe             # modelo treinado só no treino (80%)
        pred_te, pred_tr = pipe.predict(X_te), pipe.predict(X_tr)
        r = cross_validate(pipe, X, y, cv=cv,
                           scoring=("r2", "neg_mean_absolute_error",
                                    "neg_mean_absolute_percentage_error"))
        m = metricas(y_te, pred_te)
        r2_tr = r2_score(y_tr, pred_tr)
        linhas.append({
            "Modelo": nome,
            "R² teste": m["R²"], "MAE teste": m["MAE"],
            "MAPE teste (%)": m["MAPE (%)"], "RMSE teste": m["RMSE"],
            "R² treino": r2_tr,
            "R² CV (média)": r["test_r2"].mean(), "R² CV (desvio)": r["test_r2"].std(),
            "MAE CV": -r["test_neg_mean_absolute_error"].mean(),
            "MAPE CV (%)": -r["test_neg_mean_absolute_percentage_error"].mean() * 100,
            "Gap (treino − teste)": r2_tr - m["R²"],
            "RMSE/MAE (teste)": m["RMSE"] / m["MAE"],
        })
        previsoes[nome] = pd.DataFrame({"real": y_te.values, "previsto": pred_te})

        # sensibilidade: métricas sem o carro de maior preço do teste
        res = y_te.values - pred_te
        m_sem = metricas(y_te.values[mascara], pred_te[mascara])
        sens.append({
            "Modelo": nome,
            "Previsto (carro extremo)": pred_te[pos_extremo],
            "Resíduo": res[pos_extremo],
            "% do erro quadrático": 100 * res[pos_extremo] ** 2 / np.sum(res ** 2),
            "R² sem ele": m_sem["R²"], "MAE sem ele": m_sem["MAE"],
            "RMSE sem ele": m_sem["RMSE"],
        })

        final = criar_modelos()[nome]
        final.fit(X, y)                       # modelo final (simulador): todos os carros
        finais[nome] = final
    info_extremo = {"nome": nome_extremo,
                    "real": float(y_te.values[pos_extremo]),
                    "present": float(X_te.iloc[pos_extremo]["Present_Price"]),
                    "ano": int(_carros.loc[y_te.index[pos_extremo], "Year"]),
                    "n_treino": len(X_tr), "n_teste": len(X_te)}
    return (pd.DataFrame(linhas), previsoes, finais, treino_split,
            pd.DataFrame(sens), info_extremo)


(tabela_modelos, previsoes, modelos_finais, modelos_treino,
 tabela_sens, extremo) = treinar_e_avaliar(carros, len(carros))


# ----------------------------------------------------------------------------
# MOEDA
# ----------------------------------------------------------------------------
st.sidebar.header("💱 Câmbio (ajuste à cotação do dia)")
inr_por_usd = st.sidebar.number_input("₹ por US$ 1", 50.0, 150.0, 95.0, 0.5)
brl_por_inr = st.sidebar.number_input("R$ por ₹ 1", 0.01, 0.20, 0.0548, 0.0001, format="%.4f")
st.sidebar.caption("Valores aproximados de set/2026, NÃO confirmados. Atualize antes de "
                   "apresentar. Preços do dataset: lakhs de rupias (1 lakh = ₹100.000) e "
                   "datados de ~2018: a conversão é nominal.")


def lakh_para_inr(v):
    return v * LAKH


def lakh_para_usd(v):
    return v * LAKH / inr_por_usd


def lakh_para_brl(v):
    return v * LAKH * brl_por_inr


def texto_valor(v):
    return (f"₹ {lakh_para_inr(v):,.0f}  |  US$ {lakh_para_usd(v):,.0f}  |  "
            f"R$ {lakh_para_brl(v):,.0f}").replace(",", ".")


# ----------------------------------------------------------------------------
# CABEÇALHO
# ----------------------------------------------------------------------------
st.title("🚗 Oráculo Preditivo: preço de veículos usados na Índia")
st.markdown(
    "Dataset **Car Price Prediction** (CarDekho). Os preços estão em **lakhs de rupias** "
    "(₹ 100.000). O dashboard converte para **US\\$** e **R\\$** usando o câmbio da barra lateral. "
    "Os modelos são treinados apenas com **carros** (as motos são separadas na limpeza)."
)

abas = st.tabs(["📋 Dados e qualidade", "📊 Análise exploratória",
                "🤖 Modelos", "🔮 Simulador", "💱 Valores ₹ / US\\$ / R\\$"])

# ----------------------------------------------------------------------------
# ABA 1 — DADOS E QUALIDADE
# ----------------------------------------------------------------------------
with abas[0]:
    st.subheader("Amostra dos dados limpos")
    n = st.slider("Linhas exibidas", 5, len(df), 10)
    st.dataframe(df.drop(columns=["Retencao"]).head(n))

    st.subheader("Auditoria de qualidade (arquivo bruto)")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Linhas brutas", auditoria["linhas_brutas"])
    c2.metric("Valores nulos", auditoria["nulos"])
    c3.metric("Linhas duplicadas", auditoria["duplicadas"])
    c4.metric("Linhas após limpeza", auditoria["linhas_limpas"])

    aud = pd.DataFrame({
        "Verificação": ["Nomes com espaço duplo", "Nomes com espaço no início/fim",
                        "Nomes com maiúsculas misturadas", "Duplicatas exatas removidas"],
        "Ocorrências": [auditoria["nomes_espaco_duplo"], auditoria["nomes_espaco_borda"],
                        auditoria["nomes_maiusculas_mistas"], auditoria["duplicadas"]],
        "Tratamento": ["Colapsar espaços", "strip()", "lower()", "drop_duplicates()"],
    })
    st.dataframe(aud)

    st.subheader("Carros × motos")
    resumo_tipo = (df.groupby("Tipo")
                   .agg(Veículos=("Car_Name", "size"),
                        Preço_médio_lakh=("Selling_Price", "mean"),
                        Preço_novo_médio_lakh=("Present_Price", "mean"),
                        Km_mediana=("Kms_Driven", "median"))
                   .round(2))
    st.dataframe(resumo_tipo)
    st.caption("O arquivo mistura carros e motos/scooters (Bajaj, Hero, Royal Enfield, "
               "Activa, TVS, Yamaha, KTM...). Como o negócio é revenda de carros, "
               "os modelos usam só `Tipo = Carro`. A separação usa uma expressão regular "
               "de marcas de moto (heurística não validada linha a linha).")

# ----------------------------------------------------------------------------
# ABA 2 — EDA
# ----------------------------------------------------------------------------
with abas[1]:
    tipos = st.multiselect("Tipos de veículo", ["Carro", "Moto"], default=["Carro"])
    d = df[df["Tipo"].isin(tipos)] if tipos else df

    col1, col2 = st.columns(2)
    with col1:
        fig = px.histogram(d, x="Selling_Price", nbins=30, color="Tipo",
                           title="Distribuição do preço de venda (lakh ₹)",
                           labels={"Selling_Price": "Preço de venda (lakh ₹)"})
        st.plotly_chart(fig, key="eda_hist")
        st.caption(f"Assimetria (skew) = {d['Selling_Price'].skew():.2f} → cauda à direita.")
    with col2:
        fig = px.scatter(d, x="Present_Price", y="Selling_Price", color="Fuel_Type",
                         symbol="Tipo", hover_name="Car_Name",
                         title="Preço novo × preço de venda (lakh ₹)",
                         labels={"Present_Price": "Preço novo (lakh ₹)",
                                 "Selling_Price": "Preço de venda (lakh ₹)"})
        st.plotly_chart(fig, key="eda_disp")

    col3, col4 = st.columns(2)
    with col3:
        fig = px.box(d, y="Kms_Driven", points="outliers", hover_name="Car_Name",
                     title="Boxplot da quilometragem (outliers destacados)")
        st.plotly_chart(fig, key="eda_box_km")
    with col4:
        fig = px.box(d, y="Present_Price", points="outliers", hover_name="Car_Name",
                     title="Boxplot do preço novo (lakh ₹)")
        st.plotly_chart(fig, key="eda_box_preco")

    st.subheader("Matriz de correlação")
    corr = d[["Selling_Price", "Present_Price", "Kms_Driven", "Owner", "Car_Age"]].corr()
    fig = px.imshow(corr, text_auto=".2f", color_continuous_scale="RdBu_r",
                    zmin=-1, zmax=1, aspect="auto")
    st.plotly_chart(fig, key="eda_corr")

    st.subheader("Depreciação: quanto do preço novo o carro ainda vale")
    dep = (carros.groupby("Car_Age")["Retencao"].agg(["median", "count"]).reset_index())
    fig = px.line(dep, x="Car_Age", y="median", markers=True, hover_data=["count"],
                  title="Retenção de valor mediana por idade (carros)",
                  labels={"Car_Age": "Idade (anos)", "median": "Preço venda ÷ preço novo"})
    fig.update_yaxes(range=[0, 1.05], tickformat=".0%")
    st.plotly_chart(fig, key="eda_dep")
    st.caption("Poucos carros nas idades mais altas e com 1 ano (veja `count` ao passar o "
               "mouse): essas pontas não são conclusivas.")

    st.subheader("Preço por categoria (carros)")
    cat = st.selectbox("Agrupar por", CATEGORICAS)
    fig = px.box(carros, x=cat, y="Selling_Price", points="all", hover_name="Car_Name",
                 labels={"Selling_Price": "Preço de venda (lakh ₹)"})
    st.plotly_chart(fig, key="eda_cat")
    if cat == "Seller_Type":
        st.warning(f"Entre os carros, apenas {int((carros['Seller_Type'] == 'Individual').sum())} "
                   "são de vendedor Individual: a variável quase não varia dentro dos carros. "
                   "No dataset completo ela só reflete 'moto × carro'.")

# ----------------------------------------------------------------------------
# ABA 3 — MODELOS
# ----------------------------------------------------------------------------
with abas[2]:
    st.subheader("Comparação dos três algoritmos (carros)")
    st.dataframe(tabela_modelos.round(3))
    st.caption(f"Métricas em lakh ₹. Teste = 20% dos carros ({extremo['n_teste']} carros; "
               f"treino = {extremo['n_treino']}). CV = RepeatedKFold {CV_FOLDS}×{CV_REPETICOES} "
               "nos carros inteiros (as linhas de teste entram nas dobras, então o R² de CV pode "
               "ser levemente otimista). Gap = R² treino − R² teste (indica overfitting).")

    melhor = tabela_modelos.sort_values("MAE CV").iloc[0]
    st.info(f"Menor erro médio na validação cruzada: **{melhor['Modelo']}** "
            f"(MAE CV = {melhor['MAE CV']:.2f} lakh ≈ R\\$ {lakh_para_brl(melhor['MAE CV']):,.0f}).")

    st.subheader("Valores reais × preditos (conjunto de teste)")
    cols = st.columns(3)
    for col, (nome, p) in zip(cols, previsoes.items()):
        with col:
            lim = [0, max(p["real"].max(), p["previsto"].max()) * 1.05]
            fig = px.scatter(p, x="real", y="previsto", title=nome,
                             labels={"real": "Real", "previsto": "Previsto"})
            fig.add_trace(go.Scatter(x=lim, y=lim, mode="lines", name="Ideal (45°)",
                                     line=dict(dash="dash", color="red")))
            st.plotly_chart(fig, key=f"rp_{nome}")

    st.subheader("Resíduos (real − previsto)")
    cols = st.columns(3)
    for col, (nome, p) in zip(cols, previsoes.items()):
        with col:
            r = p.assign(residuo=p["real"] - p["previsto"])
            fig = px.scatter(r, x="previsto", y="residuo", title=nome,
                             labels={"previsto": "Previsto", "residuo": "Resíduo"})
            fig.add_hline(y=0, line_dash="dash", line_color="red")
            st.plotly_chart(fig, key=f"res_{nome}")
    st.caption("Esperado: pontos aleatórios em torno de 0. Funil ou curva = erro sistemático "
               "(aqui, na cauda de preços altos).")

    st.subheader("Sensibilidade: o carro de maior preço do teste")
    st.write(f"Carro: **{extremo['nome']}** ({extremo['ano']}), preço novo "
             f"{extremo['present']:.1f} lakh, preço de venda real {extremo['real']:.1f} lakh.")
    st.dataframe(tabela_sens.round(3))
    st.caption("Análise pós-hoc (não substitui as métricas oficiais). O teste tem poucos "
               "carros, então um único carro pode inverter o ranking de R². Use CV, MAE e MAPE "
               "como base da comparação.")

    st.subheader("Interpretabilidade")
    c1, c2 = st.columns(2)
    with c1:
        pipe_rlm = modelos_treino["Regressão Linear Múltipla"]
        nomes = [n.split("__")[-1] for n in pipe_rlm.named_steps["pre"].get_feature_names_out()]
        coef = (pd.DataFrame({"variavel": nomes,
                              "coeficiente": pipe_rlm.named_steps["modelo"].coef_})
                .sort_values("coeficiente"))
        fig = px.bar(coef, x="coeficiente", y="variavel", orientation="h",
                     title="Coeficientes da RLM (modelo treinado no treino)")
        st.plotly_chart(fig, key="coef_rlm")
        st.caption(f"Intercepto = {pipe_rlm.named_steps['modelo'].intercept_:.3f}. "
                   "Numéricas padronizadas; dummies em lakh. Diesel/Petrol são relativos ao CNG "
                   "(2 carros) e Individual vem de poucos carros: não interpretar.")
    with c2:
        rf = modelos_treino["Random Forest"]
        nomes = [n.split("__")[-1] for n in rf.named_steps["pre"].get_feature_names_out()]
        imp = (pd.DataFrame({"variavel": nomes,
                             "importancia": rf.named_steps["modelo"].feature_importances_})
               .sort_values("importancia"))
        fig = px.bar(imp, x="importancia", y="variavel", orientation="h",
                     title="Importância das variáveis (Random Forest)")
        st.plotly_chart(fig, key="imp_rfr")
        st.caption("Importância por impureza favorece variáveis contínuas e divide crédito "
                   "entre variáveis correlacionadas.")

# ----------------------------------------------------------------------------
# ABA 4 — SIMULADOR
# ----------------------------------------------------------------------------
with abas[3]:
    st.subheader("Simulador: quanto vale este carro usado?")
    c1, c2 = st.columns(2)
    with c1:
        present = st.number_input("Preço do carro novo (lakh ₹)", 1.0, 150.0,
                                  float(round(carros["Present_Price"].median(), 2)), 0.5)
        idade = st.slider("Idade (anos)", 0, 20, int(carros["Car_Age"].median()))
        kms = st.number_input("Quilometragem rodada (km)", 0, 500_000,
                              int(carros["Kms_Driven"].median()), 1000)
        owner = st.selectbox("Proprietários anteriores", sorted(carros["Owner"].unique()))
    with c2:
        fuel = st.selectbox("Combustível", sorted(carros["Fuel_Type"].unique()))
        seller = st.selectbox("Tipo de vendedor", sorted(carros["Seller_Type"].unique()))
        trans = st.selectbox("Transmissão", sorted(carros["Transmission"].unique()))
        st.caption(f"₹ {lakh_para_inr(present):,.0f} novo ≈ US\\$ {lakh_para_usd(present):,.0f} "
                   f"≈ R\\$ {lakh_para_brl(present):,.0f}".replace(",", "."))

    entrada = pd.DataFrame([{
        "Present_Price": present, "Kms_Driven": kms, "Owner": owner, "Car_Age": idade,
        "Fuel_Type": fuel, "Seller_Type": seller, "Transmission": trans}])

    fora = []
    for col in NUMERICAS:
        if entrada[col].iloc[0] > carros[col].max() or entrada[col].iloc[0] < carros[col].min():
            fora.append(col)
    if fora:
        st.warning("Valores fora da faixa vista no treino (" + ", ".join(fora) +
                   "). Random Forest não extrapola e a RLM extrapola em linha reta: "
                   "a previsão pode ser pouco confiável.")
    if present > 15:
        st.warning("Nenhum modelo foi confiável para preços altos (acima de ≈15–20 lakh "
                   "de preço de venda) no conjunto de teste.")

    if st.button("Estimar preço de revenda"):
        linhas = []
        for nome, modelo in modelos_finais.items():
            v = max(float(modelo.predict(entrada)[0]), 0.0)
            linhas.append({"Modelo": nome, "Lakh ₹": round(v, 2),
                           "₹": round(lakh_para_inr(v)),
                           "US$": round(lakh_para_usd(v)), "R$": round(lakh_para_brl(v))})
        res = pd.DataFrame(linhas)
        st.dataframe(res)
        fig = px.bar(res, x="Modelo", y="R$", text="R$", title="Estimativa em reais")
        st.plotly_chart(fig, key="sim_barras")
        st.caption("Previsões negativas são limitadas a zero. A diferença entre os modelos "
                   "mostra a incerteza da estimativa.")

# ----------------------------------------------------------------------------
# ABA 5 — VALORES ₹ / US$ / R$
# ----------------------------------------------------------------------------
with abas[4]:
    st.subheader("Preços médios convertidos")
    base = df.groupby("Tipo")[["Selling_Price", "Present_Price"]].mean()
    base.loc["Todos"] = df[["Selling_Price", "Present_Price"]].mean()
    out = []
    for tipo, linha in base.iterrows():
        for campo, rotulo in (("Present_Price", "Preço novo (médio)"),
                              ("Selling_Price", "Preço de venda (médio)")):
            v = linha[campo]
            out.append({"Grupo": tipo, "Indicador": rotulo, "Lakh ₹": round(v, 2),
                        "₹": round(lakh_para_inr(v)), "US$": round(lakh_para_usd(v)),
                        "R$": round(lakh_para_brl(v))})
    st.dataframe(pd.DataFrame(out))
    st.caption(f"1 lakh = ₹100.000 = US\\$ {lakh_para_usd(1):,.0f} = "
               f"R\\$ {lakh_para_brl(1):,.0f} (câmbio da barra lateral).".replace(",", "."))
    st.info("Comparar com preços no Brasil exige dados da FIPE e considerar impostos e "
            "tarifas diferentes nos dois países; aqui a conversão é apenas cambial.")