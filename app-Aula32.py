import streamlit as st
import pandas as pd

# Configuração da página do aplicativo
st.set_page_config(page_title="Preditor de Preço de Veículos", page_icon="🚗", layout="wide")

st.title("🚗 Dashboard Oráculo Preditivo: Previsão de Preços de Veículos")
st.markdown("Este aplicativo utiliza análise de dados e Machine Learning para estimar o valor de mercado de veículos.")

# Carregando o dataset de veículos
@st.cache_data
def carregar_dados():
    df = pd.read_csv("Vehicle-DS.csv")
    return df

df = carregar_dados()

# Exibindo os dados na tela
st.subheader("📊 Visualização dos Dados Brutos")
st.dataframe(df.head())
