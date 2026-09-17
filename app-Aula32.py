import streamlit as st
import pandas as pd
import plotly.express as px
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

# Configuração da página do aplicativo
st.set_page_config(page_title="Preditor de Preço de Veículos", page_icon="🚗", layout="wide")

st.title("🚗 Dashboard Oráculo Preditivo: Previsão de Preços de Veículos")
st.markdown("Este aplicativo utiliza análise de dados e Machine Learning para estimar o valor de mercado de veículos com base em suas características.")

# Carregando o dataset de veículos
@st.cache_data
def carregar_dados():
    df = pd.read_csv("Vehicle-DS.csv")
    return df

df = carregar_dados()

# -------------------------------------------------------------
# TREINAMENTO AUTOMÁTICO DO MODELO DE MACHINE LEARNING
# -------------------------------------------------------------
@st.cache_resource
def treinar_modelo():
    X = df[['Year', 'Present_Price', 'Kms_Driven', 'Fuel_Type', 'Seller_Type', 'Transmission', 'Owner']]
    y = df['Selling_Price']
    
    categorical_features = ['Fuel_Type', 'Seller_Type', 'Transmission']
    
    preprocessor = ColumnTransformer(
        transformers=[
            ('cat', OneHotEncoder(drop='first', handle_unknown='ignore'), categorical_features)
        ],
        remainder='passthrough'
    )
    
    model = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('regressor', LinearRegression())
    ])
    
    model.fit(X, y)
    return model

modelo = treinar_modelo()

# -------------------------------------------------------------
# 1. SIDEBAR (BARRA LATERAL) - SIMULADOR INTERATIVO DE DADOS
# -------------------------------------------------------------
st.sidebar.header("🔍 Parâmetros do Veículo para Simulação")

def user_input_features():
    year = st.sidebar.slider("Ano do Veículo", int(df['Year'].min()), int(df['Year'].max()), int(df['Year'].median()))
    
    # Campo direto para o preço de tabela em dólares
    present_price = st.sidebar.number_input(
        "Preço de Tabela (US$)", 
        min_value=float(df['Present_Price'].min()), 
        max_value=float(df['Present_Price'].max()) * 1000, 
        value=float(df['Present_Price'].mean())
    )
    
    driven_kms = st.sidebar.number_input("Quilometragem Rodada (Kms_Driven)", int(df['Kms_Driven'].min()), int(df['Kms_Driven'].max()), int(df['Kms_Driven'].mean()))
    fuel_type = st.sidebar.selectbox("Tipo de Combustível", df['Fuel_Type'].unique())
    seller_type = st.sidebar.selectbox("Tipo de Vendedor (Seller_Type)", df['Seller_Type'].unique())
    transmission = st.sidebar.selectbox("Transmissão", df['Transmission'].unique())
    owner = st.sidebar.selectbox("Número de Proprietários Anteriores (Owner)", sorted(df['Owner'].unique()))
    
    data = {
        'Year': [year],
        'Present_Price': [present_price],
        'Kms_Driven': [driven_kms],
        'Fuel_Type': [fuel_type],
        'Seller_Type': [seller_type],
        'Transmission': [transmission],
        'Owner': [owner]
    }
    return pd.DataFrame(data)

df_usuario = user_input_features()

# -------------------------------------------------------------
# 2. CORPO PRINCIPAL DO DASHBOARD (ABAS)
# -------------------------------------------------------------
aba1, aba2, aba3 = st.tabs(["📊 Visão Geral dos Dados", "📈 Análise Exploratória & Gráficos", "🔮 Simulador de Predição"])

with aba1:
    st.subheader("Amostra do Dataset Bruto (`Vehicle-DS.csv`)")
    
    num_linhas = st.slider("Selecione a quantidade de registros para exibir na tabela:", 5, len(df), 10)
    st.dataframe(df.head(num_linhas), use_container_width=True)
    
    st.markdown(f"**Total de registros cadastrados na base:** {df.shape[0]} veículos.")

with aba2:
    st.subheader("Análise Visual e Distribuição de Preços")
    
    col1, col2 = st.columns(2)
    
    with col1:
        fig_scatter = px.scatter(
            df, x='Present_Price', y='Selling_Price', color='Fuel_Type',
            title="Preço de Tabela vs. Preço de Venda Real",
            labels={'Present_Price': 'Preço de Tabela (US$)', 'Selling_Price': 'Preço de Venda (US$)'}
        )
        st.plotly_chart(fig_scatter, use_container_width=True)
        
    with col2:
        fig_hist = px.histogram(
            df, x='Selling_Price', nbins=30,
            title="Distribuição da Variável Alvo (Preço de Venda)",
            labels={'Selling_Price': 'Preço de Venda (US$)'}
        )
        st.plotly_chart(fig_hist, use_container_width=True)

with aba3:
    st.subheader("⚙️ Dados Selecionados pelo Usuário")
    st.dataframe(df_usuario)
    
    if st.button("Estimar Preço de Mercado"):
        predicao = modelo.predict(df_usuario)
        
        # Exibição direta do valor estimado em dólares
        st.success(f"🎉 Valor de Mercado Estimado: **US$ {predicao[0]:,.2f}**")
        st.balloons()
        