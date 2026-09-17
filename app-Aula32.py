import streamlit as st
import pandas as pd
import plotly.express as px

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
# 1. SIDEBAR (BARRA LATERAL) - SIMULADOR INTERATIVO DE DADOS
# -------------------------------------------------------------
st.sidebar.header("🔍 Parâmetros do Veículo para Simulação")

def user_input_features():
    year = st.sidebar.slider("Ano do Veículo", int(df['Year'].min()), int(df['Year'].max()), int(df['Year'].median()))
    present_price = st.sidebar.number_input("Preço de Tabela (Present Price)", float(df['Present_Price'].min()), float(df['Present_Price'].max()), float(df['Present_Price'].mean()))
    driven_kms = st.sidebar.number_input("Quilometragem Rodada (Kms_Driven)", int(df['Kms_Driven'].min()), int(df['Kms_Driven'].max()), int(df['Kms_Driven'].mean()))
    fuel_type = st.sidebar.selectbox("Tipo de Combustível", df['Fuel_Type'].unique())
    seller_type = st.sidebar.selectbox("Tipo de Vendedor (Seller_Type)", df['Seller_Type'].unique())
    transmission = st.sidebar.selectbox("Transmissão", df['Transmission'].unique())
    owner = st.sidebar.selectbox("Número de Proprietários Anteriores (Owner)", sorted(df['Owner'].unique()))
    
    data = {
        'Year': year,
        'Present_Price': present_price,
        'Kms_Driven': driven_kms,
        'Fuel_Type': fuel_type,
        'Seller_Type': seller_type,
        'Transmission': transmission,
        'Owner': owner
    }
    return pd.DataFrame(data, index=[0])

# Chamando a função para capturar os dados do usuário na sidebar
df_usuario = user_input_features()

# -------------------------------------------------------------
# 2. CORPO PRINCIPAL DO DASHBOARD (ABAS)
# -------------------------------------------------------------
aba1, aba2, aba3 = st.tabs(["📊 Visão Geral dos Dados", "📈 Análise Exploratória & Gráficos", "🔮 Simulador de Predição"])

with aba1:
    st.subheader("Amostra do Dataset Bruto (`Vehicle-DS.csv`)")
    st.dataframe(df.head(10))
    st.markdown(f"**Total de registros na base:** {df.shape[0]} veículos.")

with aba2:
    st.subheader("Análise Visual e Distribuição de Preços")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Gráfico de dispersão: Preço de Tabela vs Preço de Venda
        fig_scatter = px.scatter(
            df, x='Present_Price', y='Selling_Price', color='Fuel_Type',
            title="Preço de Tabela vs. Preço de Venda Real",
            labels={'Present_Price': 'Preço de Tabela', 'Selling_Price': 'Preço de Venda'}
        )
        st.plotly_chart(fig_scatter, use_container_width=True)
        
    with col2:
        # Histograma da variável alvo (Selling_Price)
        fig_hist = px.histogram(
            df, x='Selling_Price', nbins=30,
            title="Distribuição da Variável Alvo (Preço de Venda)",
            labels={'Selling_Price': 'Preço de Venda'}
        )
        st.plotly_chart(fig_hist, use_container_width=True)

with aba3:
    st.subheader("⚙️ Dados Selecionados pelo Usuário")
    st.dataframe(df_usuario)
    
    if st.button("Estimar Preço de Mercado"):
        st.success("✅ Parâmetros validados com sucesso! Na próxima etapa, conectaremos o modelo treinado (`.pkl`) para calcular o valor predito.")
        