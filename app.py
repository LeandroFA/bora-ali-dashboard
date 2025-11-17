
import streamlit as st
import pandas as pd
import plotly.express as px

st.set_page_config(layout="wide", page_title="Bora Ali Dashboard")

@st.cache_data
def load_data():
    df = pd.read_csv("INMET_ANAC_ROTAS_APENAS_CAPITAIS.csv", low_memory=False)
    ipca = pd.read_csv("IPCAUNIFICADO.csv")
    return df, ipca

df, ipca = load_data()

st.title("📊 Dashboard Bora Ali — Capitais do Brasil")

st.sidebar.header("Filtros")

anos = sorted(df["ANO"].dropna().unique())
selected_years = st.sidebar.multiselect("Ano", anos, default=anos)

capitais = sorted(df["DESTINO"].dropna().unique())
selected_cities = st.sidebar.multiselect("Capital (Destino)", capitais, default=capitais[:5])

f = df.copy()
f = f[f["ANO"].isin(selected_years)]
f = f[f["DESTINO"].isin(selected_cities)]

st.subheader("📈 Tarifa média por Mês")

if "TARIFA" in f.columns:
    tarifa_mensal = f.groupby(["ANO", "MES"])["TARIFA"].mean().reset_index()
    tarifa_mensal["DATA"] = pd.to_datetime(tarifa_mensal["ANO"].astype(str) + "-" + tarifa_mensal["MES"].astype(str) + "-01")

    fig = px.line(tarifa_mensal, x="DATA", y="TARIFA", title="Evolução da Tarifa Média")
    st.plotly_chart(fig, use_container_width=True)

else:
    st.warning("Coluna TARIFA não encontrada no CSV.")

st.subheader("🌡️ Relação Temperatura x Tarifa")

if "TARIFA" in f.columns and "TEMP_MEDIA" in f.columns:
    fig2 = px.scatter(f, x="TEMP_MEDIA", y="TARIFA", color="DESTINO",
                      title="Temperatura Média x Tarifa", trendline="ols")
    st.plotly_chart(fig2, use_container_width=True)
else:
    st.warning("Colunas TEMP_MEDIA ou TARIFA ausentes.")

st.subheader("📊 IPCA Mensal x Tarifa Média")

if "TARIFA" in df.columns:
    tar = df.groupby(["ANO", "MES"])["TARIFA"].mean().reset_index()
    merged = pd.merge(tar, ipca, on=["ANO", "MES"], how="inner")
    merged["DATA"] = merged["DATA"]

    fig3 = px.line(merged, x="DATA", y=["TARIFA", "IPCA_MENSAL"],
                   title="Tarifa Média vs IPCA")
    st.plotly_chart(fig3, use_container_width=True)

else:
    st.warning("Não foi possível calcular Tarifa x IPCA")
