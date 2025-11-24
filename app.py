# ============================
#   BORA ALÍ — DASHBOARD SR2
#   Travel Insights + Previsões 2026
# ============================

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import unicodedata
from prophet import Prophet

# ============================
# CONFIGURAÇÃO DO APP
# ============================
st.set_page_config(
    page_title="Bora Alí — Capitais",
    layout="wide",
    page_icon="✈️"
)

# ============================
# FUNÇÃO: NORMALIZAR NOMES
# ============================
def normalize_city_name(city):
    if not isinstance(city, str):
        return city
    c = ''.join(ch for ch in unicodedata.normalize('NFKD', city)
                if not unicodedata.combining(ch))
    c = c.replace("_", " ").replace("-", " ")
    c = " ".join(c.split())
    c = c.strip().title()
    return c

# ============================
# CARREGAR DADOS NA RAIZ
# ============================
CSV = "INMET_ANAC_ROTAS_APENAS_CAPITAIS.csv"

@st.cache_data
def load():
    df = pd.read_csv(CSV, sep=",")
    df.columns = df.columns.str.upper().str.strip()
    if "TEMP_MEDIA" not in df.columns:
        df["TEMP_MEDIA"] = df[["TEMP_MIN", "TEMP_MAX"]].mean(axis=1)
    df["ORIGEM"] = df["ORIGEM"].apply(normalize_city_name)
    df["DESTINO"] = df["DESTINO"].apply(normalize_city_name)
    df["ROTA"] = df["ORIGEM"] + " → " + df["DESTINO"]
    return df

try:
    df = load()
except:
    st.error("❌ O arquivo **INMET_ANAC_ROTAS_APENAS_CAPITAIS.csv** deve estar na RAIZ do repositório.")
    st.stop()

# ============================
# FUNÇÃO: DEFINIR ESTAÇÃO
# ============================
def estacao(mes):
    return {
        12:"Verão", 1:"Verão", 2:"Verão",
        3:"Outono", 4:"Outono", 5:"Outono",
        6:"Inverno", 7:"Inverno", 8:"Inverno",
        9:"Primavera", 10:"Primavera", 11:"Primavera"
    }[mes]

df["ESTACAO"] = df["MES"].apply(estacao)

# ============================
# FILTROS LATERAIS
# ============================
st.sidebar.header("🎯 Filtros Rápidos")

anos = st.sidebar.multiselect("Ano", sorted(df["ANO"].unique()), default=df["ANO"].unique())
companhia = st.sidebar.multiselect("Companhia", sorted(df["COMPANHIA"].unique()), default=df["COMPANHIA"].unique())
meses = st.sidebar.multiselect("Mês", sorted(df["MES"].unique()), default=df["MES"].unique())

df_filtrado = df[(df["ANO"].isin(anos)) &
                 (df["COMPANHIA"].isin(companhia)) &
                 (df["MES"].isin(meses))]

st.title("✈️ Bora Alí — Capitais do Brasil")

# ============================
# KPIs DESTAQUE
# ============================
col1, col2, col3 = st.columns(3)

col1.metric("📌 Registros", f"{df_filtrado.shape[0]:,}".replace(",", "."))
col2.metric("💰 Tarifa Média (R$)", f"{df_filtrado['TARIFA'].mean():.0f}")
col3.metric("🌡️ Temperatura Média (°C)", f"{df_filtrado['TEMP_MEDIA'].mean():.1f}")

# ============================
# 1 — TARIFA POR ESTAÇÃO DO ANO
# ============================
st.subheader("🌤️ Tarifa Média por Estação — Jovem, simples e direta")
fig_est = px.bar(df_filtrado.groupby("ESTACAO")["TARIFA"].mean().round(),
                 title="Tarifa Média por Estação (R$)",
                 labels={"value":"Tarifa Média (R$)", "ESTACAO":"Estação"},
                 color=["Verão","Outono","Inverno","Primavera"],
                 text_auto=True)
st.plotly_chart(fig_est, use_container_width=True)

# ============================
# 2 — TARIFA POR REGIÃO DO BRASIL
# ============================
regioes = {
"Sudeste":["São Paulo","Rio De Janeiro","Belo Horizonte","Vitória"],
"Sul":["Curitiba","Florianópolis","Porto Alegre"],
"Nordeste":["Recife","Fortaleza","Maceió","Natal","João Pessoa","Teresina","São Luís","Aracaju","Salvador"],
"Centro-Oeste":["Brasília","Cuiabá","Campo Grande","Goiânia"],
"Norte":["Manaus","Rio Branco","Macapá","Belém","Boa Vista","Porto Velho","Palmas"]
}

def classifica_regiao(cidade):
    for k,v in regioes.items():
        if cidade in v:
            return k
    return "Outra"

df_filtrado["REGIAO"] = df_filtrado["DESTINO"].apply(classifica_regiao)

st.subheader("🌎 Tarifa Média por Região do Brasil")
fig_reg = px.bar(df_filtrado.groupby("REGIAO")["TARIFA"].mean().round(),
                 color=df_filtrado.groupby("REGIAO")["TARIFA"].mean().round(),
                 text_auto=True, labels={"value":"Tarifa Média (R$)", "REGIAO":"Região"},
                 title="Tarifa Média por Região")
st.plotly_chart(fig_reg, use_container_width=True)

# ============================
# 3 — PREVISÃO 2026 (PROPHET)
# ============================
st.subheader("🔮 Predição de Tarifas para 2026 (Prophet)")

df_prophet = df_filtrado.groupby(["ANO","MES"])["TARIFA"].mean().reset_index()
df_prophet["DATA"] = pd.to_datetime(df_prophet["ANO"].astype(str) + "-" + df_prophet["MES"].astype(str) + "-01")
df_prophet = df_prophet[["DATA","TARIFA"]].rename(columns={"DATA":"ds","TARIFA":"y"})

m = Prophet()
m.fit(df_prophet)
future = m.make_future_dataframe(periods=12, freq="M")
forecast = m.predict(future)

fig_pred = px.line(forecast, x="ds", y="yhat", title="Previsão Tarifária — 2026", markers=True)
fig_pred.update_traces(line_color="#0052cc")
st.plotly_chart(fig_pred, use_container_width=True)

st.success("💙 Dashboard atualizado com Estações + Regiões + Previsão 2026!")

