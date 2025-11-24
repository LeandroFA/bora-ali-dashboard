# ============================================================
# Bora Alí — SR2 • Dashboard Premium de Rotas Aéreas
# Foco: Rotas com Tendência de Alta (Evite em 2026)
# Visual: ROXO URBANO (Marca Bora Alí)
# ============================================================

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.linear_model import LinearRegression

# -------------------------------------------
# CONFIGURAÇÃO DE PÁGINA
# -------------------------------------------
st.set_page_config(
    page_title="Bora Alí — SR2",
    layout="wide",
    page_icon="✈️"
)

# -------------------------------------------
# PALETA SR2 – ROXO
# -------------------------------------------
ROXO = "#5A189A"
ROSA = "#E11D48"
LARANJA = "#FF6A00"
GRAFITE = "#1E1E1E"

st.markdown("""
<style>
h1,h2,h3,h4 { color: #5A189A !important; }
.stButton>button {
    background: #5A189A;
    color:white;
    border-radius:12px;
    padding:8px 16px;
    font-weight:700;
}
</style>
""", unsafe_allow_html=True)

# -------------------------------------------
# CARREGAR ARQUIVO
# -------------------------------------------
df = pd.read_csv("INMET_ANAC_ROTAS_APENAS_CAPITAIS.csv")

df["ANO"] = df["ANO"].astype(int)
df["MES"] = df["MES"].astype(int)
df["DATA"] = pd.to_datetime(df["ANO"].astype(str) + "-" + df["MES"].astype(str).str.zfill(2) + "-01")
df["ROTA"] = df["ORIG"] + " → " + df["DEST"]

# -------------------------------------------
# SIDEBAR — FILTROS
# -------------------------------------------
st.sidebar.header("🎯 Filtros Inteligentes — SR2")

origem = st.sidebar.selectbox("Origem", sorted(df["ORIG"].unique()))
destino = st.sidebar.selectbox("Destino", sorted(df["DEST"].unique()))

dff = df[(df["ORIG"] == origem) & (df["DEST"] == destino)]

if dff.empty:
    st.error("Nenhum dado encontrado para essa rota.")
    st.stop()

# -------------------------------------------
# TÍTULO
# -------------------------------------------
st.title("🟣 Bora Alí — SR2 Premium")
st.subheader(f"Rota Selecionada: **{origem} → {destino}**")

# ============================================================
# 1) PREVISÃO 2026 (REGRESSÃO SUAVIZADA)
# ============================================================

st.markdown("## 🔮 Previsão 2026 — Tarifa esperada por mês")

# preparar dados
g = dff.groupby("DATA")["TARIFA"].mean().reset_index()
g["t"] = np.arange(len(g))

# regressão
model = LinearRegression()
model.fit(g[["t"]], g["TARIFA"])

# prever 12 meses futuros
future_t = np.arange(len(g), len(g) + 12)
pred = model.predict(future_t.reshape(-1, 1))

future_dates = pd.date_range(start=g["DATA"].max() + pd.DateOffset(months=1), periods=12, freq="MS")

df_pred = pd.DataFrame({
    "DATA": future_dates,
    "PREVISAO": pred
})

fig = go.Figure()
fig.add_trace(go.Scatter(x=g["DATA"], y=g["TARIFA"], mode="lines+markers",
    name="Histórico", line=dict(color=ROXO)))
fig.add_trace(go.Scatter(x=df_pred["DATA"], y=df_pred["PREVISAO"], mode="lines+markers",
    name="Previsão 2026", line=dict(color=ROSA, dash="dash")))

fig.update_layout(
    title="Previsão Suavizada de Tarifas",
    yaxis_title="Tarifa Média (R$)"
)

st.plotly_chart(fig, use_container_width=True)

# ============================================================
# 2) PREÇOS POR ESTAÇÃO (COM IMAGENS)
# ============================================================

st.markdown("## 🌦 Tarifas por Estação — Visual Premium")

# imagens
cols = st.columns(4)
stations = ["Verão", "Outono", "Inverno", "Primavera"]
imgs = [
    "https://i.imgur.com/Yp7Gkcp.png",
    "https://i.imgur.com/z0o5TmU.png",
    "https://i.imgur.com/bxSdjMk.png",
    "https://i.imgur.com/1O2hm7x.png"
]

for c, s, img in zip(cols, stations, imgs):
    c.image(img, width=100)
    c.markdown(f"### {s}")

# dados
dff["ESTACAO"] = dff["MES"].map({
    12: "Verão", 1: "Verão", 2: "Verão",
    3: "Outono", 4: "Outono", 5: "Outono",
    6: "Inverno", 7: "Inverno", 8: "Inverno",
    9: "Primavera", 10: "Primavera", 11: "Primavera"
})

est = dff.groupby("ESTACAO")["TARIFA"].mean().reset_index()

fig_est = px.bar(est, x="ESTACAO", y="TARIFA",
                 color="ESTACAO",
                 color_discrete_sequence=[ROSA, LARANJA, ROXO, "#7C3AED"],
                 text=est["TARIFA"].round(0))
fig_est.update_traces(textposition="outside")
fig_est.update_layout(yaxis_title="Tarifa Média (R$)")

st.plotly_chart(fig_est, use_container_width=True)

# ============================================================
# 3) PREÇO MÉDIO POR MÊS (ROTA ESCOLHIDA)
# ============================================================

st.markdown("## 🗓 Preço Médio por Mês — Rota Selecionada")

m = dff.groupby("MES")["TARIFA"].mean().reset_index()

fig_mes = px.line(m, x="MES", y="TARIFA",
                  markers=True,
                  color_discrete_sequence=[ROXO])
fig_mes.update_layout(yaxis_title="Tarifa Média (R$)")

st.plotly_chart(fig_mes, use_container_width=True)

# ============================================================
# 4) RECOMENDAÇÃO AUTOMÁTICA (SR2)
# ============================================================

st.markdown("## 🧠 Recomendação Inteligente — SR2")

tc = df_pred["PREVISAO"].mean()
tmax = df_pred["PREVISAO"].max()

if tmax - tc > 150:
    st.error("🛑 **Evite essa rota em 2026** — tendência forte de alta.")
elif tmax - tc > 70:
    st.warning("⚠️ **Atenção** — leve tendência de aumento.")
else:
    st.success("🟢 Essa rota está estável.")

