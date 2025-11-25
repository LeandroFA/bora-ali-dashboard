# ================================================================
# Bora Alí — Dashboard SR2 (ARQUIVO: app.py)
# Dataset: INMET_ANAC_EXTREMAMENTE_REDUZIDO.csv
# NÃO USA IPCA • Foco sazonal + previsão tarifas 2026
# ================================================================

import streamlit as st
import pandas as pd
import numpy as np
import os
import plotly.express as px
import pydeck as pdk
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import GroupKFold, cross_val_score
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

# ----------- CONFIG VISUAL ----------
st.set_page_config(page_title="Bora Alí — SR2", layout="wide")

st.markdown("""
<style>
.neon-title { font-size:38px; font-weight:800; color:#C77DFF;
text-shadow:0 0 10px #B15EFF, 0 0 25px #9B4DFF;}
.card { padding:12px; border-radius:12px; background:rgba(255,255,255,0.06);
box-shadow:0 4px 15px rgba(0,0,0,0.2);}
</style>
""", unsafe_allow_html=True)

# ---------- FUNÇÕES ----------
def month_to_season(m):
    if m in [12,1,2]: return "VERÃO"
    if m in [3,4,5]: return "OUTONO"
    if m in [6,7,8]: return "INVERNO"
    return "PRIMAVERA"

@st.cache_data
def load_data():
    return pd.read_csv("INMET_ANAC_EXTREMAMENTE_REDUZIDO.csv", parse_dates=["DATA"])

# ---------- CARREGAMENTO ----------
df = load_data()
df["ANO"] = df["DATA"].dt.year
df["MES"] = df["DATA"].dt.month
df["SEASON"] = df["MES"].apply(month_to_season)

st.markdown("<div class='neon-title'>📌 Bora Alí — SR2 Dashboard</div>", unsafe_allow_html=True)
st.write("Filtros inteligentes, sazonalidade e previsão de tarifas 2026.")

# ---------- SIDEBAR ----------
st.sidebar.header("Filtros Inteligentes")

sel_year = st.sidebar.multiselect("Ano", sorted(df["ANO"].unique()), default=sorted(df["ANO"].unique()))
sel_month = st.sidebar.multiselect("Mês", list(range(1,13)), default=list(range(1,13)))
sel_comp = st.sidebar.multiselect("Companhia", sorted(df["COMPANHIA"].unique()), default=sorted(df["COMPANHIA"].unique()))
sel_cap = st.sidebar.multiselect("Destino (Capitais)", sorted(df["DESTINO"].unique()), default=sorted(df["DESTINO"].unique()))
sel_season = st.sidebar.multiselect("Estação", ["VERÃO","OUTONO","INVERNO","PRIMAVERA"], default=["VERÃO","OUTONO","INVERNO","PRIMAVERA"])

df_filtered = df[
    (df["ANO"].isin(sel_year)) &
    (df["MES"].isin(sel_month)) &
    (df["COMPANHIA"].isin(sel_comp)) &
    (df["DESTINO"].isin(sel_cap)) &
    (df["SEASON"].isin(sel_season))
]

# ---------------- KPIs ----------------
col1, col2, col3 = st.columns(3)
col1.markdown(f"<div class='card'>📌 **Registros:** {len(df_filtered):,}</div>", unsafe_allow_html=True)
col2.markdown(f"<div class='card'>💰 **Tarifa média:** R$ {df_filtered['TARIFA'].mean():.2f}</div>", unsafe_allow_html=True)
col3.markdown(f"<div class='card'>🌡️ **Temperatura média:** {df_filtered['TEMP_MEDIA'].mean():.1f}°C</div>", unsafe_allow_html=True)

# ------------- SAZONALIDADE ---------------
st.subheader("📊 Sazonalidade — Tarifa Média por Estação")
season_stats = df_filtered.groupby("SEASON")["TARIFA"].mean().reset_index()
fig_season = px.bar(season_stats, x="SEASON", y="TARIFA", color="SEASON",
                    color_discrete_sequence=["#C77DFF","#7BFFB7","#FFA24D","#28F8FF"])
st.plotly_chart(fig_season, use_container_width=True)

# ------------- MAPA -----------------------
st.subheader("🗺️ Mapa — Tarifas Médias por Capital")

if "DEST_LAT" in df.columns:
    map_df = df_filtered.groupby(["DESTINO","DEST_LAT","DEST_LON"])["TARIFA"].mean().reset_index()
    deck = pdk.Deck(
        initial_view_state=pdk.ViewState(latitude=-14, longitude=-51, zoom=3.7),
        layers=[
            pdk.Layer(
                "ScatterplotLayer",
                data=map_df,
                get_position='[DEST_LON, DEST_LAT]',
                get_fill_color='[255, 120, 220, 180]',
                get_radius=45000,
                pickable=True
            )
        ],
        tooltip={"text": "{DESTINO}\nTarifa Média: R${TARIFA}"}
    )
    st.pydeck_chart(deck)
else:
    st.warning("⚠ O dataset não contém coordenadas. Adicione DEST_LAT e DEST_LON para ativar o mapa.")

# ------------- PREVISÃO --------------------
st.subheader("📈 Previsão de Tarifas — 2026")

rota_sel = st.selectbox("Escolha uma rota:", sorted(df["ROTA"].unique()))

df_r = df[df["ROTA"]==rota_sel].copy()

if st.button("Gerar previsão para 2026"):
    X = df[["ANO","MES","ORIGEM","DESTINO","COMPANHIA","TEMP_MEDIA"]]
    y = df["TARIFA"]
    X["sin"] = np.sin(2*np.pi*X["MES"]/12)
    X["cos"] = np.cos(2*np.pi*X["MES"]/12)

    pre = ColumnTransformer([("cat", OneHotEncoder(handle_unknown="ignore"), ["ORIGEM","DESTINO","COMPANHIA"])], remainder="passthrough")
    model = Pipeline([("pre", pre), ("gbr", GradientBoostingRegressor())])
    model.fit(X, y)

    future = pd.DataFrame({"ANO":2026,"MES":np.arange(1,13)})
    future["ORIGEM"]=df_r["ORIGEM"].iloc[0]
    future["DESTINO"]=df_r["DESTINO"].iloc[0]
    future["COMPANHIA"]=df_r["COMPANHIA"].mode().iloc[0]
    future["TEMP_MEDIA"]=df_r.groupby("MES")["TEMP_MEDIA"].mean().reindex(range(1,13)).bfill().values
    future["sin"]=np.sin(2*np.pi*future["MES"]/12)
    future["cos"]=np.cos(2*np.pi*future["MES"]/12)
    pred = model.predict(future)

    future["PRED"] = pred
    st.dataframe(future[["MES","PRED"]].rename(columns={"PRED":"Tarifa Prevista (R$)"}))

    fig_pred = px.line(future, x="MES", y="PRED", markers=True, title=f"Previsão 2026 — {rota_sel}")
    st.plotly_chart(fig_pred, use_container_width=True)
