# ================================
# ✈️ Bora Alí — Dashboard Jovem, Profissional e 100% PT-BR
# ================================

import os
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import pydeck as pdk
from prophet import Prophet
from prophet.plot import plot_plotly

# ================================
# CONFIGURAÇÃO GERAL E ESTILO
# ================================
CSV_FILE = "INMET_ANAC_ROTAS_APENAS_CAPITAIS.csv"
OUTPUT_PATH = "outputs"
os.makedirs(OUTPUT_PATH, exist_ok=True)

# Paleta Jovem e Profissional
PRIMARY = "#006DCE"      # Azul Bora Alí
ACCENT = "#FF6B4A"       # Coral Bora Alí
GOLD = "#FDBA74"
PURPLE = "#6366F1"
GREEN = "#16A34A"
RED = "#E11D48"
BG = "#F7F9FB"
TEXT = "#0F172A"

st.set_page_config(page_title="Bora Alí — Capitais", layout="wide")

# CSS Clean Jovem
st.markdown(f"""
<style>
body {{background-color:{BG};}}
h1, h2, h3, h4 {{color:{PRIMARY}; font-weight:800;}}
.stButton>button {{background-color:{PRIMARY}; color:white; border-radius:10px; font-weight:700;}}
</style>
""", unsafe_allow_html=True)


# ================================
# CARREGAR DADOS
# ================================
@st.cache_data
def load_data():
    try:
        df = pd.read_csv(CSV_FILE, low_memory=False)
    except FileNotFoundError:
        st.error(f"❌ O arquivo '{CSV_FILE}' precisa estar na raiz do repositório.")
        st.stop()

    df["ROTA"] = df["ROTA"].astype(str).str.replace(" - ", " → ")
    df["DATA"] = pd.to_datetime(df["ANO"].astype(str) + "-" + df["MES"].astype(str) + "-01")
    return df

df = load_data()


# ================================
# FILTROS
# ================================
st.sidebar.header("🎛️ Filtros — Escolha sua viagem!")
anos = st.sidebar.multiselect("📅 Ano", sorted(df["ANO"].unique()), default=sorted(df["ANO"].unique()))
meses = st.sidebar.multiselect("🗓️ Mês", list(range(1, 13)), default=list(range(1, 13)))
companias = st.sidebar.multiselect("🛫 Companhias aéreas", sorted(df["COMPANHIA"].unique()), default=sorted(df["COMPANHIA"].unique()))

df = df[df["ANO"].isin(anos) & df["MES"].isin(meses) & df["COMPANHIA"].isin(companias)]


# ================================
# LISTA DE CAPITAIS (COORDENADAS)
# ================================
CAPITAIS = {
    'Rio Branco': (-9.97499, -67.8243),'Maceió': (-9.649847, -35.70895),'Macapá': (0.034934, -51.0694),
    'Manaus': (-3.119028, -60.021731),'Salvador': (-12.97139, -38.50139),'Fortaleza': (-3.71722, -38.543366),
    'Brasília': (-15.793889, -47.882778),'Vitória': (-20.3155, -40.3128),'Goiânia': (-16.686891, -49.264788),
    'São Luís': (-2.52972, -44.30278),'Cuiabá': (-15.601415, -56.097892),'Campo Grande': (-20.4433, -54.6465),
    'Belo Horizonte': (-19.916681, -43.934493),'Belém': (-1.455833, -48.504444),'João Pessoa': (-7.119495, -34.845011),
    'Curitiba': (-25.429596, -49.271272),'Recife': (-8.047562, -34.8770),'Teresina': (-5.08921, -42.8016),
    'Rio de Janeiro': (-22.906847, -43.172896),'Natal': (-5.795, -35.209),'Porto Alegre': (-30.034647, -51.217658),
    'Porto Velho': (-8.7608, -63.9039),'Boa Vista': (2.8196, -60.6733),'Florianópolis': (-27.595377, -48.548046),
    'Aracaju': (-10.9472, -37.0731),'São Paulo': (-23.55052, -46.633308),'Palmas': (-10.184, -48.333)
}

caps = st.sidebar.multiselect("🏙️ Capitais", list(CAPITAIS.keys()), default=["São Paulo","Rio de Janeiro","Recife","Brasília","Manaus"])


# ================================
# SEPARAR ROTA E FILTRAR
# ================================
def rota(r):
    if "→" in r:
        p = r.split("→")
        return p[0].strip(), p[-1].strip()
    return None, None

df[["ORIG","DEST"]] = df["ROTA"].apply(lambda x: pd.Series(rota(x)))
df = df[df["ORIG"].isin(caps) & df["DEST"].isin(caps)]


# ================================
# 🌍 MAPA — TARIFAS X CLIMA
# ================================
st.header("📍 Mapa Bora Alí — Onde Clima e Tarifas se Encontram ✈️🌤️")

agg = df.groupby("DEST").agg(tarifa=("TARIFA","mean"), temp=("TEMP_MEDIA","mean")).reset_index()
agg["lat"] = agg["DEST"].map(lambda x: CAPITAIS.get(x,(np.nan,np.nan))[0])
agg["lon"] = agg["DEST"].map(lambda x: CAPITAIS.get(x,(np.nan,np.nan))[1])

m1 = px.scatter_mapbox(
    agg.dropna(), lat="lat", lon="lon", size="tarifa", color="temp",
    size_max=45, zoom=3, color_continuous_scale="thermal", hover_name="DEST"
)
m1.update_layout(mapbox_style="carto-positron", margin=0)
st.plotly_chart(m1, use_container_width=True)


# ================================
# 🛰️ MAPA — ROTAS
# ================================
st.header("🛫 Bora de rota entre capitais?")

rotas = df.groupby("ROTA").agg(tarifa=("TARIFA","mean")).reset_index()
rotas[["ORIG","DEST"]] = rotas["ROTA"].apply(lambda x: pd.Series(rota(x)))

for c in ["ORIG","DEST"]:
    rotas[f"{c.lower()}lat"] = rotas[c].map(lambda x: CAPITAIS.get(x,(np.nan,np.nan))[0])
    rotas[f"{c.lower()}lon"] = rotas[c].map(lambda x: CAPITAIS.get(x,(np.nan,np.nan))[1])

st.pydeck_chart(pdk.Deck(
    layers=[pdk.Layer("ArcLayer", data=rotas.dropna(),
                      get_source_position=["origlon","origlat"], get_target_position=["destlon","destlat"],
                      get_width="tarifa", get_source_color=[6,110,204], get_target_color=[255,107,74])],
    initial_view_state=pdk.ViewState(latitude=-14, longitude=-51, zoom=3.2),
    map_style="mapbox://styles/mapbox/light-v9"
))

# ================================
# 📊 TARIFA POR ESTAÇÕES (COM LABEL)
# ================================
st.header("🌦️ Quanto custa voar nas estações do ano?")

def estacao(mes):
    return ("Verão" if mes in [12,1,2] else
            "Outono" if mes in [3,4,5] else
            "Inverno" if mes in [6,7,8] else "Primavera")

df["ESTACAO"] = df["MES"].apply(estacao)
est = df.groupby("ESTACAO").agg(tarifa=("TARIFA","mean")).reset_index()
est["tarifa"] = est["tarifa"].round(0)

fig_est = px.bar(
    est, x="ESTACAO", y="tarifa", color="ESTACAO",
    text="tarifa",
    color_discrete_sequence=[PRIMARY, ACCENT, GOLD, PURPLE],
    title="💸 Tarifa média por estação (R$)"
)
fig_est.update_traces(texttemplate="R$ %{text:.0f}", textposition="outside")
fig_est.update_layout(yaxis_title="Preço médio (R$)")
st.plotly_chart(fig_est, use_container_width=True)


# ================================
# 🌎 TARIFA POR REGIÃO (COM LABEL)
# ================================
st.header("🧭 Qual região do Brasil é mais cara para voar?")

REGIOES = {
    "Norte": ["Belém","Macapá","Manaus","Boa Vista","Rio Branco","Porto Velho","Palmas"],
    "Nordeste": ["São Luís","Teresina","Fortaleza","Natal","João Pessoa","Recife","Maceió","Aracaju","Salvador"],
    "Centro-Oeste": ["Brasília","Goiânia","Campo Grande","Cuiabá"],
    "Sudeste": ["São Paulo","Rio de Janeiro","Belo Horizonte","Vitória"],
    "Sul": ["Curitiba","Florianópolis","Porto Alegre"]
}

def regiao(cidade):
    for reg, cids in REGIOES.items():
        if cidade in cids:
            return reg
    return None

df["REGIAO"] = df["DEST"].apply(regiao)
reg = df.groupby("REGIAO").agg(tarifa=("TARIFA","mean")).reset_index()
reg["tarifa"] = reg["tarifa"].round(0)

fig_reg = px.bar(
    reg, x="REGIAO", y="tarifa", color="REGIAO",
    text="tarifa",
    color_discrete_sequence=[PRIMARY, ACCENT, GREEN, PURPLE, RED],
    title="📍 Tarifa média por região do Brasil"
)
fig_reg.update_traces(texttemplate="R$ %{text:.0f}", textposition="outside")
fig_reg.update_layout(yaxis_title="Preço médio (R$)")
st.plotly_chart(fig_reg, use_container_width=True)


# ================================
# 📈 PREVISÃO — 2026
# ================================
st.header("📈 Bora prever a tarifa da sua rota em 2026? 👀✨")

sel = st.selectbox("Selecione uma rota para prever:", sorted(df["ROTA"].unique()))
dfm = df[df["ROTA"] == sel].groupby("DATA").agg(tarifa=("TARIFA","mean"), temp=("TEMP_MEDIA","mean")).reset_index()

if dfm.shape[0] > 12:
    dfm2 = dfm.rename(columns={"DATA":"ds","tarifa":"y"})
    m = Prophet(yearly_seasonality=True)
    m.fit(dfm2)
    fut = m.make_future_dataframe(periods=12, freq="MS")
    pred = m.predict(fut)
    st.plotly_chart(plot_plotly(m, pred), use_container_width=True)
else:
    st.warning("⚠️ Dados insuficientes para prever esta rota.")


st.markdown("💙 Feito com carinho aéreo pelo **Bora Alí** ✈️🧳")
