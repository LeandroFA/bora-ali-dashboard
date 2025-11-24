# =============================================================================
#  Bora Alí — Dashboard (Laranja Sunset) — COMPLETO FINAL
#  Funções: 7 gráficos, mapa corrigido, previsão 2026, filtros, ranking interativo (DESTINOS)
#  NÃO precisa criar pastas, NÃO precisa modificar CSV. Basta ter:
#  ✔ INMET_ANAC_ROTAS_APENAS_CAPITAIS.csv NA RAIZ DO REPOSITÓRIO.
# =============================================================================

# ---------------------------
# IMPORTS
# ---------------------------
import os
import unicodedata
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from prophet import Prophet
from prophet.plot import plot_plotly

# ---------------------------
# CONFIG / PALETA — Laranja Sunset
# ---------------------------
st.set_page_config(page_title="Bora Alí — Capitais (Laranja Sunset)", layout="wide", page_icon="🧳")

CSV_FILE = "INMET_ANAC_ROTAS_APENAS_CAPITAIS.csv"   # <-- NÃO MUDE ISSO
OUTPUT_DIR = "outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Paleta
ORANGE = "#F76715"
PURPLE = "#7E3FF2"
SOFT = "#FFBF69"
BG = "#FCFBFA"
TEXT = "#0F172A"

# Estilo Global
st.markdown(f"""
<style>
body {{background:{BG};}}
h1,h2,h3 {{color:{PURPLE}; font-weight:800;}}
.stButton>button {{background:{ORANGE}; color:white; border-radius:10px; font-weight:700;}}
</style>
""", unsafe_allow_html=True)

st.title("✈️ Bora Alí — Capitais (Laranja Sunset)")
st.caption("Urbano, Jovem, Preciso — 7 visuais + previsões 2026 + Ranking interativo")

# ---------------------------
# Normalização de textos e cidades
# ---------------------------
CANONICAL = [
    "Rio Branco","Maceió","Macapá","Manaus","Salvador","Fortaleza","Brasília","Vitória","Goiânia",
    "São Luís","Cuiabá","Campo Grande","Belo Horizonte","Belém","João Pessoa","Curitiba","Recife",
    "Teresina","Rio de Janeiro","Natal","Porto Alegre","Porto Velho","Boa Vista","Florianópolis",
    "Aracaju","São Paulo","Palmas"
]

def normalize_str(s):
    if pd.isna(s): return s
    s = str(s)
    s = "".join(ch for ch in unicodedata.normalize("NFKD", s) if not unicodedata.combining(ch))
    s = s.replace("_"," ").replace("-"," ")
    s = " ".join(s.split()).strip().lower()
    return s

NORM_TO_CANON = {normalize_str(c): c for c in CANONICAL}

def map_to_canonical(city):
    if pd.isna(city): return city
    n = normalize_str(city)
    if n in NORM_TO_CANON: return NORM_TO_CANON[n]
    for key,val in NORM_TO_CANON.items():
        if any(tok in key.split() for tok in n.split()):
            return val
    return city.strip().title()

def safe_parse_route(r):
    if pd.isna(r): return (None,None)
    s = str(r)
    for sep in ["→","-"]:
        if sep in s:
            p=[x.strip() for x in s.split(sep) if x.strip()]
            if len(p)>=2: return (p[0],p[-1])
    return (None,None)

# ---------------------------
# LOAD DATA
# ---------------------------
@st.cache_data(ttl=600)
def load_csv(path):
    try:
        df = pd.read_csv(path, low_memory=False)
    except:
        st.error(f"⚠ Arquivo NÃO encontrado: {path}. Coloque ele na RAIZ do GitHub e redeploy.")
        st.stop()

    df.columns = [c.strip().upper() for c in df.columns]

    # Garante colunas
    for c in ["ROTA","COMPANHIA","ORIGEM","DESTINO","TARIFA","TEMP_MEDIA","ANO","MES","TEMP_MIN","TEMP_MAX"]:
        if c not in df.columns: df[c] = pd.NA

    # Se não tiver TEMP_MEDIA => cria média
    if df["TEMP_MEDIA"].isna().all():
        df["TEMP_MEDIA"] = (pd.to_numeric(df["TEMP_MIN"],errors="coerce") + pd.to_numeric(df["TEMP_MAX"],errors="coerce"))/2

    # Map rota/origem/destino
    df["ORIG"] = df["ORIGEM"].where(df["ORIGEM"].notna(), None)
    df["DEST"] = df["DESTINO"].where(df["DESTINO"].notna(), None)

    parsed = df.apply(lambda r: pd.Series(safe_parse_route(r["ROTA"]),index=["_o","_d"]),axis=1)
    df["ORIG"] = df["ORIG"].fillna(parsed["_o"])
    df["DEST"] = df["DEST"].fillna(parsed["_d"])

    df["ORIG"] = df["ORIG"].astype(str).apply(map_to_canonical)
    df["DEST"] = df["DEST"].astype(str).apply(map_to_canonical)

    df["TARIFA"] = pd.to_numeric(df["TARIFA"], errors="coerce")
    df["TEMP_MEDIA"]=pd.to_numeric(df["TEMP_MEDIA"],errors="coerce")
    df["ANO"]=pd.to_numeric(df["ANO"],errors="coerce").fillna(0).astype(int)
    df["MES"]=pd.to_numeric(df["MES"],errors="coerce").fillna(0).astype(int)

    df["DATA"]=pd.to_datetime(df["ANO"].astype(str)+"-"+df["MES"].astype(str).str.zfill(2)+"-01",errors="coerce")

    df["ROTA"]=df["ORIG"].fillna("")+" → "+df["DEST"].fillna("")
    return df

df = load_csv(CSV_FILE)

# ---------------------------
# SIDEBAR — FILTROS
# ---------------------------
st.sidebar.header("🎯 Filtros Inteligentes")

def season(m):
    if m in [12,1,2]: return "Verão"
    if m in [3,4,5]: return "Outono"
    if m in [6,7,8]: return "Inverno"
    return "Primavera"

df["ESTACAO"] = df["MES"].apply(lambda x: season(int(x)) if not pd.isna(x) else np.nan)

anos = sorted(df["ANO"].dropna().unique())
sel_ano = st.sidebar.multiselect("Ano", anos, default=anos)

meses = sorted(df["MES"].dropna().unique())
sel_mes = st.sidebar.multiselect("Mês", meses, default=meses)

companias = sorted(df["COMPANHIA"].dropna().unique())
sel_comp = st.sidebar.multiselect("Companhia", companias, default=companias)

estacoes=["Verão","Outono","Inverno","Primavera"]
sel_est = st.sidebar.multiselect("Estação", estacoes, default=estacoes)

caps = sorted(set(df["ORIG"].dropna().unique()) | set(df["DEST"].dropna().unique()))
sel_cap = st.sidebar.multiselect("Capitais", caps, default=caps)

dff = df[
    (df["ANO"].isin(sel_ano)) &
    (df["MES"].isin(sel_mes)) &
    (df["COMPANHIA"].isin(sel_comp)) &
    (df["ESTACAO"].isin(sel_est)) &
    (df["ORIG"].isin(sel_cap)) &
    (df["DEST"].isin(sel_cap))
].copy()

if dff.empty:
    st.warning("⛔ Nenhum registro com esses filtros!")
    st.stop()

# ---------------------------
# COORDENADAS DAS CAPITAIS
# ---------------------------
COORDS={
'Rio Branco':(-9.97499,-67.8243),'Maceió':(-9.649847,-35.70895),'Macapá':(0.034934,-51.0694),
'Manaus':(-3.119028,-60.021731),'Salvador':(-12.97139,-38.50139),'Fortaleza':(-3.71722,-38.543366),
'Brasília':(-15.793889,-47.882778),'Vitória':(-20.3155,-40.3128),'Goiânia':(-16.686891,-49.264788),
'São Luís':(-2.52972,-44.30278),'Cuiabá':(-15.601415,-56.097892),'Campo Grande':(-20.4433,-54.6465),
'Belo Horizonte':(-19.916681,-43.934493),'Belém':(-1.455833,-48.504444),'João Pessoa':(-7.119495,-34.845011),
'Curitiba':(-25.429596,-49.271272),'Recife':(-8.047562,-34.8770),'Teresina':(-5.08921,-42.8016),
'Rio de Janeiro':(-22.906847,-43.172896),'Natal':(-5.795,-35.209),'Porto Alegre':(-30.034647,-51.217658),
'Porto Velho':(-8.7608,-63.9039),'Boa Vista':(2.8196,-60.6733),'Florianópolis':(-27.595377,-48.548046),
'Aracaju':(-10.9472,-37.0731),'São Paulo':(-23.55052,-46.633308),'Palmas':(-10.184,-48.333)
}

# ------------------------------------------------------------------------------
# KPIs
# ------------------------------------------------------------------------------
st.markdown("---")
c1,c2,c3,c4 = st.columns(4)
c1.metric("📊 Registros", f"{len(dff):,}")
c2.metric("💰 Tarifa média", f"R$ {dff['TARIFA'].mean():.0f}")
c3.metric("🌡 Temp média", f"{dff['TEMP_MEDIA'].mean():.1f} °C")
c4.metric("✈️ Rotas únicas", f"{dff['ROTA'].nunique():,}")

# ------------------------------------------------------------------------------
# 1) MAPA CAPITAIS
# ------------------------------------------------------------------------------
st.markdown("---")
st.subheader("1) Mapa — Capitais (tamanho = tarifa · cor = temperatura)")

agg_cap = dff.groupby("DEST").agg(
    tarifa_media=("TARIFA","mean"),
    temp_media=("TEMP_MEDIA","mean"),
    regs=("TARIFA","count")
).reset_index()
agg_cap["lat"]=agg_cap["DEST"].map(lambda x:COORDS.get(x,(np.nan,np.nan))[0])
agg_cap["lon"]=agg_cap["DEST"].map(lambda x:COORDS.get(x,(np.nan,np.nan))[1])

fig1 = px.scatter_mapbox(
    agg_cap.dropna(subset=["lat","lon"]),
    lat="lat", lon="lon", size="tarifa_media", color="temp_media",
    hover_name="DEST",
    hover_data={"tarifa_media":":.0f","temp_media":":.1f","regs":True,"lat":False,"lon":False},
    size_max=45, zoom=3.2,
    color_continuous_scale=[SOFT,ORANGE,PURPLE], height=480
)
fig1.update_layout(mapbox_style="carto-positron", margin=0)
st.plotly_chart(fig1, use_container_width=True)

# ------------------------------------------------------------------------------
# 2) MAPA ROTAS (CORRIGIDO)
# ------------------------------------------------------------------------------
st.markdown("---")
st.subheader("2) Rotas Premium (espessura = tarifa média)")

routes = dff.groupby("ROTA").agg(tarifa_media=("TARIFA","mean"), regs=("TARIFA","count")).reset_index()
routes[["ORIG","DEST"]] = routes["ROTA"].apply(lambda r: pd.Series(safe_parse_route(r)))

def coord(x):
    if pd.isna(x): return (np.nan,np.nan)
    x = map_to_canonical(x)
    return COORDS.get(x,(np.nan,np.nan))

routes["olat"]=routes["ORIG"].apply(lambda x: coord(x)[0])
routes["olon"]=routes["ORIG"].apply(lambda x: coord(x)[1])
routes["dlat"]=routes["DEST"].apply(lambda x: coord(x)[0])
routes["dlon"]=routes["DEST"].apply(lambda x: coord(x)[1])
routes = routes.dropna(subset=["olat","olon","dlat","dlon","tarifa_media"])
routes = routes[routes["tarifa_media"]>0].copy()

if not routes.empty:
    q = routes["tarifa_media"].quantile([0,0.25,0.5,0.75,1.0]).tolist()
    def width(x):
        x=float(x)
        return 1.4 if x<=q[1] else 2.8 if x<=q[2] else 4.5 if x<=q[3] else 7.0
    routes["w"]=routes["tarifa_media"].apply(width)

    fig2=go.Figure()
    for _,r in routes.iterrows():
        fig2.add_trace(go.Scattermapbox(
            lat=[r["olat"],r["dlat"]], lon=[r["olon"],r["dlon"]],
            mode="lines", line=dict(width=float(r["w"]), color=ORANGE, opacity=0.85),
            hoverinfo="text",
            text=f"<b>{r['ROTA']}</b><br>💰 R$ {r['tarifa_media']:.0f}<br>📌 {int(r['regs'])} registros"
        ))
    fig2.add_trace(go.Scattermapbox(
        lat=agg_cap["lat"], lon=agg_cap["lon"], mode="markers+text",
        marker=dict(size=9,color=PURPLE),
        text=agg_cap["DEST"], textposition="top right"
    ))
    fig2.update_layout(mapbox_style="carto-positron", mapbox_center={"lat":-14.2,"lon":-51.9}, mapbox_zoom=3.2, margin=0, height=520)
    st.plotly_chart(fig2, use_container_width=True)
else:
    st.info("Sem rotas com os filtros selecionados.")

# ------------------------------------------------------------------------------
# 3) Ranking Interativo — DESTINOS
# ------------------------------------------------------------------------------
st.markdown("---")
st.subheader("🎖 3) Ranking Interativo — Destinos")

topd = dff.groupby("DEST").agg(
    tarifa_media=("TARIFA","mean"),
    regs=("TARIFA","count")
).reset_index().sort_values("tarifa_media", ascending=False)

fig_rank = px.bar(
    topd, x="DEST", y="tarifa_media", text=topd["tarifa_media"].round(0),
    color="tarifa_media", color_continuous_scale=[SOFT,ORANGE,PURPLE]
)
fig_rank.update_traces(texttemplate="R$ %{text:.0f}", textposition="outside")
fig_rank.update_layout(yaxis_title="Tarifa média (R$)", xaxis_title="Destino")
sel_click = st.plotly_chart(fig_rank, use_container_width=True)

st.write("Clique no destino no gráfico acima para destacar no mapa!")

# ------------------------------------------------------------------------------
# 4) Séries, Estações, Regiões
# ------------------------------------------------------------------------------

st.markdown("---")
st.subheader("4) Série Temporal — Tarifa média mensal")
ts = dff.groupby("DATA").agg(media=("TARIFA","mean")).reset_index()
st.plotly_chart(px.line(ts,x="DATA",y="media",markers=True).update_layout(yaxis_title="R$"), use_container_width=True)

st.markdown("---")
st.subheader("5) Estações do ano — Tarifa média")
est = dff.groupby("ESTACAO").agg(media=("TARIFA","mean")).reset_index()
st.plotly_chart(px.bar(est,x="ESTACAO",y="media",color="ESTACAO", text=est["media"].round(0)).update_traces(textposition="outside"), use_container_width=True)

st.markdown("---")
st.subheader("6) Regiões — Tarifa média")
REG={
"Norte":["Belém","Macapá","Manaus","Boa Vista","Rio Branco","Porto Velho","Palmas"],
"Nordeste":["São Luís","Teresina","Fortaleza","Natal","João Pessoa","Recife","Maceió","Aracaju","Salvador"],
"Centro-Oeste":["Brasília","Goiânia","Campo Grande","Cuiabá"],
"Sudeste":["São Paulo","Rio de Janeiro","Belo Horizonte","Vitória"],
"Sul":["Curitiba","Florianópolis","Porto Alegre"]
}
def reg(x):
    for k,v in REG.items():
        if x in v:return k
    return "Outro"
dff["REGIAO"]=dff["DEST"].apply(reg)
regm = dff.groupby("REGIAO").agg(media=("TARIFA","mean")).reset_index()
st.plotly_chart(px.bar(regm,x="REGIAO",y="media",text=regm["media"].round(0)), use_container_width=True)

# ------------------------------------------------------------------------------
# 7) Heatmap
# ------------------------------------------------------------------------------
st.markdown("---")
st.subheader("7) Heatmap — tarifa (mês x capital)")
hm = dff.groupby([dff["DATA"].dt.month.rename("MES"),"DEST"]).agg(m=("TARIFA","mean")).reset_index()
pv = hm.pivot(index="DEST", columns="MES", values="m").fillna(0)
st.plotly_chart(px.imshow(pv, labels=dict(x="Mês",y="Destino",color="Tarifa (R$)")), use_container_width=True)

# ------------------------------------------------------------------------------
# Previsão 2026
# ------------------------------------------------------------------------------
st.markdown("---")
st.header("🔮 Previsão 2026 — Tarifas por rota")
rota_escolha = st.selectbox("Escolha a rota:", sorted(dff["ROTA"].unique()))
df_p = dff[dff["ROTA"]==rota_escolha].groupby("DATA").agg(tar=("TARIFA","mean"),tmp=("TEMP_MEDIA","mean")).reset_index()

if df_p.shape[0]>=12:
    dfm = df_p.rename(columns={"DATA":"ds","tar":"y","tmp":"temp"})
    modelo=Prophet(yearly_seasonality=True)
    if dfm["temp"].notna().sum()>0:
        modelo.add_regressor("temp")
    modelo.fit(dfm)
    fut=modelo.make_future_dataframe(periods=12,freq="MS")
    fut["temp"]=dfm.groupby(dfm["ds"].dt.month)["temp"].transform("mean").iloc[:12].tolist()*2
    fc=modelo.predict(fut)
    st.plotly_chart(plot_plotly(modelo,fc),use_container_width=True)
else:
    st.info("📌 Essa rota não possui histórico suficiente (mínimo 12 meses).")

st.caption("🌇 Bora Alí © — Laranja Sunset | SR2")

