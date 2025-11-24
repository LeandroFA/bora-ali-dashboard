# =============================================================================
#  Bora Alí — Dashboard Urbano (Laranja Sunset)
#  FINAL SR2 — 7 visualizações + Previsão 2026 + Ranking por Destino
#  NÃO CRIE PASTAS. Basta ter INMET_ANAC_ROTAS_APENAS_CAPITAIS.csv NA RAIZ.
# =============================================================================

# ---------------------------------------------
# IMPORTS
# ---------------------------------------------
import os
import unicodedata
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from prophet import Prophet
from prophet.plot import plot_plotly

# ---------------------------------------------
# CONFIGURAÇÃO DA PÁGINA 🔶
# ---------------------------------------------
st.set_page_config(
    page_title="Bora Alí — Dashboard Urbano",
    layout="wide",
    page_icon="🧳"
)

CSV_FILE = "INMET_ANAC_ROTAS_APENAS_CAPITAIS.csv"

# 🎨 PALETA LARANJA SUNSET
ORANGE = "#FF6A00"
PURPLE = "#6328E0"
SOFT = "#FFD199"
BG = "#FDFBFA"
TEXT = "#0F172A"

# Estilo global CSS
st.markdown(f"""
<style>
body {{
    background-color:{BG};
}}
h1,h2,h3,h4,h5 {{
    color:{PURPLE};
    font-weight:800;
}}
.stButton>button {{
    background:{ORANGE};
    color:white;
    border-radius:10px;
    font-weight:700;
    padding:6px 12px;
}}
</style>
""", unsafe_allow_html=True)

st.title("🧳 Bora Alí — Dashboard Urbano (Laranja Sunset)")
st.caption("Capitais do Brasil, Rotas Aéreas + Temperatura | Jovem, Inteligente e Visual 🔥")

# ---------------------------------------------
# FUNÇÕES DE NORMALIZAÇÃO DE CIDADES
# ---------------------------------------------
def normalize_str(s):
    if pd.isna(s): return s
    s = str(s)
    s = "".join(ch for ch in unicodedata.normalize("NFKD", s) if not unicodedata.combining(ch))
    s = s.replace("_"," ").replace("-"," ")
    return " ".join(s.split()).strip().lower()

# Lista de capitais
CANONICAL = [
    "Rio Branco","Maceió","Macapá","Manaus","Salvador","Fortaleza","Brasília","Vitória","Goiânia",
    "São Luís","Cuiabá","Campo Grande","Belo Horizonte","Belém","João Pessoa","Curitiba","Recife",
    "Teresina","Rio de Janeiro","Natal","Porto Alegre","Porto Velho","Boa Vista","Florianópolis",
    "Aracaju","São Paulo","Palmas"
]
NORM_TO_CANON = {normalize_str(c): c for c in CANONICAL}

def map_city(city):
    if pd.isna(city): return city
    c = normalize_str(city)
    if c in NORM_TO_CANON: return NORM_TO_CANON[c]
    return city.title()

def parse_route(r):
    if pd.isna(r): return (None,None)
    s = str(r)
    for sep in ["→","-","/"]:
        if sep in s:
            p=[x.strip() for x in s.split(sep)]
            if len(p)>=2: return (p[0],p[-1])
    return (None,None)

# ---------------------------------------------
# LEITURA DO CSV + TRATAMENTO
# ---------------------------------------------
@st.cache_data
def load_csv(path):
    try:
        df = pd.read_csv(path, low_memory=False)
    except:
        st.error(f"⛔ CSV NÃO ENCONTRADO: {path} — Coloque o arquivo na raiz e recarregue.")
        st.stop()

    df.columns = [c.upper().strip() for c in df.columns]
    for c in ["TARIFA","TEMP_MEDIA","TEMP_MIN","TEMP_MAX"]:
        df[c] = pd.to_numeric(df.get(c), errors="coerce")

    # Tratar temperatura média se faltar
    if df["TEMP_MEDIA"].isna().all():
        df["TEMP_MEDIA"] = (df["TEMP_MIN"] + df["TEMP_MAX"]) / 2

    # Tratar origem/destino/rota
    parsed = df["ROTA"].apply(lambda r: pd.Series(parse_route(r), index=["_o","_d"]))
    df["ORIG"] = df.get("ORIGEM", parsed["_o"]).fillna(parsed["_o"]).apply(map_city)
    df["DEST"] = df.get("DESTINO", parsed["_d"]).fillna(parsed["_d"]).apply(map_city)

    # Datas
    df["ANO"]=pd.to_numeric(df["ANO"],errors="coerce").fillna(0).astype(int)
    df["MES"]=pd.to_numeric(df["MES"],errors="coerce").fillna(0).astype(int)
    df["DATA"]=pd.to_datetime(df["ANO"].astype(str)+"-"+df["MES"].astype(str).str.zfill(2)+"-01")

    # Nome do mês
    MESES_NOME = {
        1:"Janeiro",2:"Fevereiro",3:"Março",4:"Abril",5:"Maio",6:"Junho",
        7:"Julho",8:"Agosto",9:"Setembro",10:"Outubro",11:"Novembro",12:"Dezembro"
    }
    df["MES_NOME"]=df["MES"].map(MESES_NOME)

    # Estações
    def est(m):
        if m in [12,1,2]: return "Verão"
        if m in [3,4,5]: return "Outono"
        if m in [6,7,8]: return "Inverno"
        return "Primavera"
    df["ESTACAO"] = df["MES"].apply(est)

    df["ROTA"]=df["ORIG"]+" → "+df["DEST"]
    return df

df = load_csv(CSV_FILE)

# ---------------------------------------------
# COORDENADAS DAS CAPITAIS DO BRASIL
# ---------------------------------------------
COORDS={
'Rio Branco':(-9.97499,-67.8243),'Maceió':(-9.6498,-35.7089),'Macapá':(0.0349,-51.0694),
'Manaus':(-3.1190,-60.0217),'Salvador':(-12.9713,-38.5013),'Fortaleza':(-3.7172,-38.5433),
'Brasília':(-15.7938,-47.8827),'Vitória':(-20.3155,-40.3128),'Goiânia':(-16.6868,-49.2647),
'São Luís':(-2.52972,-44.3027),'Cuiabá':(-15.6014,-56.0978),'Campo Grande':(-20.4433,-54.6465),
'Belo Horizonte':(-19.9166,-43.9344),'Belém':(-1.4558,-48.5044),'João Pessoa':(-7.1194,-34.8450),
'Curitiba':(-25.4295,-49.2712),'Recife':(-8.0475,-34.8770),'Teresina':(-5.08921,-42.8016),
'Rio de Janeiro':(-22.9068,-43.1728),'Natal':(-5.795,-35.209),'Porto Alegre':(-30.0346,-51.2176),
'Porto Velho':(-8.7608,-63.9039),'Boa Vista':(2.8196,-60.6733),'Florianópolis':(-27.5953,-48.5480),
'Aracaju':(-10.9472,-37.0731),'São Paulo':(-23.55052,-46.633308),'Palmas':(-10.184,-48.333)
}

# ---------------------------------------------
# SIDEBAR — FILTROS
# ---------------------------------------------
st.sidebar.header("🎯 Filtros Inteligentes")

anos = sorted(df["ANO"].unique())
meses = sorted(df["MES_NOME"].dropna().unique())
companias = sorted(df["COMPANHIA"].dropna().unique())
estacoes=["Verão","Outono","Inverno","Primavera"]
caps = sorted(list(COORDS.keys()))

sel_ano = st.sidebar.multiselect("Ano", anos, default=anos)
sel_mes = st.sidebar.multiselect("Mês", meses, default=meses)
sel_comp = st.sidebar.multiselect("Companhia", companias, default=companias)
sel_est = st.sidebar.multiselect("Estação", estacoes, default=estacoes)
sel_cap = st.sidebar.multiselect("Capitais", caps, default=caps)

dff = df[
    (df["ANO"].isin(sel_ano)) &
    (df["MES_NOME"].isin(sel_mes)) &
    (df["COMPANHIA"].isin(sel_comp)) &
    (df["ESTACAO"].isin(sel_est)) &
    (df["ORIG"].isin(sel_cap)) &
    (df["DEST"].isin(sel_cap))
]

if dff.empty:
    st.error("⛔ Nenhum registro com esses filtros!")
    st.stop()

# ---------------------------------------------
# KPIs
# ---------------------------------------------
st.markdown("---")
c1,c2,c3,c4 = st.columns(4)
c1.metric("📊 Registros", f"{len(dff):,}")
c2.metric("💰 Tarifa média", f"R$ {dff['TARIFA'].mean():.0f}")
c3.metric("🌡 Temp média", f"{dff['TEMP_MEDIA'].mean():.1f} °C")
c4.metric("✈️ Rotas únicas", dff["ROTA"].nunique())

# ---------------------------------------------
# 1) MAPA — CAPITAIS (TARIFA & TEMPERATURA)
# ---------------------------------------------
st.markdown("---")
st.subheader("🗺️ 1) Mapa das Capitais — Tarifa vs Temperatura")

agg = dff.groupby("DEST").agg(
    tarifa=("TARIFA","mean"),
    temp=("TEMP_MEDIA","mean"),
    regs=("TARIFA","count")
).reset_index()
agg["lat"]=agg["DEST"].map(lambda x:COORDS[x][0])
agg["lon"]=agg["DEST"].map(lambda x:COORDS[x][1])

fig1 = px.scatter_mapbox(
    agg, lat="lat", lon="lon",
    size="tarifa", color="temp",
    hover_name="DEST",
    hover_data={"tarifa":":.0f","temp":":.1f","regs":True,"lat":False,"lon":False},
    size_max=45, zoom=3.1,
    color_continuous_scale=[SOFT,ORANGE,PURPLE]
)
fig1.update_layout(
    mapbox_style="carto-positron",
    margin=dict(l=0,r=0,t=0,b=0)
)
st.plotly_chart(fig1, use_container_width=True)

# ---------------------------------------------
# 2) MAPA DE ROTAS — Premium
# ---------------------------------------------
st.markdown("---")
st.subheader("🛫 2) Rotas Premium (espessura proporcional à tarifa)")

routes = dff.groupby("ROTA").agg(tm=("TARIFA","mean"),regs=("TARIFA","count")).reset_index()
routes[["O","D"]] = routes["ROTA"].apply(lambda r: pd.Series(parse_route(r)))
routes["olat"]=routes["O"].map(lambda x:COORDS.get(map_city(x),(np.nan,np.nan))[0])
routes["olon"]=routes["O"].map(lambda x:COORDS.get(map_city(x),(np.nan,np.nan))[1])
routes["dlat"]=routes["D"].map(lambda x:COORDS.get(map_city(x),(np.nan,np.nan))[0])
routes["dlon"]=routes["D"].map(lambda x:COORDS.get(map_city(x),(np.nan,np.nan))[1])
routes=routes.dropna()

if not routes.empty:
    q=routes["tm"].quantile([0.25,0.5,0.75])
    def width(x):
        return 1.2 if x<=q[0.25] else 2.5 if x<=q[0.5] else 4 if x<=q[0.75] else 6
    routes["w"]=routes["tm"].apply(width)

    fig2=go.Figure()
    for _,r in routes.iterrows():
        fig2.add_trace(go.Scattermapbox(
            lat=[r["olat"],r["dlat"]], lon=[r["olon"],r["dlon"]],
            mode="lines",
            line=dict(width=r["w"],color=ORANGE),
            hoverinfo="text",
            text=f"<b>{r['ROTA']}</b><br>💰 R$ {r['tm']:.0f}<br>📌 {int(r['regs'])} registros"
        ))
    fig2.update_layout(
        mapbox_style="carto-positron",
        mapbox_center={"lat":-14.2,"lon":-51.9},
        mapbox_zoom=3.1,
        height=540,
        margin=dict(l=0,r=0,t=0,b=0)
    )
    st.plotly_chart(fig2, use_container_width=True)
else:
    st.info("Sem rotas com os filtros selecionados.")

# ---------------------------------------------
# 3) Ranking Interativo — DESTINO
# ---------------------------------------------
st.markdown("---")
st.subheader("🏆 3) Ranking Interativo — Destinos mais caros")

rank = dff.groupby("DEST").agg(m=("TARIFA","mean"),reg=("TARIFA","count")).reset_index()
rank = rank.sort_values("m",ascending=False)

fig3 = px.bar(rank, x="DEST", y="m", color="m",
              text=rank["m"].round(0),
              color_continuous_scale=[SOFT,ORANGE,PURPLE])
fig3.update_traces(textposition="outside")
fig3.update_layout(yaxis_title="Tarifa média (R$)", xaxis_title="Destino")
st.plotly_chart(fig3, use_container_width=True)

# ---------------------------------------------
# 4) Temporal
# ---------------------------------------------
st.markdown("---")
st.subheader("📈 4) Série Temporal — Variação de Tarifas")

ts = dff.groupby("DATA").agg(m=("TARIFA","mean")).reset_index()
st.plotly_chart(px.line(ts,x="DATA",y="m",markers=True,color_discrete_sequence=[ORANGE])
                .update_layout(yaxis_title="Tarifa média (R$)"), use_container_width=True)

# ---------------------------------------------
# 5) Estações
# ---------------------------------------------
st.markdown("---")
st.subheader("🌦 5) Tarifa por Estação")

est = dff.groupby("ESTACAO").agg(m=("TARIFA","mean")).reset_index()
st.plotly_chart(px.bar(est,x="ESTACAO",y="m",text=est["m"].round(0),
        color="ESTACAO",color_discrete_sequence=[SOFT,ORANGE,PURPLE,"#FFC872"])
                .update_traces(textposition="outside"),use_container_width=True)

# ---------------------------------------------
# 6) Regiões
# ---------------------------------------------
st.markdown("---")
st.subheader("🌎 6) Regiões com Tarifas mais caras")

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
regm=dff.groupby("REGIAO").agg(m=("TARIFA","mean")).reset_index()

st.plotly_chart(
    px.bar(regm,x="REGIAO",y="m",text=regm["m"].round(0),
    color="REGIAO",color_discrete_sequence=[ORANGE,PURPLE,SOFT,"#A6E3E9","#FF8C42"])
    .update_traces(textposition="outside")
    .update_layout(yaxis_title="Tarifa média (R$)"), use_container_width=True
)

# ---------------------------------------------
# 7) Heatmap
# ---------------------------------------------
st.markdown("---")
st.subheader("🔥 7) Heatmap — Tarifas (Mês x Destino)")

hm=dff.groupby(["MES_NOME","DEST"]).agg(m=("TARIFA","mean")).reset_index()
pv=hm.pivot(index="DEST",columns="MES_NOME",values="m")
st.plotly_chart(px.imshow(pv,color_continuous_scale=[SOFT,ORANGE,PURPLE],
                          labels=dict(color="Tarifa média (R$)")),use_container_width=True)

# ---------------------------------------------
# 📌 PREVISÃO 2026 — Prophet
# ---------------------------------------------
st.markdown("---")
st.header("🔮 Previsão 2026 — Tarifas por Rota")

rota_escolha=st.selectbox("Escolha uma Rota:",sorted(dff["ROTA"].unique()))
dfp=dff[dff["ROTA"]==rota_escolha].groupby("DATA").agg(tar=("TARIFA","mean"),
                                                        temp=("TEMP_MEDIA","mean")).reset_index()

if dfp.shape[0]>=12:
    dfp2=dfp.rename(columns={"DATA":"ds","tar":"y","temp":"temp"})
    model=Prophet(yearly_seasonality=True)
    model.add_regressor("temp",mode="additive")
    model.fit(dfp2)
    future=model.make_future_dataframe(periods=12,freq="MS")
    future["temp"]=dfp2["temp"].mean()
    fc=model.predict(future)
    st.plotly_chart(plot_plotly(model,fc),use_container_width=True)
else:
    st.warning("📌 Essa rota tem histórico insuficiente (mínimo 12 meses).")

st.caption("🌇 Bora Alí © — Laranja Sunset | SR2 — Design Jovem, Urbano e Inteligente ✨")

