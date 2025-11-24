# =============================================================================
# Bora Alí — SR2 Premium (Rotas em Alerta • Previsão 2026 • Insights)
# 100% estável • Previsão suave • Insights SR2 • Mapa só Brasil
# =============================================================================

import os
import unicodedata
from datetime import datetime
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from sklearn.linear_model import LinearRegression

# =============================================================================
# CONFIGURAÇÃO DE TEMA
# =============================================================================

st.set_page_config(page_title="Bora Alí — SR2 Premium", layout="wide", page_icon="✈️")

PURPLE = "#5A189A"
PINK = "#E11D48"
ORANGE = "#FF6A00"
GREEN = "#10B981"
GRAFITE = "#1F1F1F"
BG = "#FCF8FF"

st.markdown(f"""
<style>
body {{ background-color: {BG}; }}
h1,h2,h3,h4 {{
    color: {PURPLE};
    font-weight: 800;
}}
.card {{
    background: white;
    padding: 16px;
    border-radius: 14px;
    box-shadow: 0px 4px 12px rgba(0,0,0,0.12);
    margin-bottom: 12px;
}}
.small {{ font-size: 14px; color: #555; }}
</style>
""", unsafe_allow_html=True)

# =============================================================================
# ARQUIVO CSV
# =============================================================================

CSV_FILE = "INMET_ANAC_ROTAS_APENAS_CAPITAIS.csv"

if not os.path.exists(CSV_FILE):
    st.error("Arquivo CSV não encontrado. Coloque INMET_ANAC_ROTAS_APENAS_CAPITAIS.csv na raiz.")
    st.stop()

# =============================================================================
# FUNÇÕES AUXILIARES
# =============================================================================

def clean(s):
    if pd.isna(s): return None
    s = str(s)
    s = "".join(c for c in unicodedata.normalize("NFKD", s) if not unicodedata.combining(c))
    return s.strip().title()

def parse_route(r):
    if pd.isna(r): return (None,None)
    s = str(r)
    for sep in ["→","->","-","/"]:
        if sep in s:
            p = [x.strip() for x in s.split(sep) if x.strip()]
            if len(p) >= 2:
                return (clean(p[0]), clean(p[-1]))
    return (None, None)

MES_NOME = {
    1:"Janeiro",2:"Fevereiro",3:"Março",4:"Abril",5:"Maio",6:"Junho",
    7:"Julho",8:"Agosto",9:"Setembro",10:"Outubro",11:"Novembro",12:"Dezembro"
}

def estacao(m):
    if m in [12,1,2]: return "Verão"
    if m in [3,4,5]: return "Outono"
    if m in [6,7,8]: return "Inverno"
    return "Primavera"

def normalize_company(c):
    if pd.isna(c): return None
    c = c.strip().upper()
    if "AZUL" in c: return "Azul"
    if "GOL" in c: return "GOL"
    if "LATAM" in c or "TAM" in c: return "LATAM"
    return c.title()

# =============================================================================
# CARREGAR CSV
# =============================================================================

@st.cache_data
def load_df(path):
    df = pd.read_csv(path)
    df.columns = [c.upper() for c in df.columns]

    df["COMP_NORM"] = df["COMPANHIA"].apply(normalize_company)

    parsed = df["ROTA"].apply(lambda r: pd.Series(parse_route(r), index=["_o","_d"]))
    df["ORIG"] = parsed["_o"].fillna(df.get("ORIGEM")).apply(clean)
    df["DEST"] = parsed["_d"].fillna(df.get("DESTINO")).apply(clean)

    df["ANO"] = df["ANO"].astype(int)
    df["MES"] = df["MES"].astype(int)

    df["DATA"] = pd.to_datetime(df["ANO"].astype(str)+"-"+df["MES"].astype(str).str.zfill(2)+"-01")

    df["MES_NOME"] = df["MES"].map(MES_OK)
    df["ESTACAO"] = df["MES"].map(estacao)

    df["ROTA"] = df["ORIG"]+" → "+df["DEST"]

    return df.dropna(subset=["TARIFA","ORIG","DEST"])

df = load_df(CSV_FILE)

# =============================================================================
# SIDEBAR
# =============================================================================

st.sidebar.header("Filtros SR2")

sel_anos = st.sidebar.multiselect("Ano", sorted(df["ANO"].unique()), default=sorted(df["ANO"].unique()))
sel_comp = st.sidebar.multiselect("Companhia", sorted(df["COMP_NORM"].unique()), default=sorted(df["COMP_NORM"].unique()))
sel_est = st.sidebar.multiselect("Estação", ["Verão","Outono","Inverno","Primavera"], default=["Verão","Outono","Inverno","Primavera"])

dff = df[
    df["ANO"].isin(sel_anos) &
    df["COMP_NORM"].isin(sel_comp) &
    df["ESTACAO"].isin(sel_est)
]

if dff.empty:
    st.error("Nenhum dado encontrado com esses filtros.")
    st.stop()

# =============================================================================
# KPIs
# =============================================================================

st.markdown("---")
a,b,c,d = st.columns(4)
a.metric("Registros", f"{len(dff):,}")
b.metric("Tarifa Média", f"R$ {dff['TARIFA'].mean():.0f}")
c.metric("Rotas Únicas", dff["ROTA"].nunique())
d.metric("Companhias", dff["COMP_NORM"].nunique())

# =============================================================================
# RANKING REGRESSÃO 2026 (SUAVE)
# =============================================================================

@st.cache_data
def regression_2026(df):
    out=[]
    grp = df.groupby(["ROTA","DATA"]).agg(m=("TARIFA","mean")).reset_index()

    for rota,g in grp.groupby("ROTA"):
        g=g.sort_values("DATA")
        if len(g)<6: continue

        g["t"] = np.arange(len(g))

        lr = LinearRegression().fit(g[["t"]], g["m"])

        future_t = np.arange(len(g), len(g)+12)
        preds = lr.predict(future_t.reshape(-1,1))

        out.append({
            "ROTA":rota,
            "ATUAL": g["m"].mean(),
            "PREV_2026": preds.mean(),
            "VAR_%": (preds.mean()-g["m"].mean())/g["m"].mean()
        })

    r = pd.DataFrame(out)
    if not r.empty:
        def sinal(v):
            if v>=0.20: return "🛑 Forte alta"
            if v>=0.05: return "⚠️ Atenção"
            if v<0: return "📉 Queda"
            return "📈 Leve alta"
        r["SINAL"] = r["VAR_%"].apply(sinal)
        r = r.sort_values("PREV_2026", ascending=False)
    return r

rank = regression_2026(dff)

st.markdown("---")
st.subheader("🏆 Ranking SR2 — Previsão 2026 (suave)")

st.dataframe(rank[["ROTA","ATUAL","PREV_2026","VAR_%","SINAL"]].round(2))

# =============================================================================
# MAPA SÓ DO BRASIL (LIMITE CORRETO)
# =============================================================================

BR_CENTER_LAT = -14.235
BR_CENTER_LON = -51.925

BR_BOUNDS = dict(
    west=-74,
    east=-34,
    south=-34,
    north=6
)

COORDS = {
'Rio Branco':(-9.97499,-67.8243),'Maceió':(-9.6498,-35.7089),'Macapá':(0.0349,-51.0694),
'Manaus':(-3.119,-60.021),'Salvador':(-12.9713,-38.5013),'Fortaleza':(-3.7172,-38.5433),
'Brasília':(-15.7938,-47.8827),'Vitória':(-20.3155,-40.3128),'Goiânia':(-16.6868,-49.2647),
'São Luís':(-2.52972,-44.3027),'Cuiabá':(-15.6014,-56.0978),'Campo Grande':(-20.4433,-54.6465),
'Belo Horizonte':(-19.9166,-43.9344),'Belém':(-1.4558,-48.5044),'João Pessoa':(-7.1194,-34.845),
'Curitiba':(-25.4295,-49.2712),'Recife':(-8.0475,-34.877),'Teresina':(-5.08921,-42.8016),
'Rio De Janeiro':(-22.9068,-43.1728),'Natal':(-5.795,-35.209),'Porto Alegre':(-30.0346,-51.2176),
'Porto Velho':(-8.7608,-63.9039),'Boa Vista':(2.8196,-60.6733),'Florianópolis':(-27.5953,-48.548),
'Aracaju':(-10.9472,-37.0731),'São Paulo':(-23.5505,-46.6333),'Palmas':(-10.184,-48.333)
}

st.markdown("---")
st.subheader("🗺️ Mapa — Rotas em Alerta (Brasil)")

fig = go.Figure()

if not rank.empty:
    R = rank.copy()
    R[["O","D"]] = R["ROTA"].apply(lambda r: pd.Series(parse_route(r)))

    for _,row in R.iterrows():
        o=row["O"]; d=row["D"]
        if o in COORDS and d in COORDS:
            olat,olon = COORDS[o]; dlat,dlon = COORDS[d]

            color = PINK if row["SINAL"]=="🛑 Forte alta" else ORANGE if row["SINAL"]=="⚠️ Atenção" else GREEN
            width = 6 if row["SINAL"]=="🛑 Forte alta" else 3

            fig.add_trace(go.Scattermapbox(
                lat=[olat,dlat], lon=[olon,dlon],
                mode="lines+markers",
                line=dict(color=color, width=width),
                marker=dict(size=6,color=color),
                hovertext=f"{row['ROTA']}<br>{row['SINAL']}<br>Prev 2026: R$ {row['PREV_2026']:.0f}"
            ))

fig.update_layout(
    mapbox_style="carto-positron",
    mapbox_center={"lat": BR_CENTER_LAT, "lon": BR_CENTER_LON},
    mapbox_zoom=3.4,
    mapbox_bounds=BR_BOUNDS,
    height=520,
    margin=dict(l=0,r=0,t=0,b=0)
)

st.plotly_chart(fig, use_container_width=True)

# =============================================================================
# INSIGHTS SR2 — 6 SINAIS
# =============================================================================

st.markdown("---")
st.subheader("💡 Insights SR2 — Fatos importantes")

ins_container = st.container()
cols = ins_container.columns(3)
ins = []

# 1 maior alerta
top1 = rank[rank["SINAL"]=="🛑 Forte alta"].head(1)
if not top1.empty:
    r=top1.iloc[0]
    ins.append(("🛑 Maior Alerta", f"{r['ROTA']} — previsão R$ {r['PREV_2026']:.0f} (+{r['VAR_%']:.0%})"))

# 2 estação crítica
est = dff.groupby("ESTACAO")["TARIFA"].mean().reset_index().sort_values("TARIFA",ascending=False)
if not est.empty:
    e=est.iloc[0]
    ins.append(("🌦 Estação Mais Cara", f"{e['ESTACAO']} — R$ {e['TARIFA']:.0f}"))

# 3 companhia mais cara
comp = dff.groupby("COMP_NORM")["TARIFA"].mean().reset_index().sort_values("TARIFA",ascending=False)
if not comp.empty:
    c=comp.iloc[0]
    ins.append(("✈️ Companhia Mais Cara", f"{c['COMP_NORM']} — R$ {c['TARIFA']:.0f}"))

# 4 mês de pico
m = dff.groupby("MES")["TARIFA"].mean().reset_index().sort_values("TARIFA",ascending=False)
if not m.empty:
    mm=m.iloc[0]
    ins.append(("📅 Mês de Pico", f"{MES_NOME[int(mm['MES'])]} — R$ {mm['TARIFA']:.0f}"))

# 5 rota mais volátil
vol = dff.groupby("ROTA").agg(std=("TARIFA","std"), mean=("TARIFA","mean")).dropna()
if not vol.empty:
    vol["cv"] = vol["std"]/vol["mean"]
    vv = vol.sort_values("cv",ascending=False).head(1)
    rota_nome = vv.index[0]
    x = vv.iloc[0]
    ins.append(("⚡ Volatilidade Máxima", f"{rota_nome} — CV {x['cv']:.2f}"))

# 6 melhor oportunidade
down = rank[rank["SINAL"]=="📉 Queda"].head(1)
if not down.empty:
    d=down.iloc[0]
    ins.append(("🎯 Oportunidade", f"{d['ROTA']} — queda prevista em 2026"))

for i,(title,text) in enumerate(ins):
    with cols[i%3]:
        st.markdown(f"<div class='card'><h4>{title}</h4><div class='small'>{text}</div></div>", unsafe_allow_html=True)

# =============================================================================
# PREVISÃO POR ROTA (SUAVE)
# =============================================================================

st.markdown("---")
st.header("🔮 Previsão Mensal 2026 — Rota Selecionada")

colA,colB,colC = st.columns([3,3,1])
orig = colA.selectbox("Origem", sorted(dff["ORIG"].unique()))
dest = colB.selectbox("Destino", sorted(dff["DEST"].unique()))

rota_sel = f"{orig} → {dest}"

if colC.button("Gerar"):

    sub = dff[dff["ROTA"]==rota_sel].groupby("DATA")["TARIFA"].mean().reset_index()
    if len(sub)<6:
        st.warning("Poucos dados para previsão.")
    else:
        sub=sub.sort_values("DATA")
        sub["t"]=np.arange(len(sub))

        lr=LinearRegression().fit(sub[["t"]], sub["TARIFA"])

        fut_t=np.arange(len(sub), len(sub)+12)
        preds = lr.predict(fut_t.reshape(-1,1))

        out = pd.DataFrame({
            "Mes":[MES_NOME[m] for m in range(1,13)],
            "Tarifa Prevista (R$)": preds.round(0)
        })

        st.dataframe(out)

        fig2 = px.line(out, x="Mes", y="Tarifa Prevista (R$)", markers=True, color_discrete_sequence=[PURPLE])
        st.plotly_chart(fig2, use_container_width=True)

st.caption("Bora Alí — SR2 Premium • versão estável e otimizada")
