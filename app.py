# app.py — Bora Alí (FINAL com previsões por destino + manter TODOS registros)
# Requisitos: INMET_ANAC_ROTAS_APENAS_CAPITAIS.csv na raiz do repositório.
# Recomendo requirements: streamlit, pandas, numpy, plotly, prophet, kaleido, cmdstanpy

import os
import unicodedata
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from prophet import Prophet
from prophet.plot import plot_plotly
from datetime import datetime

# -----------------------
# Config / Palette
# -----------------------
st.set_page_config(page_title="Bora Alí — Capitais (Final)", layout="wide", page_icon="🧳")
CSV_FILE = "INMET_ANAC_ROTAS_APENAS_CAPITAIS.csv"
OUTPUT_DIR = "outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

ORANGE = "#FF6A00"
PURPLE = "#6328E0"
SOFT = "#FFD199"
BG = "#FDFBFA"

st.markdown(f"""
<style>
body {{ background:{BG}; }}
h1,h2,h3 {{ color:{PURPLE}; font-weight:800; }}
.stButton>button {{ background:{ORANGE}; color:white; border-radius:8px; font-weight:700; }}
</style>
""", unsafe_allow_html=True)

st.title("🧳 Bora Alí — Capitais (Final)")
st.caption("Dashboard urbano • previsões 2026 por destino • picos históricos • todos os registros (opcional)")

# -----------------------
# Helpers
# -----------------------
def normalize_city(x):
    if pd.isna(x): return x
    s = str(x)
    s = "".join(ch for ch in unicodedata.normalize("NFKD", s) if not unicodedata.combining(ch))
    s = s.replace("_"," ").replace("-"," ")
    return " ".join(s.split()).title()

def parse_route(r):
    if pd.isna(r): return (None,None)
    s = str(r)
    for sep in ["→","-","/"]:
        if sep in s:
            parts = [p.strip() for p in s.split(sep) if p.strip()]
            if len(parts)>=2:
                return parts[0], parts[-1]
    return (None,None)

# Month mapping and ordered list
MESES = {1:"Janeiro",2:"Fevereiro",3:"Março",4:"Abril",5:"Maio",6:"Junho",
         7:"Julho",8:"Agosto",9:"Setembro",10:"Outubro",11:"Novembro",12:"Dezembro"}
MESES_ORDEM = [MESES[i] for i in range(1,13)]

def season_of(m):
    if m in [12,1,2]: return "Verão"
    if m in [3,4,5]: return "Outono"
    if m in [6,7,8]: return "Inverno"
    return "Primavera"

# -----------------------
# Load CSV robust
# -----------------------
@st.cache_data(ttl=600)
def load_data(path):
    try:
        df = pd.read_csv(path, low_memory=False)
    except Exception as e:
        st.error(f"Arquivo não encontrado na raiz: {path} — coloque o CSV e redeploy. Erro: {e}")
        st.stop()
    df.columns = [c.strip().upper() for c in df.columns]
    # Ensure essential cols
    for c in ["ROTA","COMPANHIA","ORIGEM","DESTINO","TARIFA","TEMP_MEDIA","TEMP_MIN","TEMP_MAX","ANO","MES"]:
        if c not in df.columns:
            df[c] = pd.NA
    # temp media fallback
    df["TARIFA"] = pd.to_numeric(df["TARIFA"], errors="coerce")
    df["TEMP_MEDIA"] = pd.to_numeric(df["TEMP_MEDIA"], errors="coerce")
    if df["TEMP_MEDIA"].isna().all() and ("TEMP_MIN" in df.columns and "TEMP_MAX" in df.columns):
        df["TEMP_MEDIA"] = (pd.to_numeric(df["TEMP_MIN"], errors="coerce") + pd.to_numeric(df["TEMP_MAX"], errors="coerce"))/2
    # parse origin/dest
    parsed = df["ROTA"].apply(lambda r: pd.Series(parse_route(r), index=["_o","_d"]))
    df["ORIG"] = df["ORIGEM"].where(df["ORIGEM"].notna(), parsed["_o"])
    df["DEST"] = df["DESTINO"].where(df["DESTINO"].notna(), parsed["_d"])
    df["ORIG"] = df["ORIG"].astype(str).apply(normalize_city)
    df["DEST"] = df["DEST"].astype(str).apply(normalize_city)
    # dates
    df["ANO"] = pd.to_numeric(df["ANO"], errors="coerce").fillna(0).astype(int)
    df["MES"] = pd.to_numeric(df["MES"], errors="coerce").fillna(0).astype(int)
    df["MES_NOME"] = df["MES"].map(MESES)
    df["ESTACAO"] = df["MES"].apply(lambda x: season_of(int(x)) if not pd.isna(x) else np.nan)
    df["DATA"] = pd.to_datetime(df["ANO"].astype(str) + "-" + df["MES"].astype(str).str.zfill(2) + "-01", errors="coerce")
    df["ROTA"] = df["ORIG"].fillna("") + " → " + df["DEST"].fillna("")
    return df

df = load_data(CSV_FILE)

# -----------------------
# Coordinates (canonical capitals)
# -----------------------
COORDS = {
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

# -----------------------
# Sidebar: filters + "Mostrar todos"
# -----------------------
st.sidebar.header("Filtros — Bora Alí")
# Always show months in correct chronological order in sidebar
years = sorted(df["ANO"].dropna().unique())
months_all = MESES_ORDEM  # ordered names
companies = sorted(df["COMPANHIA"].dropna().unique())
seasons = ["Verão","Outono","Inverno","Primavera"]
capitals = sorted(list(COORDS.keys()))

# Toggle to show all records ignoring filters
show_all = st.sidebar.checkbox("🔓 Mostrar TODOS os registros (ignorar filtros)", value=True)

if show_all:
    dff = df.copy()
else:
    sel_years = st.sidebar.multiselect("Ano", years, default=years)
    sel_months = st.sidebar.multiselect("Mês", months_all, default=months_all)
    sel_companies = st.sidebar.multiselect("Companhia", companies, default=companies)
    sel_seasons = st.sidebar.multiselect("Estação", seasons, default=seasons)
    sel_caps = st.sidebar.multiselect("Capitais", capitals, default=capitals)

    dff = df[
        (df["ANO"].isin(sel_years)) &
        (df["MES_NOME"].isin(sel_months)) &
        (df["COMPANHIA"].isin(sel_companies)) &
        (df["ESTACAO"].isin(sel_seasons)) &
        (df["ORIG"].isin(sel_caps)) &
        (df["DEST"].isin(sel_caps))
    ].copy()

if dff.empty:
    st.warning("Nenhum registro após filtros. Ajuste os filtros.")
    st.stop()

# -----------------------
# KPIs
# -----------------------
st.markdown("---")
k1,k2,k3,k4 = st.columns(4)
k1.metric("Registros (filtrados)", f"{len(dff):,}")
k2.metric("Tarifa média (R$)", f"{dff['TARIFA'].mean():.0f}")
k3.metric("Temp média (°C)", f"{dff['TEMP_MEDIA'].mean():.1f}")
k4.metric("Rotas únicas", f"{dff['ROTA'].nunique():,}")

# -----------------------
# Map 1 — Capitals
# -----------------------
st.markdown("---")
st.subheader("1) Mapa — Capitais (tamanho = tarifa · cor = temperatura)")
agg = dff.groupby("DEST").agg(tarifa_media=("TARIFA","mean"), temp_media=("TEMP_MEDIA","mean"), regs=("TARIFA","count")).reset_index()
agg["lat"] = agg["DEST"].map(lambda x: COORDS.get(x,(np.nan,np.nan))[0])
agg["lon"] = agg["DEST"].map(lambda x: COORDS.get(x,(np.nan,np.nan))[1])

fig1 = px.scatter_mapbox(
    agg.dropna(subset=["lat","lon"]),
    lat="lat", lon="lon",
    size="tarifa_media", color="temp_media",
    hover_name="DEST",
    hover_data={"tarifa_media":":.0f","temp_media":":.1f","regs":True,"lat":False,"lon":False},
    size_max=45, zoom=3.2, color_continuous_scale=[SOFT, ORANGE, PURPLE], height=480
)
fig1.update_layout(mapbox_style="carto-positron", margin=dict(l=0,r=0,t=0,b=0))
st.plotly_chart(fig1, use_container_width=True)

# -----------------------
# Map 2 — Routes (clear text for leigos)
# -----------------------
st.markdown("---")
st.subheader("2) Rotas Mais Caras do Brasil (quanto mais grossa, maior o preço médio)")
st.caption("Linhas mais espessas = rotas com tarifa média mais alta. Hover: rota, preço médio (R$), registros.")

routes = dff.groupby("ROTA").agg(tarifa_media=("TARIFA","mean"), regs=("TARIFA","count")).reset_index()
routes[["ORIG","DEST"]] = routes["ROTA"].apply(lambda r: pd.Series(parse_route(r)))
routes["olat"] = routes["ORIG"].map(lambda x: COORDS.get(x,(np.nan,np.nan))[0])
routes["olon"] = routes["ORIG"].map(lambda x: COORDS.get(x,(np.nan,np.nan))[1])
routes["dlat"] = routes["DEST"].map(lambda x: COORDS.get(x,(np.nan,np.nan))[0])
routes["dlon"] = routes["DEST"].map(lambda x: COORDS.get(x,(np.nan,np.nan))[1])
routes = routes.dropna(subset=["olat","olon","dlat","dlon","tarifa_media"])
routes = routes[routes["tarifa_media"]>0].copy()

if not routes.empty:
    q = routes["tarifa_media"].quantile([0,0.25,0.5,0.75]).tolist()
    def safe_width(x):
        try: x = float(x)
        except: return 1.0
        if x <= q[0]: return 1.2
        if x <= q[1]: return 2.6
        if x <= q[2]: return 4.0
        return 6.5
    routes["width"] = routes["tarifa_media"].apply(safe_width)

    fig2 = go.Figure()
    for _, r in routes.iterrows():
        fig2.add_trace(go.Scattermapbox(
            lat=[r["olat"], r["dlat"]], lon=[r["olon"], r["dlon"]],
            mode="lines",
            line=dict(width=float(r["width"]), color=ORANGE, opacity=0.85),
            hoverinfo="text",
            text=f"<b>Rota</b>: {r['ROTA']}<br><b>Tarifa média</b>: R$ {r['tarifa_media']:.0f}<br><b>Registros</b>: {int(r['regs'])}"
        ))
    fig2.update_layout(mapbox_style="carto-positron", mapbox_center={"lat":-14.2,"lon":-51.9}, mapbox_zoom=3.2,
                       margin=dict(l=0,r=0,t=0,b=0), height=520)
    st.plotly_chart(fig2, use_container_width=True)
else:
    st.info("Sem rotas disponíveis com os filtros atuais.")

# -----------------------
# Ranking interativo por DEST
# -----------------------
st.markdown("---")
st.subheader("3) Ranking — Destinos (interativo)")

rank = dff.groupby("DEST").agg(tarifa_media=("TARIFA","mean"), regs=("TARIFA","count")).reset_index().sort_values("tarifa_media", ascending=False)
fig_rank = px.bar(rank, x="DEST", y="tarifa_media", color="tarifa_media", text=rank["tarifa_media"].round(0), color_continuous_scale=[SOFT,ORANGE,PURPLE])
fig_rank.update_traces(texttemplate="R$ %{text:.0f}", textposition="outside")
fig_rank.update_layout(yaxis_title="Tarifa média (R$)", xaxis_title="Destino")
st.plotly_chart(fig_rank, use_container_width=True)

# -----------------------
# Time series
# -----------------------
st.markdown("---")
st.subheader("4) Série temporal — Tarifa média mensal (agregado)")
ts = dff.groupby("DATA").agg(tarifa_media=("TARIFA","mean")).reset_index().sort_values("DATA")
fig_ts = px.line(ts, x="DATA", y="tarifa_media", markers=True, title="Tarifa média mensal")
fig_ts.update_layout(yaxis_title="Tarifa média (R$)")
st.plotly_chart(fig_ts, use_container_width=True)

# -----------------------
# Tarifa por estação
# -----------------------
st.markdown("---")
st.subheader("5) Tarifa média por estação")
est = dff.groupby("ESTACAO").agg(tarifa_media=("TARIFA","mean")).reindex(["Verão","Outono","Inverno","Primavera"]).reset_index()
est["tarifa_media"] = est["tarifa_media"].round(0)
fig_est = px.bar(est, x="ESTACAO", y="tarifa_media", color="ESTACAO", text="tarifa_media", color_discrete_sequence=[ORANGE, SOFT, PURPLE, "#7BC4C4"])
fig_est.update_traces(texttemplate="R$ %{text:.0f}", textposition="outside")
fig_est.update_layout(yaxis_title="Tarifa média (R$)")
st.plotly_chart(fig_est, use_container_width=True)

# -----------------------
# Regiões
# -----------------------
st.markdown("---")
st.subheader("6) Tarifa média por região")
REG = {
    "Norte": ["Belém","Macapá","Manaus","Boa Vista","Rio Branco","Porto Velho","Palmas"],
    "Nordeste": ["São Luís","Teresina","Fortaleza","Natal","João Pessoa","Recife","Maceió","Aracaju","Salvador"],
    "Centro-Oeste": ["Brasília","Goiânia","Campo Grande","Cuiabá"],
    "Sudeste": ["São Paulo","Rio de Janeiro","Belo Horizonte","Vitória"],
    "Sul": ["Curitiba","Florianópolis","Porto Alegre"]
}
def region_of(city):
    for k,v in REG.items():
        if city in v: return k
    return "Outro"

dff["REGIAO"] = dff["DEST"].apply(region_of)
regm = dff.groupby("REGIAO").agg(tarifa_media=("TARIFA","mean")).reset_index()
regm["tarifa_media"] = regm["tarifa_media"].round(0)
fig_reg = px.bar(regm, x="REGIAO", y="tarifa_media", text="tarifa_media", color="REGIAO",
                 color_discrete_sequence=[ORANGE, PURPLE, "#16A34A", "#9333EA", "#E11D48"])
fig_reg.update_traces(texttemplate="R$ %{text:.0f}", textposition="outside")
fig_reg.update_layout(yaxis_title="Tarifa média (R$)")
st.plotly_chart(fig_reg, use_container_width=True)

# -----------------------
# 7) PREVISÃO 2026 por DESTINO + picos históricos
# -----------------------
st.markdown("---")
st.subheader("7) 🔮 Previsão 2026 — Tarifa média por destino & picos históricos")

# caching per-destination predictions to avoid heavy recompute
@st.cache_data(ttl=3600)
def predict_destinations(df_in):
    results = []
    # group monthly per destination
    dests = sorted(df_in["DEST"].unique())
    for i, dest in enumerate(dests):
        tmp = df_in[df_in["DEST"]==dest].groupby("DATA").agg(tarifa=("TARIFA","mean")).reset_index().dropna()
        if tmp.shape[0] < 12:
            # insufficient history -> skip or mark NaN
            results.append({"DESTINO": dest, "TARIFA_2026": np.nan, "HIST_PICO_MES": np.nan, "HIST_PICO_ESTACAO": np.nan})
            continue
        # fit prophet
        dfp = tmp.rename(columns={"DATA":"ds","tarifa":"y"})
        try:
            m = Prophet(yearly_seasonality=True)
            m.fit(dfp)
            future = m.make_future_dataframe(periods=12, freq="MS")
            forecast = m.predict(future)
            # mean yhat for 2026
            yhat2026 = forecast[forecast["ds"].dt.year==2026]["yhat"].mean()
        except Exception:
            yhat2026 = np.nan
        # historical peak month & season
        hist = tmp.copy()
        hist["MES"] = hist["DATA"].dt.month
        month_avg = hist.groupby("MES")["tarifa"].mean().reset_index()
        if not month_avg.empty:
            peak_month_num = int(month_avg.loc[month_avg["tarifa"].idxmax(),"MES"])
            peak_month_name = MESES.get(peak_month_num, str(peak_month_num))
            peak_season = season_of(peak_month_num)
        else:
            peak_month_name = np.nan
            peak_season = np.nan
        results.append({"DESTINO": dest, "TARIFA_2026": yhat2026, "HIST_PICO_MES": peak_month_name, "HIST_PICO_ESTACAO": peak_season})
    return pd.DataFrame(results)

with st.spinner("Calculando previsões 2026 por destino (pode levar alguns segundos)..."):
    df_preds = predict_destinations(dff)

# prepare display: round prices
df_preds["TARIFA_2026_ROUND"] = df_preds["TARIFA_2026"].round(0)
df_preds_sorted = df_preds.sort_values("TARIFA_2026", ascending=False)

st.markdown("**Previsão média anual (2026) por destino** — valores em R$ (média anual prevista):")
st.dataframe(df_preds_sorted[["DESTINO","TARIFA_2026_ROUND","HIST_PICO_MES","HIST_PICO_ESTACAO"]].rename(columns={
    "DESTINO":"Destino","TARIFA_2026_ROUND":"Tarifa prevista 2026 (R$)","HIST_PICO_MES":"Pico histórico (mês)","HIST_PICO_ESTACAO":"Pico histórico (estação)"
}), height=360)

# Visual: top 12 predicted destinations
top_n = 12
top_preds = df_preds_sorted.head(top_n).dropna(subset=["TARIFA_2026"])
if not top_preds.empty:
    fig_pred = px.bar(top_preds, x="DESTINO", y="TARIFA_2026", text=top_preds["TARIFA_2026_ROUND"].astype(int),
                      color="TARIFA_2026", color_continuous_scale=[SOFT,ORANGE,PURPLE])
    fig_pred.update_traces(texttemplate="R$ %{text:.0f}", textposition="outside")
    fig_pred.update_layout(yaxis_title="Tarifa média prevista 2026 (R$)", xaxis_title="Destino")
    st.plotly_chart(fig_pred, use_container_width=True)
else:
    st.info("Dados insuficientes para gerar previsões por destino.")

# Insights: top 3 predicted and their historic peaks
valid_preds = df_preds_sorted.dropna(subset=["TARIFA_2026"]).head(5)
if not valid_preds.empty:
    top3 = valid_preds.head(3)
    insight_lines = []
    for _, row in top3.iterrows():
        insight_lines.append(f"• {row['DESTINO']}: R$ {row['TARIFA_2026_ROUND']:.0f} (pico histórico em {row['HIST_PICO_MES']} — {row['HIST_PICO_ESTACAO']})")
    st.success("🔎 Insights rápidos: " + "  ".join(insight_lines))

# Export predictions CSV button
if st.button("Exportar previsões 2026 (CSV)"):
    out = os.path.join(OUTPUT_DIR, f"previsoes_2026_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv")
    df_preds_sorted.to_csv(out, index=False)
    st.success(f"Arquivo salvo em: {out}")

st.caption("Dashboard final — se quiser, peço permissão para ajustar visual (ícones, fontes) e gerar capa.")
