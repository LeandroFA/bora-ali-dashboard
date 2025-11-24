# =============================================================================
# Bora Alí — SR2 (Versão Final Definitiva)
# - Compatível com CSV: COMPANHIA, ANO, MES, DESTINO, ROTA, TARIFA, TEMP_MEDIA
# - Origem extraída de ROTA (pois seu CSV não tem ORIGEM)
# - Previsões suaves + cap, mapa do Brasil com "S", 6 insights organizados
# - Remove imagens de companhias/estações (substituído por gráfico claro: Tarifas por Estação)
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

# Prophet optional
try:
    from prophet import Prophet
    HAS_PROPHET = True
except Exception:
    HAS_PROPHET = False

# ---------------------------
# Config & Theme
# ---------------------------
st.set_page_config(page_title="Bora Alí — SR2", layout="wide", page_icon="✈️")
PURPLE = "#5A189A"
PINK = "#E11D48"
ORANGE = "#FF6A00"
GREEN = "#10B981"
BG = "#FCF8FF"
TEXT = "#0F172A"

st.markdown(f"""
<style>
body {{ background-color:{BG}; color:{TEXT}; }}
h1,h2,h3,h4 {{ color:{PURPLE}; font-weight:800; }}
.stButton>button {{ background:{PURPLE}; color:white; border-radius:8px; padding:8px 14px; font-weight:700; }}
.card {{ background: white; padding:14px; border-radius:12px; box-shadow:0 6px 18px rgba(0,0,0,0.06); }}
.small {{ color:#4b5563; font-size:14px; }}
.big-title {{ font-size:28px; font-weight:900; color:{PURPLE}; }}
</style>
""", unsafe_allow_html=True)

st.title("🛑 Bora Alí — SR2: Rotas em Alerta (Final)")
st.caption("Visual jovem, previsões suavizadas e recomendações claras — Português (Brasil)")

# ---------------------------
# File path (uploaded)
# ---------------------------
# caminho local do CSV (usando o arquivo que você já enviou)
CSV_FILE = "/mnt/data/INMET_ANAC_ROTAS_APENAS_CAPITAIS.csv"

if not os.path.exists(CSV_FILE):
    st.error(f"Arquivo CSV não encontrado em: {CSV_FILE}\nColoque o arquivo na raiz /mnt/data/ e recarregue.")
    st.stop()

# ---------------------------
# Helpers
# ---------------------------
def clean_text(s):
    if pd.isna(s): return None
    s = str(s)
    s = "".join(ch for ch in unicodedata.normalize("NFKD", s) if not unicodedata.combining(ch))
    return " ".join(s.split()).strip().title()

def parse_route(route):
    if pd.isna(route): return (None, None)
    s = str(route)
    for sep in ["→","->","–","-","/"]:
        if sep in s:
            parts = [p.strip() for p in s.split(sep) if p.strip()]
            if len(parts) >= 2:
                return (clean_text(parts[0]), clean_text(parts[-1]))
    return (None, None)

def normalize_company(raw):
    if pd.isna(raw): return "Outras"
    x = str(raw).upper()
    if "LATAM" in x or "TAM" in x: return "LATAM"
    if "GOL" in x: return "GOL"
    if "AZUL" in x: return "AZUL"
    return clean_text(raw)

MES_NAME = {1:"Janeiro",2:"Fevereiro",3:"Março",4:"Abril",5:"Maio",6:"Junho",
            7:"Julho",8:"Agosto",9:"Setembro",10:"Outubro",11:"Novembro",12:"Dezembro"}
def season_from_month(m):
    if m in [12,1,2]: return "Verão"
    if m in [3,4,5]: return "Outono"
    if m in [6,7,8]: return "Inverno"
    return "Primavera"

# ---------------------------
# Load & prepare (robust)
# ---------------------------
@st.cache_data(show_spinner=False)
def load_prep(path):
    df = pd.read_csv(path, low_memory=False)
    df.columns = [c.upper().strip() for c in df.columns]

    # numeric conversions if present
    for c in ["TARIFA","TEMP_MEDIA","TEMP_MIN","TEMP_MAX","ANO","MES"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    # normalize company
    if "COMPANHIA" in df.columns:
        df["COMP_NORM"] = df["COMPANHIA"].apply(normalize_company)
    else:
        df["COMP_NORM"] = "Outras"

    # parse ROTA into ORIG/DEST (since ORIGEM not present)
    parsed = pd.DataFrame(df.get("ROTA", "").apply(lambda r: parse_route(r)).tolist(), columns=["_ORIG","_DEST"])
    df["ORIG"] = parsed["_ORIG"].apply(clean_text)
    df["DEST"] = parsed["_DEST"].apply(clean_text)

    # if DESTINO column exists, use to fill DEST gaps
    if "DESTINO" in df.columns:
        df["DEST"] = df["DEST"].fillna(df["DESTINO"].apply(clean_text))

    # enforce year/month and date
    df["ANO"] = pd.to_numeric(df.get("ANO", pd.NA), errors="coerce").fillna(0).astype(int)
    df["MES"] = pd.to_numeric(df.get("MES", pd.NA), errors="coerce").fillna(0).astype(int)
    df = df[(df["ANO"]>0) & (df["MES"]>0)]

    df["DATA"] = pd.to_datetime(df["ANO"].astype(str) + "-" + df["MES"].astype(str).str.zfill(2) + "-01", errors="coerce")

    # drop any rows missing essential fields
    df = df.dropna(subset=["DATA","ORIG","DEST","TARIFA"])

    # canonical route
    df["ROTA"] = df["ORIG"].astype(str) + " → " + df["DEST"].astype(str)
    df["MES_NOME"] = df["MES"].map(MES_NAME)
    df["ESTACAO"] = df["MES"].apply(season_from_month)
    return df

df = load_prep(CSV_FILE)
if df.empty:
    st.error("Após limpeza os dados ficaram vazios — verifique o CSV.")
    st.stop()

# ---------------------------
# Sidebar filters
# ---------------------------
st.sidebar.header("Filtros — Bora Alí (SR2)")
anos = sorted(df["ANO"].unique())
sel_anos = st.sidebar.multiselect("Ano", anos, default=anos)
companies = sorted(df["COMP_NORM"].unique())
sel_comp = st.sidebar.multiselect("Companhia", companies, default=companies)
estacoes = ["Verão","Outono","Inverno","Primavera"]
sel_est = st.sidebar.multiselect("Estação", estacoes, default=estacoes)
top_n = st.sidebar.slider("Top N rotas (mapa/ranking)", 5, 25, value=12)

# filtered DF for dashboard
dff = df[(df["ANO"].isin(sel_anos)) & (df["COMP_NORM"].isin(sel_comp)) & (df["ESTACAO"].isin(sel_est))]
if dff.empty:
    st.error("Filtros retornaram conjunto vazio.")
    st.stop()

# ---------------------------
# Top KPIs
# ---------------------------
st.markdown("---")
k1,k2,k3,k4 = st.columns(4)
k1.metric("Registros (filtros)", f"{len(dff):,}")
k2.metric("Tarifa média", f"R$ {dff['TARIFA'].mean():.0f}")
k3.metric("Rotas únicas", dff["ROTA"].nunique())
k4.metric("Companhias", dff["COMP_NORM"].nunique())

# ---------------------------
# Regression ranking (smoothed and capped)
# ---------------------------
@st.cache_data(show_spinner=False)
def regression_ranking(df_input, min_points=6):
    groups = df_input.groupby(["ROTA","DATA"]).agg(tar_media=("TARIFA","mean")).reset_index()
    rows=[]
    for rota, grp in groups.groupby("ROTA"):
        g = grp.sort_values("DATA").copy()
        if len(g) < min_points:
            continue
        start = g["DATA"].min()
        g["t"] = ((g["DATA"].dt.year - start.year) * 12 + (g["DATA"].dt.month - start.month)).astype(int)
        X = g[["t"]].values
        y = g["tar_media"].values
        lr = LinearRegression().fit(X, y)
        # preds for 2026 months
        t_2026 = np.array([ (pd.Timestamp(2026,m,1).year - start.year)*12 + (pd.Timestamp(2026,m,1).month - start.month) for m in range(1,13) ])
        preds = lr.predict(t_2026.reshape(-1,1))
        # smoothing: blend with recent average
        recent = float(np.nanmean(y[-6:])) if len(y)>=6 else float(np.nanmean(y))
        preds_smooth = 0.7*preds + 0.3*recent
        mean_pred = float(np.nanmean(preds_smooth))
        # cap growth at +30%
        if recent > 0:
            mean_pred = min(mean_pred, recent * 1.30)
        mean_now = float(np.nanmean(y))
        pct_change = (mean_pred - mean_now)/mean_now if mean_now>0 else np.nan
        rows.append({"ROTA":rota, "pred_2026_mean":mean_pred, "current_mean":mean_now, "pct_change":pct_change, "slope":float(lr.coef_[0]), "n_obs":len(g)})
    res = pd.DataFrame(rows)
    if not res.empty:
        def sinal(r):
            pct = r["pct_change"]
            slope = r["slope"]
            if pd.isna(pct): return "Sem dados"
            if pct >= 0.30 or slope > 10: return "🛑 Forte alta"
            if pct >= 0.08: return "⚠️ Atenção"
            if pct < -0.05: return "📉 Queda"
            return "⚠️ Atenção"
        res["SINAL"] = res.apply(sinal, axis=1)
        res = res.sort_values("pred_2026_mean", ascending=False).reset_index(drop=True)
    return res

with st.spinner("Calculando ranking (suave)..."):
    rank_reg = regression_ranking(dff)

# ---------------------------
# Ranking table
# ---------------------------
st.markdown("---")
st.subheader("🏆 Ranking SR2 — Evite ou Aproveite (Previsão 2026)")

if rank_reg.empty:
    st.info("Sem rotas suficientes para ranking.")
else:
    display = rank_reg[["ROTA","current_mean","pred_2026_mean","pct_change","SINAL","n_obs"]].rename(columns={
        "current_mean":"Atual (R$)","pred_2026_mean":"Prev 2026 (R$)","pct_change":"Δ relativo","n_obs":"Obs"
    })
    st.dataframe(display.round(0))
    st.download_button("⬇️ Baixar ranking (CSV)", rank_reg.to_csv(index=False), file_name="ranking_sr2.csv", mime="text/csv")

# ---------------------------
# Map: Brasil only (open-street-map ensures 'Brasil' with S)
# ---------------------------
st.markdown("---")
st.subheader("🗺️ Mapa — Rotas em Alerta (Brasil)")

# coords for capitals (kept)
COORDS = {
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

fig = go.Figure()
if not rank_reg.empty:
    top_routes = rank_reg.head(top_n)["ROTA"].tolist()
    viz = rank_reg[rank_reg["ROTA"].isin(top_routes)].copy()
    viz[["O","D"]] = viz["ROTA"].apply(lambda r: pd.Series(parse_route(r)))
    for _, row in viz.dropna(subset=["O","D"]).iterrows():
        o,d = row["O"], row["D"]
        # try matching normalized strings (some city names may differ - best effort)
        if o in COORDS and d in COORDS:
            olat,olon = COORDS[o]; dlat,dlon = COORDS[d]
            color = PINK if row["SINAL"]=="🛑 Forte alta" else ORANGE if row["SINAL"]=="⚠️ Atenção" else GREEN
            width = 6 if row["SINAL"]=="🛑 Forte alta" else 3
            fig.add_trace(go.Scattermapbox(
                lat=[olat,dlat], lon=[olon,dlon],
                mode="lines+markers",
                line=dict(color=color,width=width),
                marker=dict(size=7,color=color),
                hoverinfo="text",
                hovertext=f"{row['ROTA']}<br>{row['SINAL']}<br>Prev 2026: R$ {row['pred_2026_mean']:.0f}"
            ))

fig.update_layout(
    mapbox_style="open-street-map",
    mapbox_center={"lat": -14.235004, "lon": -51.92528},
    mapbox_zoom=3.7,
    mapbox_bounds={"west": -74, "east": -34, "south": -34, "north": 6},
    height=540,
    margin=dict(l=0,r=0,t=0,b=0),
    title_text="Mapa — Brasil (rotas em alerta e oportunidades)"
)
st.plotly_chart(fig, use_container_width=True)

# ---------------------------
# Forecast widget (suave) Origem -> Destino
# ---------------------------
st.markdown("---")
st.header("🔮 Previsão Mensal 2026 — Origem → Destino (Suave)")

colA,colB,colC = st.columns([3,3,1])
orig = colA.selectbox("Origem", sorted(dff["ORIG"].unique()))
dest = colB.selectbox("Destino", sorted(dff["DEST"].unique()))
btn = colC.button("Gerar previsão")

def forecast_suave(route, df_all, use_prophet=True):
    sub = df_all[df_all["ROTA"]==route].groupby("DATA").agg(tar_media=("TARIFA","mean"), temp=("TEMP_MEDIA","mean")).reset_index().sort_values("DATA")
    if sub.shape[0] < 6:
        return None, "Histórico insuficiente (<6 meses)."
    sub = sub.copy()
    sub["t"] = np.arange(len(sub))
    lr = LinearRegression().fit(sub[["t"]], sub["tar_media"].values)
    t_vals = np.arange(len(sub), len(sub)+12)
    preds = lr.predict(t_vals.reshape(-1,1))
    recent = float(sub["tar_media"].rolling(3, min_periods=1).mean().iloc[-1])
    preds_soft = 0.65*preds + 0.35*recent
    preds_soft = np.minimum(preds_soft, recent * 1.30)  # cap +30%
    out = pd.DataFrame({"ds":[pd.Timestamp(2026,m,1) for m in range(1,13)], "yhat":preds_soft})
    # Prophet blend if available & enough data
    if use_prophet and HAS_PROPHET and sub.shape[0] >= 12:
        try:
            dfp = sub.rename(columns={"DATA":"ds","tar_media":"y"})
            m = Prophet(yearly_seasonality=True)
            if sub["temp"].notna().any():
                try: m.add_regressor("temp")
                except: pass
            m.fit(dfp)
            future = m.make_future_dataframe(periods=12, freq="MS")
            if "temp" in dfp.columns:
                future["temp"] = dfp["temp"].mean()
            fc = m.predict(future)
            fc2026 = fc[fc["ds"].dt.year==2026][["ds","yhat"]].rename(columns={"yhat":"yhat_prop"})
            out = out.merge(fc2026, on="ds", how="left")
            out["yhat_final"] = out["yhat_prop"].fillna(out["yhat"])
            out["yhat_final"] = out["yhat_final"].rolling(3, min_periods=1).mean()
        except Exception:
            out["yhat_final"] = out["yhat"]
    else:
        out["yhat_final"] = out["yhat"]
    out["Mes"] = out["ds"].dt.month.map(MES_NAME)
    out["Tarifa Prevista (R$)"] = out["yhat_final"].round(0)
    return out[["ds","Mes","Tarifa Prevista (R$)"]], None

if btn:
    rota_sel = f"{orig} → {dest}"
    with st.spinner("Gerando previsão suave..."):
        table, err = forecast_suave(rota_sel, dff, use_prophet=True)
    if err:
        st.warning(err)
    else:
        st.markdown(f"### Previsão mensal — 2026 — {rota_sel}")
        st.dataframe(table.assign(Data=lambda df: df["ds"].dt.strftime("%Y-%m-%d")).drop(columns=["ds"]).reset_index(drop=True))
        figp = px.line(table, x="Mes", y="Tarifa Prevista (R$)", markers=True, color_discrete_sequence=[PURPLE])
        figp.update_layout(yaxis_title="Tarifa média prevista (R$)", xaxis_title="Mês")
        st.plotly_chart(figp, use_container_width=True)
        st.download_button("⬇️ Exportar previsão (CSV)", table.to_csv(index=False), file_name=f"previsao_2026_{orig}_{dest}.csv", mime="text/csv")
        if not HAS_PROPHET:
            st.info("Nota: Prophet não instalado — usamos regressão suave como fallback.")

# ---------------------------
# Tarifas por Estação (novo gráfico solicitado)
# ---------------------------
st.markdown("---")
st.subheader("📈 Tarifas por Estação — Média e Dispersão")

# compute per-station stats
est_stats = dff.groupby("ESTACAO").agg(media=("TARIFA","mean"), mediana=("TARIFA","median"), std=("TARIFA","std"), n=("TARIFA","count")).reset_index()
# order seasons in a typical brazilian sequence
order = ["Verão","Outono","Inverno","Primavera"]
est_stats["ESTACAO"] = pd.Categorical(est_stats["ESTACAO"], categories=order, ordered=True)
est_stats = est_stats.sort_values("ESTACAO")

# bar + errorbars (mean +/- std)
fig_est = go.Figure()
fig_est.add_trace(go.Bar(
    x=est_stats["ESTACAO"].astype(str),
    y=est_stats["media"],
    name="Média (R$)",
    marker=dict(color=PURPLE),
    error_y=dict(type="data", array=est_stats["std"].fillna(0))
))
# add median as line
fig_est.add_trace(go.Scatter(x=est_stats["ESTACAO"].astype(str), y=est_stats["mediana"], mode="markers+lines", name="Mediana", marker=dict(color=ORANGE)))
fig_est.update_layout(yaxis_title="Tarifa (R$)", xaxis_title="Estação", title="Tarifa média por Estação (média ± desvio padrão)", height=420)
st.plotly_chart(fig_est, use_container_width=True)

# ---------------------------
# Sparklines: top destinos (more polished)
# ---------------------------
st.markdown("---")
st.subheader("✨ Sparklines — Top destinos (série histórica mensal)")

top_dests = dff.groupby("DEST").agg(avg=("TARIFA","mean")).reset_index().sort_values("avg", ascending=False).head(12)["DEST"].tolist()
sparks = dff[dff["DEST"].isin(top_dests)].groupby(["DEST","DATA"]).agg(m=("TARIFA","mean")).reset_index()
if not sparks.empty:
    fig_spark = px.line(sparks, x="DATA", y="m", color="DEST", facet_col="DEST", facet_col_wrap=4, height=560)
    fig_spark.update_layout(showlegend=False)
    fig_spark.for_each_annotation(lambda a: a.update(text=a.text.split("=")[-1]))
    fig_spark.update_traces(line=dict(width=2))
    st.plotly_chart(fig_spark, use_container_width=True)

# ---------------------------
# Insights — organized, clear, larger text
# ---------------------------
st.markdown("---")
st.subheader("💡 Insights SR2 — Sinais e Recomendações")

# build insights list (6)
insights = []

# 1 - maior alerta (alta)
top_up = rank_reg[rank_reg["SINAL"]=="🛑 Forte alta"].head(1)
if not top_up.empty:
    r = top_up.iloc[0]
    insights.append(("🛑 Maior Alerta (Alta)", f"{r['ROTA']} — Prev: R$ {r['pred_2026_mean']:.0f} (+{r['pct_change']:.0%})"))

# 2 - maior queda (oportunidade)
top_down = rank_reg[rank_reg["SINAL"]=="📉 Queda"].head(1)
if not top_down.empty:
    d = top_down.iloc[0]
    insights.append(("📉 Oportunidade (Queda)", f"{d['ROTA']} — queda prevista em 2026"))

# 3 - estação crítica
est_max = dff.groupby("ESTACAO").agg(m=("TARIFA","mean")).reset_index().sort_values("m", ascending=False)
if not est_max.empty:
    e = est_max.iloc[0]
    insights.append(("🌦 Estação Crítica", f"{e['ESTACAO']} — média: R$ {e['m']:.0f}"))

# 4 - companhia mais cara
comp_max = dff.groupby("COMP_NORM").agg(m=("TARIFA","mean")).reset_index().sort_values("m", ascending=False)
if not comp_max.empty:
    c = comp_max.iloc[0]
    insights.append(("✈️ Companhia Mais Cara", f"{c['COMP_NORM']} — média: R$ {c['m']:.0f}"))

# 5 - maior volatilidade (correctly read index)
vol = dff.groupby("ROTA").agg(std=("TARIFA","std"), mean=("TARIFA","mean")).dropna()
if not vol.empty:
    vol["cv"] = vol["std"]/vol["mean"].replace(0,np.nan)
    topv = vol.sort_values("cv", ascending=False).head(1)
    if not topv.empty:
        rota_nome = topv.index[0]
        v = topv.iloc[0]
        insights.append(("⚡ Maior Volatilidade", f"{rota_nome} — CV {v['cv']:.2f}"))

# 6 - mês mais caro
mth = dff.groupby("MES").agg(m=("TARIFA","mean")).reset_index().sort_values("m", ascending=False)
if not mth.empty:
    mm = mth.iloc[0]
    insights.append(("📅 Mês mais caro", f"{MES_NAME[int(mm['MES'])]} — média: R$ {mm['m']:.0f}"))

# render as nice cards 3 columns (2 rows)
cols = st.columns(3)
for i, (title, text) in enumerate(insights[:6]):
    with cols[i%3]:
        st.markdown(f"<div class='card'><h3 style='color:{PURPLE}; margin:0'>{title}</h3><div class='small' style='margin-top:6px'>{text}</div></div>", unsafe_allow_html=True)

# ---------------------------
# Footer
# ---------------------------
st.markdown("---")
st.caption("Bora Alí — SR2 • Visual refinado • Previsões suavizadas • Português (Brasil) — Boa apresentação!")

# =============================================================================
# END
# =============================================================================
