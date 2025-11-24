# =============================================================================
# Bora Alí — SR2 (Foco TOTAL: Rotas em Alerta + Previsões 2026)
# - Sem assets externos (tudo Plotly / Streamlit)
# - Entrada: INMET_ANAC_ROTAS_APENAS_CAPITAIS.csv na raiz
# - Modelo MISTO: regressão rápido + Prophet onde disponível (se instalado)
# =============================================================================

import os
import unicodedata
from datetime import datetime
import math
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

from sklearn.linear_model import LinearRegression

# Tentativa de importar Prophet — se não existir, siga com regressão apenas.
try:
    from prophet import Prophet
    HAS_PROPHET = True
except Exception:
    HAS_PROPHET = False

# ---------------------------
# Página & tema
# ---------------------------
st.set_page_config(page_title="Bora Alí — SR2 (Rotas em Alerta)", layout="wide", page_icon="🛑")
PURPLE = "#5A189A"
PINK_ALERT = "#E11D48"
ORANGE = "#FF6A00"
BG = "#FCF8FF"
TEXT = "#0F172A"

st.markdown(f"""
<style>
body {{ background-color: {BG}; color: {TEXT}; }}
h1,h2,h3,h4 {{ color: {PURPLE}; font-weight:800; }}
.stButton>button {{ background: {PURPLE}; color: white; border-radius:8px; padding:6px 10px; }}
</style>
""", unsafe_allow_html=True)

st.title("🛑 Bora Alí — SR2: Rotas em Alerta")
st.caption("Foco total: identificar rotas entre capitais com tendência de alta e apresentar previsões mensais 2026.")

# ---------------------------
# Arquivo CSV esperado
# ---------------------------
CSV_FILE = "INMET_ANAC_ROTAS_APENAS_CAPITAIS.csv"
if not os.path.exists(CSV_FILE):
    st.error(f"⛔ Arquivo não encontrado: {CSV_FILE}. Coloque o CSV na raiz do projeto e recarregue.")
    st.stop()

# ---------------------------
# Funções utilitárias
# ---------------------------
def normalize_str(s):
    if pd.isna(s): return s
    s = str(s)
    s = "".join(ch for ch in unicodedata.normalize("NFKD", s) if not unicodedata.combining(ch))
    s = s.replace("_"," ").replace("-"," ")
    return " ".join(s.split()).strip().title()

def parse_route(r):
    if pd.isna(r): return (None,None)
    s = str(r)
    for sep in ["→","->","-","/"]:
        if sep in s:
            p = [x.strip() for x in s.split(sep) if x.strip()]
            if len(p) >= 2:
                return (p[0], p[-1])
    return (None,None)

MES_NAME = {
    1:"Janeiro",2:"Fevereiro",3:"Março",4:"Abril",5:"Maio",6:"Junho",
    7:"Julho",8:"Agosto",9:"Setembro",10:"Outubro",11:"Novembro",12:"Dezembro"
}

def estacao_por_mes(m):
    if m in [12,1,2]: return "Verão"
    if m in [3,4,5]: return "Outono"
    if m in [6,7,8]: return "Inverno"
    return "Primavera"

# ---------------------------
# Carregar e preparar dados
# ---------------------------
@st.cache_data(show_spinner=False)
def load_prep(path):
    df = pd.read_csv(path, low_memory=False)
    df.columns = [c.upper().strip() for c in df.columns]
    # converter campos numéricos
    for c in ["TARIFA","TEMP_MEDIA","TEMP_MIN","TEMP_MAX"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    # tentar extrair origem/destino
    parsed = pd.DataFrame(df.get("ROTA", "").apply(lambda r: parse_route(r)).tolist(), columns=["_ORIG","_DEST"])
    df["ORIG"] = df.get("ORIGEM", parsed["_ORIG"]).fillna(parsed["_ORIG"]).apply(normalize_str)
    df["DEST"] = df.get("DESTINO", parsed["_DEST"]).fillna(parsed["_DEST"]).apply(normalize_str)
    # datas
    df["ANO"] = pd.to_numeric(df.get("ANO", pd.NA), errors="coerce").fillna(0).astype(int)
    df["MES"] = pd.to_numeric(df.get("MES", pd.NA), errors="coerce").fillna(0).astype(int)
    df = df[(df["ANO"]>0) & (df["MES"]>0)]
    df["DATA"] = pd.to_datetime(df["ANO"].astype(str) + "-" + df["MES"].astype(str).str.zfill(2) + "-01", errors="coerce")
    df = df.dropna(subset=["DATA","ORIG","DEST","TARIFA"])
    df["ROTA"] = df["ORIG"] + " → " + df["DEST"]
    df["MES_NOME"] = df["MES"].map(MES_NAME)
    df["ESTACAO"] = df["MES"].apply(estacao_por_mes)
    return df

df = load_prep(CSV_FILE)
st.sidebar.header("Filtros SR2")
anos = sorted(df["ANO"].unique())
sel_anos = st.sidebar.multiselect("Ano", anos, default=anos)
companias = sorted(df["COMPANHIA"].dropna().unique()) if "COMPANHIA" in df.columns else []
sel_comp = st.sidebar.multiselect("Companhia", companias, default=companias)
sel_est = st.sidebar.multiselect("Estação", ["Verão","Outono","Inverno","Primavera"], default=["Verão","Outono","Inverno","Primavera"])

dff = df[(df["ANO"].isin(sel_anos)) & (df["ESTACAO"].isin(sel_est))]
if "COMPANHIA" in df.columns and len(sel_comp)>0:
    dff = dff[dff["COMPANHIA"].isin(sel_comp)]

if dff.empty:
    st.error("⛔ Filtros resultaram em conjunto de dados vazio.")
    st.stop()

# ---------------------------
# KPIs
# ---------------------------
st.markdown("---")
c1,c2,c3,c4 = st.columns(4)
c1.metric("📊 Registros (filtrados)", f"{len(dff):,}")
c2.metric("💰 Tarifa média (filtrada)", f"R$ {dff['TARIFA'].mean():.0f}")
c3.metric("✈️ Rotas únicas", dff["ROTA"].nunique())
c4.metric("📅 Intervalo", f"{dff['ANO'].min()} → {dff['ANO'].max()}")

# ---------------------------
# Função: regressão por rota (rápida)
# ---------------------------
@st.cache_data(show_spinner=False)
def regression_by_route(df_input, min_points=6):
    grp = df_input.groupby(["ROTA","DATA"]).agg(tar_media=("TARIFA","mean")).reset_index()
    results = []
    for rota, g in grp.groupby("ROTA"):
        g = g.sort_values("DATA")
        if len(g) < min_points:
            continue
        start = g["DATA"].min()
        g = g.copy()
        g["t"] = ((g["DATA"].dt.year - start.year) * 12 + (g["DATA"].dt.month - start.month)).astype(int)
        X = g[["t"]].values
        y = g["tar_media"].values
        lr = LinearRegression().fit(X, y)
        # prev para 2026
        t_2026 = []
        for m in range(1,13):
            dt = pd.Timestamp(year=2026, month=m, day=1)
            t_val = (dt.year - start.year) * 12 + (dt.month - start.month)
            t_2026.append(t_val)
        preds = lr.predict(np.array(t_2026).reshape(-1,1))
        mean_pred = float(np.nanmean(preds))
        mean_now = float(np.nanmean(y))
        pct_change = (mean_pred-mean_now)/mean_now if mean_now>0 else np.nan
        results.append({
            "ROTA": rota,
            "pred_2026_mean": mean_pred,
            "current_mean": mean_now,
            "pct_change": pct_change,
            "slope": float(lr.coef_[0]),
            "n_obs": len(g)
        })
    res = pd.DataFrame(results)
    if not res.empty:
        def sinal(r):
            pct = r["pct_change"]
            slope = r["slope"]
            if pd.isna(pct): return "Sem dados"
            if pct >= 0.20 or slope > 5: return "🛑 Forte alta"
            if pct >= 0.05: return "⚠️ Atenção"
            if pct < 0: return "📉 Queda"
            return "⚠️ Atenção"
        res["SINAL"] = res.apply(sinal, axis=1)
        res = res.sort_values("pred_2026_mean", ascending=False).reset_index(drop=True)
    return res

with st.spinner("Calculando ranking rápido (regressão)..."):
    rank_reg = regression_by_route(dff)

st.markdown("---")
st.subheader("🏆 Ranking rápido — Previsão média 2026 (Regressão)")

if rank_reg.empty:
    st.info("Sem rotas suficientes para gerar ranking.")
else:
    st.dataframe(rank_reg[["ROTA","current_mean","pred_2026_mean","pct_change","SINAL","n_obs"]].rename(
        columns={"current_mean":"Atual (R$)","pred_2026_mean":"Prev 2026 (R$)","pct_change":"Δ relativo"}).round(0))

# botão para baixar ranking
if not rank_reg.empty:
    st.download_button("⬇️ Baixar Ranking SR2 (CSV)", rank_reg.to_csv(index=False), file_name="ranking_sr2.csv", mime="text/csv")

# ---------------------------
# Mapa: rotas em alerta (Plotly)
# ---------------------------
st.markdown("---")
st.subheader("🗺️ Mapa — Rotas em Alerta")

# coordenadas básicas para capitais (fallback)
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

fig_map = go.Figure()
if not rank_reg.empty:
    viz = rank_reg.copy()
    viz[["O","D"]] = viz["ROTA"].apply(lambda r: pd.Series(parse_route(r)))
    for _, row in viz.dropna(subset=["O","D"]).iterrows():
        o,d = row["O"], row["D"]
        if o in COORDS and d in COORDS:
            olat,olon = COORDS[o]
            dlat,dlon = COORDS[d]
            color = PINK_ALERT if row["SINAL"]=="🛑 Forte alta" else ORANGE if row["SINAL"]=="⚠️ Atenção" else "green"
            width = 6 if row["SINAL"]=="🛑 Forte alta" else 3 if row["SINAL"]=="⚠️ Atenção" else 1.5
            fig_map.add_trace(go.Scattermapbox(
                lat=[olat,dlat], lon=[olon,dlon],
                mode="lines+markers",
                line=dict(color=color,width=width),
                marker=dict(size=6),
                hoverinfo="text",
                text=f"{row['ROTA']} — Prev 2026: R$ {row['pred_2026_mean']:.0f} — {row['SINAL']}"
            ))

fig_map.update_layout(mapbox_style="carto-positron",
                      mapbox_center={"lat":-14.2,"lon":-51.9}, mapbox_zoom=3.1,
                      height=520, margin=dict(l=0,r=0,t=0,b=0))
st.plotly_chart(fig_map, use_container_width=True)

# ---------------------------
# Componente central: Origem -> Destino -> Previsão mensal 2026
# ---------------------------
st.markdown("---")
st.header("🔮 Previsão mensal 2026 — Origem → Destino")

colA,colB,colC = st.columns([3,3,1])
orig = colA.selectbox("Origem", sorted(dff["ORIG"].unique()))
dest = colB.selectbox("Destino", sorted(dff["DEST"].unique()))
rota_sel = f"{orig} → {dest}"
btn = colC.button("📈 Gerar previsão 2026")

def forecast_route(rota, df_all, use_prophet=True):
    sub = df_all[df_all["ROTA"]==rota].groupby("DATA").agg(tar_media=("TARIFA","mean"), temp=("TEMP_MEDIA","mean")).reset_index().sort_values("DATA")
    if sub.shape[0] < 6:
        return None, "Histórico insuficiente (mínimo 6 meses)."
    start = sub["DATA"].min()
    sub = sub.copy()
    sub["t"] = ((sub["DATA"].dt.year - start.year)*12 + (sub["DATA"].dt.month - start.month)).astype(int)
    X = sub[["t"]].values
    y = sub["tar_media"].values
    lr = LinearRegression().fit(X, y)
    # LR preds 2026
    t_vals = []
    for m in range(1,13):
        dt = pd.Timestamp(year=2026, month=m, day=1)
        t_val = (dt.year - start.year)*12 + (dt.month - start.month)
        t_vals.append(t_val)
    preds_lr = lr.predict(np.array(t_vals).reshape(-1,1))
    df_lr = pd.DataFrame({"ds":[pd.Timestamp(2026,m,1) for m in range(1,13)], "yhat_lr":preds_lr})
    # Prophet if available and enough data
    df_prop = None
    if use_prophet and HAS_PROPHET and sub.shape[0] >= 12:
        try:
            dfp = sub.rename(columns={"DATA":"ds","tar_media":"y"})
            m = Prophet(yearly_seasonality=True)
            if sub["temp"].notna().any():
                try:
                    m.add_regressor("temp")
                except Exception:
                    pass
            m.fit(dfp)
            future = m.make_future_dataframe(periods=12, freq="MS")
            if "temp" in dfp.columns:
                future["temp"] = dfp["temp"].mean()
            pred = m.predict(future)
            df_prop = pred[pred["ds"].dt.year==2026][["ds","yhat"]].rename(columns={"yhat":"yhat_prophet"})
        except Exception as e:
            df_prop = None
    # merge
    out = df_lr.copy()
    if df_prop is not None:
        out = out.merge(df_prop, on="ds", how="left")
        out["yhat_final"] = out["yhat_prophet"].fillna(out["yhat_lr"])
    else:
        out["yhat_final"] = out["yhat_lr"]
    out["Mes"] = out["ds"].dt.month.map(MES_NAME)
    out["Tarifa Prevista (R$)"] = out["yhat_final"].round(0)
    return out[["ds","Mes","Tarifa Prevista (R$)"]], None

if btn:
    with st.spinner("Gerando previsão (Misto — regressão + Prophet quando disponível)..."):
        table, err = forecast_route(rota_sel, dff, use_prophet=True)
    if err:
        st.warning(err)
    else:
        st.markdown(f"### Previsão mensal — 2026 — {rota_sel}")
        st.dataframe(table.assign(Data=lambda df: df["ds"].dt.strftime("%Y-%m-%d")).drop(columns=["ds"]).reset_index(drop=True))
        fig = px.line(table, x="Mes", y="Tarifa Prevista (R$)", markers=True, title=f"📈 Previsão 2026 — {rota_sel}")
        fig.update_layout(yaxis_title="Tarifa média prevista (R$)", xaxis_title="Mês")
        st.plotly_chart(fig, use_container_width=True)
        csv_out = table.to_csv(index=False, encoding="utf-8")
        st.download_button("⬇️ Exportar previsão (CSV)", csv_out, file_name=f"previsao_2026_{orig}_{dest}.csv", mime="text/csv")
        if not HAS_PROPHET:
            st.warning("Observação: Prophet não está instalado no ambiente. Use Prophet para previsões mais precisas (instale via 'pip install prophet'). O resultado acima usa regressão linear como fallback quando necessário.")

# ---------------------------
# Visual adicional: Heatmap (mês x rota top)
# ---------------------------
st.markdown("---")
st.subheader("🔥 Heatmap — Tarifas (Mês x Destino) — visão geral")

# construir pivot com média por mês/dest
hm = dff.groupby(["MES_NOME","DEST"]).agg(m=("TARIFA","mean")).reset_index()
# ordenar meses
order = ["Janeiro","Fevereiro","Março","Abril","Maio","Junho","Julho","Agosto","Setembro","Outubro","Novembro","Dezembro"]
hm["MES_NOME"] = pd.Categorical(hm["MES_NOME"], categories=order, ordered=True)
pv = hm.pivot(index="DEST", columns="MES_NOME", values="m").fillna(0)
fig_hm = px.imshow(pv, labels=dict(x="Mês", y="Destino", color="Tarifa média (R$)"), aspect="auto")
st.plotly_chart(fig_hm, use_container_width=True)

# ---------------------------
# Footer / nota
# ---------------------------
st.markdown("---")
st.caption("Bora Alí — SR2 • Misto (Regressão + Prophet quando disponível) • Interface sem assets externos — pronta para apresentação.")
