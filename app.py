# =============================================================================
# Bora Alí — SR2 (Versão Final para Apresentação)
# - Baseado no seu código (Misto: regressão + Prophet opcional)
# - Normalização de companhias (inclui LATAM), +6 insights, imagens, sem heatmap
# =============================================================================

import os
import unicodedata
from datetime import datetime
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.linear_model import LinearRegression

# try Prophet but keep fallback
try:
    from prophet import Prophet
    HAS_PROPHET = True
except Exception:
    HAS_PROPHET = False

# ---------------------------
# Config page / theme
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
.small-muted {{ color: #6b7280; font-size:12px }}
</style>
""", unsafe_allow_html=True)

st.title("🛑 Bora Alí — SR2: Rotas em Alerta (Versão Final)")
st.caption("Dashboard focado em decisão — previsões 2026, inspeções por companhia e recomendações práticas")

# ---------------------------
# Utils
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
                return (p[0].title(), p[-1].title())
    return (None,None)

# Company normalization (map common variants -> canonical)
def normalize_company(s):
    if pd.isna(s): return "OUTRAS"
    x = str(s).strip().upper()
    # common variants for LATAM
    if "LATAM" in x or "TAM" in x:
        return "LATAM"
    if "GOL" in x:
        return "GOL"
    if "AZUL" in x:
        return "AZUL"
    # common words meaning "other"
    if x in ["OUTRAS","OUTRA","OUTROS","OTRAS","OTRA"]:
        return "OUTRAS"
    return x.title()

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
# Load & prep (robusto)
# ---------------------------
CSV_FILE = "INMET_ANAC_ROTAS_APENAS_CAPITAIS.csv"
if not os.path.exists(CSV_FILE):
    st.error(f"⛔ Arquivo não encontrado: {CSV_FILE}. Coloque o CSV na raiz e recarregue.")
    st.stop()

@st.cache_data(show_spinner=False)
def load_prep(path):
    df = pd.read_csv(path, low_memory=False)
    # normalize column names
    df.columns = [c.upper().strip() for c in df.columns]
    # convert numeric columns if exist
    for c in ["TARIFA","TEMP_MEDIA","TEMP_MIN","TEMP_MAX","ANO","MES"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    # attempt to map ORIG / DEST
    parsed = pd.DataFrame(df.get("ROTA", "").apply(lambda r: parse_route(r)).tolist(), columns=["_ORIG","_DEST"])
    df["ORIG"] = df.get("ORIGEM", parsed["_ORIG"]).fillna(parsed["_ORIG"]).apply(normalize_str)
    df["DEST"] = df.get("DESTINO", parsed["_DEST"]).fillna(parsed["_DEST"]).apply(normalize_str)
    # safe: if still missing fill from 'DEST' or header variations
    if "DEST" not in df.columns and "DESTINO" in df.columns:
        df["DEST"] = df["DESTINO"].apply(normalize_str)
    # year/month to int and date
    df["ANO"] = pd.to_numeric(df.get("ANO", pd.NA), errors="coerce").fillna(0).astype(int)
    df["MES"] = pd.to_numeric(df.get("MES", pd.NA), errors="coerce").fillna(0).astype(int)
    df = df[(df["ANO"]>0) & (df["MES"]>0)]
    df["DATA"] = pd.to_datetime(df["ANO"].astype(str) + "-" + df["MES"].astype(str).str.zfill(2) + "-01", errors="coerce")
    df = df.dropna(subset=["DATA","ORIG","DEST","TARIFA"])
    # route canonical
    df["ROTA"] = df["ORIG"].astype(str) + " → " + df["DEST"].astype(str)
    df["MES_NOME"] = df["MES"].map(MES_NAME)
    df["ESTACAO"] = df["MES"].apply(estacao_por_mes)
    # normalize companies
    if "COMPANHIA" in df.columns:
        df["COMP_NORM"] = df["COMPANHIA"].apply(normalize_company)
    else:
        df["COMP_NORM"] = "OUTRAS"
    return df

df = load_prep(CSV_FILE)

# quick checks
if df.empty:
    st.error("⛔ Dataset vazio após tratamento.")
    st.stop()

# ---------------------------
# Sidebar filters
# ---------------------------
st.sidebar.header("🎛️ Filtros — SR2 (Pronto para apresentação)")
anos = sorted(df["ANO"].unique())
sel_anos = st.sidebar.multiselect("Ano", anos, default=anos)
companies = sorted(df["COMP_NORM"].dropna().unique())
sel_comp = st.sidebar.multiselect("Companhia", companies, default=companies)
sel_est = st.sidebar.multiselect("Estação", ["Verão","Outono","Inverno","Primavera"], default=["Verão","Outono","Inverno","Primavera"])
top_n = st.sidebar.slider("Top N rotas p/ foco (mapa/ranking)", min_value=5, max_value=30, value=12, step=1)

dff = df[(df["ANO"].isin(sel_anos)) & (df["COMP_NORM"].isin(sel_comp)) & (df["ESTACAO"].isin(sel_est))]

if dff.empty:
    st.error("⛔ Nenhum registro após os filtros selecionados.")
    st.stop()

# ---------------------------
# Top KPIs row
# ---------------------------
st.markdown("---")
k1,k2,k3,k4 = st.columns(4)
k1.metric("📊 Registros (filtros)", f"{len(dff):,}")
k2.metric("💰 Tarifa média", f"R$ {dff['TARIFA'].mean():.0f}")
k3.metric("✈️ Rotas únicas", dff["ROTA"].nunique())
k4.metric("🛫 Companhias", len(dff["COMP_NORM"].unique()))

# ---------------------------
# Regression ranking (smoothed thresholds)
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
        # predict 2026 but limit extrapolation: blend LR with last observed mean (80% lr + 20% last mean)
        t_2026 = []
        for m in range(1,13):
            dt = pd.Timestamp(year=2026, month=m, day=1)
            t_val = (dt.year - start.year) * 12 + (dt.month - start.month)
            t_2026.append(t_val)
        preds = lr.predict(np.array(t_2026).reshape(-1,1))
        # smoothing: shrink extreme slopes by blending with linear rolling mean
        last_mean = float(np.nanmean(y[-6:])) if len(y)>=6 else float(np.nanmean(y))
        blended = 0.85*preds + 0.15*last_mean
        mean_pred = float(np.nanmean(blended))
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
            # milder thresholds to avoid false positives
            if pct >= 0.25 or slope > 8: return "🛑 Forte alta"
            if pct >= 0.08: return "⚠️ Atenção"
            if pct < -0.05: return "📉 Queda"
            return "⚠️ Atenção"
        res["SINAL"] = res.apply(sinal, axis=1)
        res = res.sort_values("pred_2026_mean", ascending=False).reset_index(drop=True)
    return res

with st.spinner("⏳ Calculando sinais e ranking (regressão)..."):
    rank_reg = regression_by_route(dff)

st.markdown("---")
st.subheader("🏆 Ranking SR2 — Evite essas rotas em 2026 (Regressão suavizada)")
if rank_reg.empty:
    st.info("Sem rotas suficientes para ranking.")
else:
    show_cols = ["ROTA","current_mean","pred_2026_mean","pct_change","SINAL","n_obs"]
    st.dataframe(rank_reg[show_cols].rename(columns={
        "current_mean":"Atual (R$)","pred_2026_mean":"Prev 2026 (R$)","pct_change":"Δ relativo","n_obs":"Obs"
    }).round(0))

# download ranking
if not rank_reg.empty:
    st.download_button("⬇️ Baixar Ranking SR2 (CSV)", rank_reg.to_csv(index=False), file_name="ranking_sr2.csv", mime="text/csv")

# ---------------------------
# Map: top routes visual
# ---------------------------
st.markdown("---")
st.subheader("🗺️ Mapa de Rotas — foco nas Top N (alerta visual)")

# basic coords (kept from your list)
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
    top_routes = rank_reg.head(top_n)["ROTA"].tolist()
    viz = rank_reg[rank_reg["ROTA"].isin(top_routes)].copy()
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
# Central: Origem->Destino Forecast (misto)
# ---------------------------
st.markdown("---")
st.header("🔮 Previsão Mensal 2026 — Origem → Destino (Selecione)")

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
    # shrink extremes: blend with last observed moving avg (20% weight)
    last_mean = float(sub["tar_media"].rolling(3, min_periods=1).mean().iloc[-1])
    preds_lr = 0.85*preds_lr + 0.15*last_mean
    df_lr = pd.DataFrame({"ds":[pd.Timestamp(2026,m,1) for m in range(1,13)], "yhat_lr":preds_lr})
    # Prophet if available and long history
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
            # mild smoothing: rolling mean
            df_prop["yhat_prophet"] = df_prop["yhat_prophet"].rolling(3, min_periods=1).mean()
        except Exception:
            df_prop = None
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
    with st.spinner("Gerando previsão (misto)..."):
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
            st.warning("Observação: Prophet não está instalado. Resultado misto usa regressão suavizada.")

# ---------------------------
# Substituição do heatmap -> Boxplot mensal por destino (menos agressivo)
# ---------------------------
st.markdown("---")
st.subheader("📦 Distribuição Mensal — Boxplot (Mês x Tarifa) — insights por destino/top rotas")

# select top destinations by avg price to show clearer boxes
top_dests = dff.groupby("DEST").agg(avg_t=("TARIFA","mean")).reset_index().sort_values("avg_t", ascending=False).head(12)["DEST"].tolist()
box_df = dff[dff["DEST"].isin(top_dests)].copy()
box_df["MES_NOME"] = pd.Categorical(box_df["MES_NOME"], categories=list(MES_NAME.values()), ordered=True)

fig_box = px.box(box_df, x="MES_NOME", y="TARIFA", color="MES_NOME",
                 category_orders={"MES_NOME": list(MES_NAME.values())},
                 title="Distribuição de Tarifas por Mês (Top destinos)")
fig_box.update_layout(showlegend=False, yaxis_title="Tarifa (R$)", xaxis_title="Mês")
st.plotly_chart(fig_box, use_container_width=True)

# ---------------------------
# 6 insights automáticos (SR2) — prioridades
# ---------------------------
st.markdown("---")
st.subheader("💡 Insights Automáticos — SR2 (6 sinais)")

insights = []
# 1 — rota com maior aumento previsto (from rank_reg)
if not rank_reg.empty:
    top_up = rank_reg[rank_reg["SINAL"]=="🛑 Forte alta"].head(1)
    if not top_up.empty:
        r = top_up.iloc[0]
        insights.append(("🛑 Rota com maior risco", f"{r['ROTA']} — previsão média 2026: R$ {r['pred_2026_mean']:.0f} (+{r['pct_change']:.0%})"))
# 2 — rota com maior queda
if not rank_reg.empty:
    down = rank_reg[rank_reg["SINAL"]=="📉 Queda"].head(1)
    if not down.empty:
        r = down.iloc[0]
        insights.append(("📉 Rota com queda", f"{r['ROTA']} — previsão média 2026: R$ {r['pred_2026_mean']:.0f}"))
# 3 — mês com maior média histórica (across dff)
month_avg = dff.groupby("MES").agg(m=("TARIFA","mean")).reset_index()
if not month_avg.empty:
    top_month = month_avg.sort_values("m",ascending=False).iloc[0]
    insights.append(("📅 Mês mais caro (hist.)", f"{MES_NAME[int(top_month['MES'])]} — Tarifa média: R$ {top_month['m']:.0f}"))
# 4 — companhia mais cara (média)
comp_avg = dff.groupby("COMP_NORM").agg(m=("TARIFA","mean")).reset_index().sort_values("m", ascending=False)
if not comp_avg.empty:
    top_comp = comp_avg.iloc[0]
    insights.append(("✈️ Companhia mais cara", f"{top_comp['COMP_NORM']} — Tarifa média: R$ {top_comp['m']:.0f}"))
# 5 — estação com pico (hist)
est_avg = dff.groupby("ESTACAO").agg(m=("TARIFA","mean")).reset_index().sort_values("m", ascending=False)
if not est_avg.empty:
    top_est = est_avg.iloc[0]
    insights.append(("🌦 Estação com maior preço", f"{top_est['ESTACAO']} — Tarifa média: R$ {top_est['m']:.0f}"))
# 6 — rota mais volátil (std)
vol = dff.groupby("ROTA").agg(std=("TARIFA","std"), mean=("TARIFA","mean")).reset_index().dropna()
if not vol.empty:
    vol["cv"] = vol["std"]/vol["mean"].replace(0,np.nan)
    most_vol = vol.sort_values("cv", ascending=False).head(1).iloc[0]
    insights.append(("⚡ Rota mais volátil", f"{most_vol['ROTA']} — CV: {most_vol['cv']:.2f}"))

# render insights (up to 6)
for title, text in insights[:6]:
    with st.container():
        st.markdown(f"**{title}** — {text}")

# ---------------------------
# Visuals: Company logos + Station images (inline via URLs)
# ---------------------------
st.markdown("---")
st.subheader("🎨 Visual — Companhias e Estações")

cols = st.columns([1,1,1,1])
logos = {
    "LATAM":"https://i.imgur.com/3k2G2uK.png",
    "GOL":"https://i.imgur.com/J2Q6bQf.png",
    "AZUL":"https://i.imgur.com/0VqG0bI.png",
    "OUTRAS":"https://i.imgur.com/8yQj0wK.png"
}
i=0
for comp, url in logos.items():
    try:
        cols[i].image(url, width=100)
        cols[i].markdown(f"**{comp}**")
    except Exception:
        cols[i].write(comp)
    i+=1

# station images
st.markdown("**Estações (visual)**")
st.image([
    "https://i.imgur.com/Yp7Gkcp.png",
    "https://i.imgur.com/z0o5TmU.png",
    "https://i.imgur.com/bxSdjMk.png",
    "https://i.imgur.com/1O2hm7x.png"
], width=150)

# ---------------------------
# Extra: comparação por companhia + estação (small multiples)
# ---------------------------
st.markdown("---")
st.subheader("📊 Comparação — Tarifa média por Companhia e Estação")

comp_est = dff.groupby(["COMP_NORM","ESTACAO"]).agg(m=("TARIFA","mean")).reset_index()
fig_comp_est = px.bar(comp_est, x="COMP_NORM", y="m", color="ESTACAO", barmode="group",
                      title="Tarifa média por Companhia e Estação")
fig_comp_est.update_layout(yaxis_title="Tarifa média (R$)", xaxis_title="Companhia")
st.plotly_chart(fig_comp_est, use_container_width=True)

# ---------------------------
# Footer / export
# ---------------------------
st.markdown("---")
st.caption("Bora Alí — SR2 • Versão final para apresentação. Técnicas: Regressão suavizada + Prophet opcional. Bom show! 🎤")

