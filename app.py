# =============================================================================
# Bora Alí — SR2 (Versão Apresentação: visual jovem, previsões suaves, mapa Brasil)
# - Substitua completamente o app.py existente por este.
# - CSV esperado: INMET_ANAC_ROTAS_APENAS_CAPITAIS.csv na raiz do projeto
# - Modelo MISTO: Regressão suavizada + Prophet opcional (se instalado)
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

# tenta importar Prophet (opcional)
try:
    from prophet import Prophet
    HAS_PROPHET = True
except Exception:
    HAS_PROPHET = False

# -------------------------
# Configuração da página
# -------------------------
st.set_page_config(page_title="Bora Alí — SR2 (Rotas em Alerta)", layout="wide", page_icon="🛑")
# Paleta Bora Alí (ROXO)
PURPLE = "#5A189A"
PINK = "#E11D48"
ORANGE = "#FF6A00"
BG = "#FCF8FF"
TEXT = "#0F172A"

st.markdown(f"""
<style>
body {{ background-color: {BG}; color: {TEXT}; }}
h1,h2,h3,h4 {{ color: {PURPLE}; font-weight:800; }}
.stButton>button {{ background: {PURPLE}; color:white; border-radius:8px; padding:6px 12px; }}
.card {{ background: white; border-radius:12px; padding:12px; box-shadow: 0 2px 8px rgba(0,0,0,0.06); }}
.small {{ font-size:12px; color:#6b7280; }}
</style>
""", unsafe_allow_html=True)

st.title("🛑 Bora Alí — SR2: Rotas em Alerta")
st.caption("Interface jovem, prática e em português — previsões controladas e recomendações claras para 2026.")

# -------------------------
# Utilitários
# -------------------------
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

def normalize_company(s):
    if pd.isna(s): return "Outras"
    x = str(s).upper()
    if "LATAM" in x or "TAM" in x:
        return "LATAM"
    if "GOL" in x:
        return "GOL"
    if "AZUL" in x:
        return "AZUL"
    return s.title()

MES_NAME = {
    1:"Janeiro",2:"Fevereiro",3:"Março",4:"Abril",5:"Maio",6:"Junho",
    7:"Julho",8:"Agosto",9:"Setembro",10:"Outubro",11:"Novembro",12:"Dezembro"
}

def estacao_por_mes(m):
    if m in [12,1,2]: return "Verão"
    if m in [3,4,5]: return "Outono"
    if m in [6,7,8]: return "Inverno"
    return "Primavera"

# -------------------------
# Carregar e preparar dados
# -------------------------
CSV_FILE = "INMET_ANAC_ROTAS_APENAS_CAPITAIS.csv"
if not os.path.exists(CSV_FILE):
    st.error(f"⛔ Coloque o arquivo '{CSV_FILE}' na raiz do projeto e recarregue.")
    st.stop()

@st.cache_data(show_spinner=False)
def load_prep(path):
    df = pd.read_csv(path, low_memory=False)
    df.columns = [c.upper().strip() for c in df.columns]
    # converter numéricos
    for c in ["TARIFA","TEMP_MEDIA","TEMP_MIN","TEMP_MAX","ANO","MES"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    # parse rota
    parsed = pd.DataFrame(df.get("ROTA", "").apply(lambda r: parse_route(r)).tolist(), columns=["_ORIG","_DEST"])
    df["ORIG"] = df.get("ORIGEM", parsed["_ORIG"]).fillna(parsed["_ORIG"]).apply(normalize_str)
    df["DEST"] = df.get("DESTINO", parsed["_DEST"]).fillna(parsed["_DEST"]).apply(normalize_str)
    # datas
    df["ANO"] = pd.to_numeric(df.get("ANO", pd.NA), errors="coerce").fillna(0).astype(int)
    df["MES"] = pd.to_numeric(df.get("MES", pd.NA), errors="coerce").fillna(0).astype(int)
    df = df[(df["ANO"]>0) & (df["MES"]>0)]
    df["DATA"] = pd.to_datetime(df["ANO"].astype(str) + "-" + df["MES"].astype(str).str.zfill(2) + "-01", errors="coerce")
    df = df.dropna(subset=["DATA","ORIG","DEST","TARIFA"])
    df["ROTA"] = df["ORIG"].astype(str) + " → " + df["DEST"].astype(str)
    df["MES_NOME"] = df["MES"].map(MES_NAME)
    df["ESTACAO"] = df["MES"].apply(estacao_por_mes)
    # companhia
    if "COMPANHIA" in df.columns:
        df["COMP_NORM"] = df["COMPANHIA"].apply(normalize_company)
    else:
        df["COMP_NORM"] = "Outras"
    return df

df = load_prep(CSV_FILE)
if df.empty:
    st.error("⛔ Dados vazios após limpeza.")
    st.stop()

# -------------------------
# Sidebar — filtros (pt-BR)
# -------------------------
st.sidebar.header("Filtros — Bora Alí (SR2)")
anos = sorted(df["ANO"].unique())
sel_anos = st.sidebar.multiselect("Ano", anos, default=anos)
companias = sorted(df["COMP_NORM"].unique())
sel_comp = st.sidebar.multiselect("Companhia", companias, default=companias)
estacoes = ["Verão","Outono","Inverno","Primavera"]
sel_est = st.sidebar.multiselect("Estação", estacoes, default=estacoes)
top_n = st.sidebar.slider("Top N rotas (mapa/ranking)", 5, 25, 12)

dff = df[(df["ANO"].isin(sel_anos)) & (df["COMP_NORM"].isin(sel_comp)) & (df["ESTACAO"].isin(sel_est))]
if dff.empty:
    st.error("⛔ Sem registros com esses filtros.")
    st.stop()

# -------------------------
# KPIs
# -------------------------
st.markdown("---")
k1,k2,k3,k4 = st.columns(4)
k1.metric("Registros (filtros)", f"{len(dff):,}")
k2.metric("Tarifa média", f"R$ {dff['TARIFA'].mean():.0f}")
k3.metric("Rotas únicas", dff["ROTA"].nunique())
k4.metric("Companhias", len(dff["COMP_NORM"].unique()))

# -------------------------
# Função de ranking (regressão suavizada e cap)
# -------------------------
@st.cache_data(show_spinner=False)
def regression_rank(df_input, min_points=6):
    grp = df_input.groupby(["ROTA","DATA"]).agg(tar_media=("TARIFA","mean")).reset_index()
    rows = []
    for rota, g in grp.groupby("ROTA"):
        g = g.sort_values("DATA")
        if len(g) < min_points:
            continue
        start = g["DATA"].min()
        g = g.copy()
        g["t"] = ((g["DATA"].dt.year - start.year)*12 + (g["DATA"].dt.month - start.month)).astype(int)
        X = g[["t"]].values
        y = g["tar_media"].values
        lr = LinearRegression().fit(X,y)
        # prever 2026
        t_2026 = np.array([ (pd.Timestamp(2026,m,1).year - start.year)*12 + (pd.Timestamp(2026,m,1).month - start.month) for m in range(1,13) ])
        preds = lr.predict(t_2026.reshape(-1,1))
        # suavizar: blend com média recente
        recent_mean = float(np.nanmean(y[-6:])) if len(y)>=6 else float(np.nanmean(y))
        preds_smooth = 0.7*preds + 0.3*recent_mean
        mean_pred = float(np.nanmean(preds_smooth))
        # cap: não deixar crescer mais que +30% em relação ao mean_now
        mean_now = float(np.nanmean(y))
        if not np.isnan(mean_now) and mean_now>0:
            cap = mean_now * 1.30
            if mean_pred > cap:
                mean_pred = cap
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

with st.spinner("Calculando ranking (prévia 2026)..."):
    rank_reg = regression_rank(dff)

# -------------------------
# Ranking exibido (tabela)
# -------------------------
st.markdown("---")
st.subheader("🏆 Ranking — Evite essas rotas em 2026")
if rank_reg.empty:
    st.info("Sem rotas suficientes.")
else:
    display_cols = ["ROTA","current_mean","pred_2026_mean","pct_change","SINAL","n_obs"]
    st.dataframe(rank_reg[display_cols].rename(columns={
        "current_mean":"Atual (R$)","pred_2026_mean":"Prev 2026 (R$)","pct_change":"Δ relativo","n_obs":"Obs"
    }).round(0))

    st.download_button("⬇️ Baixar ranking (CSV)", rank_reg.to_csv(index=False), file_name="ranking_sr2.csv", mime="text/csv")

# -------------------------
# Mapa só do Brasil (centralizado e zoom adequado)
# -------------------------
st.markdown("---")
st.subheader("🗺️ Mapa — Rotas em Alerta (Brasil)")

# coordenadas das capitais (mantidas)
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
            color = PINK if row["SINAL"]=="🛑 Forte alta" else ORANGE if row["SINAL"]=="⚠️ Atenção" else "green"
            width = 6 if row["SINAL"]=="🛑 Forte alta" else 3 if row["SINAL"]=="⚠️ Atenção" else 1.5
            fig_map.add_trace(go.Scattermapbox(
                lat=[olat,dlat], lon=[olon,dlon],
                mode="lines+markers",
                line=dict(color=color,width=width),
                marker=dict(size=6),
                hoverinfo="text",
                text=f"{row['ROTA']} — Prev 2026: R$ {row['pred_2026_mean']:.0f} — {row['SINAL']}"
            ))
# map centered on Brasil; restrict zoom to avoid showing continents
fig_map.update_layout(mapbox_style="carto-positron",
                      mapbox_center={"lat":-14.235004,"lon":-51.92528}, mapbox_zoom=3.0,
                      height=520, margin=dict(l=0,r=0,t=0,b=0))
fig_map.update_layout(title_text="Mapa — Brasil (rotas em alerta)")
st.plotly_chart(fig_map, use_container_width=True)

# -------------------------
# Componente Origem -> Destino (interativo)
# -------------------------
st.markdown("---")
st.header("🔮 Previsão Mensal 2026 — Origem → Destino")

c1,c2,c3 = st.columns([3,3,1])
orig = c1.selectbox("Origem", sorted(dff["ORIG"].unique()))
dest = c2.selectbox("Destino", sorted(dff["DEST"].unique()))
btn = c3.button("📈 Gerar previsão")

def forecast_route_soft(rota, df_all, use_prophet=True):
    sub = df_all[df_all["ROTA"]==rota].groupby("DATA").agg(tar_media=("TARIFA","mean"), temp=("TEMP_MEDIA","mean")).reset_index().sort_values("DATA")
    if sub.shape[0] < 6:
        return None, "Histórico insuficiente (<6 meses)."
    start = sub["DATA"].min()
    sub = sub.copy()
    sub["t"] = ((sub["DATA"].dt.year - start.year)*12 + (sub["DATA"].dt.month - start.month)).astype(int)
    X = sub[["t"]].values
    y = sub["tar_media"].values
    lr = LinearRegression().fit(X,y)
    # LR preds
    t_vals = np.array([ (pd.Timestamp(2026,m,1).year - start.year)*12 + (pd.Timestamp(2026,m,1).month - start.month) for m in range(1,13) ])
    preds = lr.predict(t_vals.reshape(-1,1))
    # blend with recent moving mean to soften
    recent_mean = float(sub["tar_media"].rolling(window=3, min_periods=1).mean().iloc[-1])
    preds_soft = 0.65*preds + 0.35*recent_mean
    # cap growth at +30% relative to recent_mean
    cap = recent_mean * 1.30
    preds_soft = np.minimum(preds_soft, cap)
    df_lr = pd.DataFrame({"ds":[pd.Timestamp(2026,m,1) for m in range(1,13)], "yhat":preds_soft})
    # prophet fallback (smoothed) if available
    if use_prophet and HAS_PROPHET and sub.shape[0] >= 12:
        try:
            dfp = sub.rename(columns={"DATA":"ds","tar_media":"y"})
            m = Prophet(yearly_seasonality=True)
            if sub["temp"].notna().any():
                try: m.add_regressor("temp")
                except: pass
            m.fit(dfp)
            future = m.make_future_dataframe(periods=12,freq="MS")
            if "temp" in dfp.columns:
                future["temp"] = dfp["temp"].mean()
            fc = m.predict(future)
            fc_2026 = fc[fc["ds"].dt.year==2026][["ds","yhat"]].rename(columns={"yhat":"yhat_prophet"})
            # blend LR soft + prophet
            df_lr = df_lr.merge(fc_2026, on="ds", how="left")
            df_lr["yhat_final"] = df_lr["yhat_prophet"].fillna(df_lr["yhat"])
            # smoothing final
            df_lr["yhat_final"] = df_lr["yhat_final"].rolling(3, min_periods=1).mean()
        except Exception:
            df_lr["yhat_final"] = df_lr["yhat"]
    else:
        df_lr["yhat_final"] = df_lr["yhat"]
    df_lr["Mes"] = df_lr["ds"].dt.month.map(MES_NAME)
    df_lr["Tarifa Prevista (R$)"] = df_lr["yhat_final"].round(0)
    return df_lr[["ds","Mes","Tarifa Prevista (R$)"]], None

if btn:
    rota_sel = f"{orig} → {dest}"
    with st.spinner("Gerando previsão suave (Misto)..."):
        table, err = forecast_route_soft(rota_sel, dff, use_prophet=True)
    if err:
        st.warning(err)
    else:
        st.markdown(f"### Previsão mensal — 2026 — {rota_sel}")
        st.dataframe(table.assign(Data=lambda df: df["ds"].dt.strftime("%Y-%m-%d")).drop(columns=["ds"]).reset_index(drop=True))
        fig = px.line(table, x="Mes", y="Tarifa Prevista (R$)", markers=True, title=f"Previsão 2026 — {rota_sel}", color_discrete_sequence=[PURPLE])
        fig.update_layout(yaxis_title="Tarifa média prevista (R$)", xaxis_title="Mês")
        st.plotly_chart(fig, use_container_width=True)
        st.download_button("⬇️ Baixar previsão (CSV)", table.to_csv(index=False), file_name=f"previsao_2026_{orig}_{dest}.csv", mime="text/csv")
        if not HAS_PROPHET:
            st.info("Obs: Prophet não está instalado aqui — usamos previsão suave por regressão.")

# -------------------------
# Visuals substituindo heatmap
# - Sparklines (small multiples) para top destinos
# -------------------------
st.markdown("---")
st.subheader("✨ Sparklines mensais — Top destinos (visual jovem)")

# top destinations by avg tariff
top_dests = dff.groupby("DEST").agg(avg=("TARIFA","mean")).reset_index().sort_values("avg", ascending=False).head(12)["DEST"].tolist()
sparks = dff[dff["DEST"].isin(top_dests)].copy()
sparks_ts = sparks.groupby(["DEST","DATA"]).agg(m=("TARIFA","mean")).reset_index()

# build a combined figure with facets for sparklines (3 columns x 4 rows)
fig_sparks = px.line(sparks_ts, x="DATA", y="m", color="DEST", facet_col="DEST", facet_col_wrap=4, height=560)
fig_sparks.update_layout(showlegend=False)
fig_sparks.for_each_annotation(lambda a: a.update(text=a.text.split("=")[-1]))
fig_sparks.update_yaxes(matches=None)  # independent y to see shapes
fig_sparks.update_traces(line=dict(width=2))
st.plotly_chart(fig_sparks, use_container_width=True)

# -------------------------
# 6 insights SR2 (mais bonitos)
# -------------------------
st.markdown("---")
st.subheader("💡 Insights SR2 — 6 sinais rápidos")
ins_container = st.container()
cols = ins_container.columns(3)

ins = []
# insight 1: rota com maior alerta
if not rank_reg.empty:
    top_alert = rank_reg[rank_reg["SINAL"]=="🛑 Forte alta"].head(1)
    if not top_alert.empty:
        r = top_alert.iloc[0]
        ins.append(("🛑 Evite", f"{r['ROTA']} — média prevista: R$ {r['pred_2026_mean']:.0f} (+{r['pct_change']:.0%})"))
# 2: companhia mais cara
comp = dff.groupby("COMP_NORM").agg(m=("TARIFA","mean")).reset_index().sort_values("m", ascending=False).head(1)
if not comp.empty:
    ins.append(("✈️ Companhia", f"{comp.iloc[0]['COMP_NORM']} — média: R$ {comp.iloc[0]['m']:.0f}"))
# 3: mês mais caro
mth = dff.groupby("MES").agg(m=("TARIFA","mean")).reset_index().sort_values("m", ascending=False).head(1)
if not mth.empty:
    ins.append(("📅 Pico Mensal", f"{MES_NAME[int(mth.iloc[0]['MES'])]} — média: R$ {mth.iloc[0]['m']:.0f}"))
# 4: estação mais cara
est = dff.groupby("ESTACAO").agg(m=("TARIFA","mean")).reset_index().sort_values("m", ascending=False).head(1)
if not est.empty:
    ins.append(("🌦 Estação", f"{est.iloc[0]['ESTACAO']} — média: R$ {est.iloc[0]['m']:.0f}"))
# 5: rota volátil
vol = dff.groupby("ROTA").agg(std=("TARIFA","std"), mean=("TARIFA","mean")).dropna()
if not vol.empty:
    vol["cv"] = vol["std"]/vol["mean"].replace(0,np.nan)
    topv = vol.sort_values("cv", ascending=False).head(1)
    if not topv.empty:
        rv = topv.iloc[0]
        ins.append(("⚡ Volatilidade", f"{rv['ROTA']} — CV: {rv['cv']:.2f}"))
# 6: oportunidade (queda)
down = rank_reg[rank_reg["SINAL"]=="📉 Queda"].head(1)
if not down.empty:
    r = down.iloc[0]
    ins.append(("🎯 Oportunidade", f"{r['ROTA']} — média prevista: R$ {r['pred_2026_mean']:.0f}"))

# render 6 cards (two rows)
for i, item in enumerate(ins[:6]):
    title, text = item
    col = cols[i % 3]
    with col:
        st.markdown(f"<div class='card'><h4>{title}</h4><div class='small'>{text}</div></div>", unsafe_allow_html=True)

# -------------------------
# Logos e estações (mais imagens)
# -------------------------
st.markdown("---")
st.subheader("🎨 Visual — Companhias & Estações")

logo_cols = st.columns(4)
logos = {
    "LATAM":"https://i.imgur.com/3k2G2uK.png",
    "GOL":"https://i.imgur.com/J2Q6bQf.png",
    "AZUL":"https://i.imgur.com/0VqG0bI.png",
    "OUTRAS":"https://i.imgur.com/8yQj0wK.png"
}
i = 0
for k,u in logos.items():
    try:
        logo_cols[i].image(u, width=110)
        logo_cols[i].markdown(f"**{k}**")
    except:
        logo_cols[i].write(k)
    i+=1

st.markdown("Estações visuais")
st.image([
    "https://i.imgur.com/Yp7Gkcp.png",
    "https://i.imgur.com/z0o5TmU.png",
    "https://i.imgur.com/bxSdjMk.png",
    "https://i.imgur.com/1O2hm7x.png"
], width=140)

# -------------------------
# Comparação por companhia e estação (gráfico compacto)
# -------------------------
st.markdown("---")
st.subheader("📊 Comparação — Tarifa média por Companhia × Estação")
comp_est = dff.groupby(["COMP_NORM","ESTACAO"]).agg(m=("TARIFA","mean")).reset_index()
fig_ce = px.bar(comp_est, x="COMP_NORM", y="m", color="ESTACAO", barmode="group", labels={"m":"Tarifa média (R$)"},
                title="Tarifa média por Companhia e Estação")
fig_ce.update_layout(xaxis_title="Companhia", yaxis_title="Tarifa média (R$)")
st.plotly_chart(fig_ce, use_container_width=True)

# -------------------------
# Footer
# -------------------------
st.markdown("---")
st.caption("Bora Alí — SR2 • Visual jovem • Previsões suavizadas • Português (Brasil). Boa apresentação! ✈️💜")

