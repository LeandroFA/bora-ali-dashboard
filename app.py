# =============================================================================
# Bora Alí — SR2 ROXO (Foco TOTAL em Rotas em Alerta + Previsões 2026)
# - Use: INMET_ANAC_ROTAS_APENAS_CAPITAIS.csv na raiz
# - Misto: Regressão para ranking + Prophet nas top rotas
# - Componente: Escolha ORIGEM & DESTINO -> Previsões mensais 2026 (tabela + gráfico + CSV)
# - Imagens/assets usadas (caminhos locais enviados pelo usuário)
# =============================================================================

import os
import unicodedata
import math
from datetime import datetime
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

from prophet import Prophet
from sklearn.linear_model import LinearRegression

# ---------------------------------------------
# CONFIGURAÇÃO DA PÁGINA — Tema ROXO Bora Alí
# ---------------------------------------------
st.set_page_config(page_title="Bora Alí — SR2 (Rotas em Alerta)", layout="wide", page_icon="🛑")

# Paleta ROXO (SR2)
PURPLE = "#5A189A"
PINK_ALERT = "#E11D48"
ORANGE = "#FF6A00"
GRAFITE = "#1E1E1E"
BG = "#FCF8FF"
TEXT = "#0F172A"

st.markdown(f"""
<style>
body {{ background-color: {BG}; color: {TEXT}; }}
h1,h2,h3,h4 {{ color: {PURPLE}; font-weight:800; }}
.stButton>button {{ background: {PURPLE}; color: white; border-radius:8px; padding:6px 10px; }}
.reportview-container .main footer {{visibility: hidden;}}
</style>
""", unsafe_allow_html=True)

st.title("🛑 Bora Alí — SR2: Rotas em Alerta (Foco total)")
st.caption("Identifique rotas entre capitais com tendência de alta nas tarifas e evite surpresas em 2026. Tema: ROXO • Bora Alí")

# ---------------------------------------------
# ASSETS (usei os arquivos que você fez upload)
# Se quiser trocar, altere as variáveis abaixo.
# ---------------------------------------------
ASSET_PDF_1 = "/mnt/data/Bora Alí — Capitais · Streamlit.pdf"
ASSET_PDF_2 = "/mnt/data/Bora Alí — Dashboard (Capitais) · Streamlit.pdf"
ASSET_PDF_3 = "/mnt/data/Untitled17.ipynb - Colab.pdf"
ASSET_PDF_4 = "/mnt/data/Processos de Acompanhamento - BORA ALÍ (Cronograma) - Página1 (1).pdf"
ASSET_PDF_5 = "/mnt/data/BORA ALÍ - SR1 (1).pdf"

# Mostrar alguns assets visuais (muitos! foco visual SR2)
with st.expander("📚 Materiais do projeto (clique para ver) — Imagens/PDFs"):
    st.markdown("**Painéis & documentação** — use como referência visual para apresentação SR2.")
    # Tentamos mostrar via tag <img>. Se não renderizar, aparece como link para download.
    for p in [ASSET_PDF_1, ASSET_PDF_2, ASSET_PDF_5]:
        if os.path.exists(p):
            st.markdown(f'<div style="margin-bottom:8px"><a href="file://{p}" target="_blank">📎 Abrir {os.path.basename(p)}</a></div>', unsafe_allow_html=True)
        else:
            st.write(f"Arquivo não encontrado: {p}")

# ---------------------------------------------
# PATH CSV
# ---------------------------------------------
CSV_FILE = "INMET_ANAC_ROTAS_APENAS_CAPITAIS.csv"

# ---------------------------------------------
# FUNÇÕES AUXILIARES
# ---------------------------------------------
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
            p=[x.strip() for x in s.split(sep)]
            if len(p)>=2: return (p[0],p[-1])
    return (None,None)

# map months to pt_BR names (manual para evitar dependência de locale)
MES_NAME = {1:"Janeiro",2:"Fevereiro",3:"Março",4:"Abril",5:"Maio",6:"Junho",
            7:"Julho",8:"Agosto",9:"Setembro",10:"Outubro",11:"Novembro",12:"Dezembro"}

# ---------------------------------------------
# CARREGAR CSV + TRATAMENTO (cacheado)
# ---------------------------------------------
@st.cache_data(show_spinner=False)
def load_and_prep(path):
    if not os.path.exists(path):
        st.error(f"⛔ CSV NÃO ENCONTRADO: {path} — coloque o arquivo na raiz e recarregue.")
        st.stop()
    df = pd.read_csv(path, low_memory=False)
    # padronizar colunas
    df.columns = [c.upper().strip() for c in df.columns]
    for c in ["TARIFA","TEMP_MEDIA","TEMP_MIN","TEMP_MAX"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df.get(c), errors="coerce")
    # parse rota / origem/destino
    parsed = pd.DataFrame(df.get("ROTA", "").apply(lambda r: parse_route(r)).tolist(), columns=["_ORIG","_DEST"])
    df["ORIG"] = df.get("ORIGEM", parsed["_ORIG"]).fillna(parsed["_ORIG"]).apply(normalize_str)
    df["DEST"] = df.get("DESTINO", parsed["_DEST"]).fillna(parsed["_DEST"]).apply(normalize_str)
    # Datas
    df["ANO"] = pd.to_numeric(df.get("ANO", pd.NA), errors="coerce").fillna(0).astype(int)
    df["MES"] = pd.to_numeric(df.get("MES", pd.NA), errors="coerce").fillna(0).astype(int)
    df = df[(df["ANO"]>0) & (df["MES"]>0)]
    df["DATA"] = pd.to_datetime(df["ANO"].astype(str) + "-" + df["MES"].astype(str).str.zfill(2) + "-01", errors="coerce")
    df = df.dropna(subset=["DATA","ORIG","DEST","TARIFA"])
    df["ROTA"] = df["ORIG"] + " → " + df["DEST"]
    df["MES_NOME"] = df["MES"].map(MES_NAME)
    # Estações
    def est(m):
        if m in [12,1,2]: return "Verão"
        if m in [3,4,5]: return "Outono"
        if m in [6,7,8]: return "Inverno"
        return "Primavera"
    df["ESTACAO"] = df["MES"].apply(est)
    return df

df = load_and_prep(CSV_FILE)

# reduzir ao conjunto de capitais que aparecem no dataset (segurança)
CAPITAIS = sorted(list(pd.unique(df["ORIG"].tolist() + df["DEST"].tolist())))
if not CAPITAIS:
    st.error("⛔ Não foram encontradas capitais no dataset após tratamento.")
    st.stop()

# ---------------------------------------------
# SIDEBAR — Filtros SR2 (foco total)
# ---------------------------------------------
st.sidebar.header("🎛️ Filtros — Foco SR2 (Rotas em Alerta)")
ano_min, ano_max = int(df["ANO"].min()), int(df["ANO"].max())
sel_anos = st.sidebar.multiselect("Ano (filtrar histórico)", sorted(df["ANO"].unique()), default=sorted(df["ANO"].unique()))
sel_comp = st.sidebar.multiselect("Companhia (opcional)", sorted(df["COMPANHIA"].dropna().unique()), default=sorted(df["COMPANHIA"].dropna().unique()))
sel_est = st.sidebar.multiselect("Estação", ["Verão","Outono","Inverno","Primavera"], default=["Verão","Outono","Inverno","Primavera"])

dff = df[(df["ANO"].isin(sel_anos)) & (df["COMPANHIA"].isin(sel_comp)) & (df["ESTACAO"].isin(sel_est))]

if dff.empty:
    st.error("⛔ Nenhum registro após filtros.")
    st.stop()

# ---------------------------------------------
# KPI Compactos
# ---------------------------------------------
st.markdown("---")
k1,k2,k3,k4 = st.columns(4)
k1.metric("📊 Registros (filtros)", f"{len(dff):,}")
k2.metric("💰 Tarifa média", f"R$ {dff['TARIFA'].mean():.0f}")
k3.metric("✈️ Rotas únicas", dff["ROTA"].nunique())
k4.metric("📅 Período", f"{dff['ANO'].min()} → {dff['ANO'].max()}")

# ---------------------------------------------
# 1) CÁLCULO RÁPIDO: Regressão linear por ROTA (rápida para ranking)
#    - Para todas as rotas calculamos previsão média 2026 via regressão linear
#    - Depois rodamos Prophet apenas nas TOP rotas (MISTO)
# ---------------------------------------------
st.markdown("---")
st.subheader("🔎 Processamento SR2 — Ranking rápido (regressão)")

@st.cache_data(show_spinner=False)
def compute_regression_rank(df_input):
    # cria série temporal mensal média por rota
    grp = df_input.groupby(["ROTA","DATA"]).agg(tar_media=("TARIFA","mean")).reset_index()
    results = []
    for rota, g in grp.groupby("ROTA"):
        g_sorted = g.sort_values("DATA")
        # mínimo de pontos para regressão (mês a mês)
        if len(g_sorted) < 6:
            continue
        # transformar DATA em índice numérico (months since start)
        start = g_sorted["DATA"].min()
        g_sorted = g_sorted.copy()
        g_sorted["t"] = ((g_sorted["DATA"].dt.year - start.year) * 12 + (g_sorted["DATA"].dt.month - start.month)).astype(int)
        X = g_sorted[["t"]].values
        y = g_sorted["tar_media"].values
        model = LinearRegression()
        model.fit(X, y)
        slope = float(model.coef_[0])
        intercept = float(model.intercept_)
        # prever 12 meses de 2026: calcular t para Jan/2026..Dec/2026
        # compute t for each month in 2026 relative to start
        t_2026 = []
        for m in range(1,13):
            dt = pd.Timestamp(year=2026, month=m, day=1)
            t_val = (dt.year - start.year) * 12 + (dt.month - start.month)
            t_2026.append(t_val)
        preds_2026 = model.predict(np.array(t_2026).reshape(-1,1))
        pred_2026_mean = float(np.nanmean(preds_2026))
        current_mean = float(np.nanmean(y))
        pct_change = (pred_2026_mean - current_mean)/current_mean if current_mean>0 else np.nan
        results.append({
            "ROTA": rota,
            "slope": slope,
            "pred_2026_mean": pred_2026_mean,
            "current_mean": current_mean,
            "pct_change": pct_change,
            "n_obs": len(g_sorted)
        })
    res_df = pd.DataFrame(results)
    # classificar alertas
    def label_row(r):
        s = r["slope"]
        pct = r["pct_change"]
        # thresholds empíricos — ajuste se quiser
        if pd.isna(pct): return "Sem dados"
        if pct >= 0.20 or s > 5: return "🛑 Forte alta"
        if pct >= 0.05 and pct < 0.20: return "⚠️ Atenção"
        if pct < 0.0: return "📉 Queda"
        return "⚠️ Atenção"
    if not res_df.empty:
        res_df["SINAL"] = res_df.apply(label_row, axis=1)
        res_df = res_df.sort_values("pred_2026_mean", ascending=False).reset_index(drop=True)
    return res_df

rank_reg = compute_regression_rank(dff)

st.write("Resumo rápido — ranking (regressão linear): as rotas com maior tarifa prevista para 2026")
if rank_reg.empty:
    st.info("Sem rotas com histórico suficiente para regressão.")
else:
    st.dataframe(rank_reg[["ROTA","current_mean","pred_2026_mean","pct_change","SINAL","n_obs"]].rename(
        columns={"current_mean":"Média Atual (R$)","pred_2026_mean":"Média Prevista 2026 (R$)","pct_change":"Δ relativo"}).round(0))

# ---------------------------------------------
# 2) Selecionar TOP N rotas para ajuste fino com Prophet
# ---------------------------------------------
st.markdown("---")
st.subheader("🔧 Ajuste Fino (Prophet) nas rotas mais relevantes — MISTO")

TOP_N = st.number_input("Quantas rotas processar com Prophet (mais detalhado)?", min_value=3, max_value=30, value=10, step=1)
run_prophet = st.button("🔮 Rodar Prophet nas Top rotas (processo mais lento)")

prophet_results = {}
if run_prophet and not rank_reg.empty:
    with st.spinner("Rodando Prophet nas top rotas... (pode demorar)"):
        top_routes = rank_reg.head(int(TOP_N))["ROTA"].tolist()
        for rota in top_routes:
            sub = dff[dff["ROTA"]==rota].groupby("DATA").agg(tar_media=("TARIFA","mean"), temp=("TEMP_MEDIA","mean")).reset_index().sort_values("DATA")
            if len(sub) < 12:
                continue
            dfp = sub.rename(columns={"DATA":"ds","tar_media":"y","temp":"temp"})
            m = Prophet(yearly_seasonality=True)
            # adicionar regressor somente se temp existir
            if "temp" in dfp.columns and dfp["temp"].notna().any():
                try:
                    m.add_regressor("temp")
                except Exception:
                    pass
            m.fit(dfp)
            future = m.make_future_dataframe(periods=12,freq="MS")
            # preencher temp no futuro com média histórica da rota (simples)
            if "temp" in dfp.columns:
                future["temp"] = dfp["temp"].mean()
            fc = m.predict(future)
            fc_2026 = fc[fc["ds"].dt.year==2026][["ds","yhat"]].copy()
            prophet_results[rota] = fc_2026
    st.success("Prophet processado nas rotas selecionadas.")

# ---------------------------------------------
# 3) MAPA DE ROTAS — SINAL DE ALERTA
# ---------------------------------------------
st.markdown("---")
st.subheader("🗺️ Mapa — Rotas em Alerta (visual)")

# Simplified map: linhas entre capitais usando mediana das coordenadas do dataset (ou dicionário se quiser)
# Aqui usamos uma lista simplificada de coordenadas internas (se desejar, substitua pelo seu dicionário)
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

# Prepara dados de rotas com sinal
if not rank_reg.empty:
    viz = rank_reg.copy()
    # extrai origem/destino para plot
    viz[["O","D"]] = viz["ROTA"].apply(lambda r: pd.Series(parse_route(r)))
    viz = viz.dropna(subset=["O","D"])
    # build map figure
    fig_map = go.Figure()
    for _, r in viz.iterrows():
        o = r["O"]
        d = r["D"]
        if o not in COORDS or d not in COORDS:
            continue
        olat, olon = COORDS[o]
        dlat, dlon = COORDS[d]
        # linha color conforme sinal
        col = PINK_ALERT if r["SINAL"]=="🛑 Forte alta" else ORANGE if r["SINAL"]=="⚠️ Atenção" else "green"
        width = 6 if r["SINAL"]=="🛑 Forte alta" else 3 if r["SINAL"]=="⚠️ Atenção" else 1.5
        fig_map.add_trace(go.Scattermapbox(
            lat=[olat,dlat], lon=[olon,dlon],
            mode="lines+markers",
            line=dict(width=width, color=col),
            marker=dict(size=6),
            hoverinfo="text",
            text=f"{r['ROTA']} — Prev 2026: R$ {r['pred_2026_mean']:.0f} — {r['SINAL']}"
        ))
    fig_map.update_layout(
        mapbox_style="carto-positron",
        mapbox_center={"lat":-14.2,"lon":-51.9},
        mapbox_zoom=3.1,
        height=520,
        margin=dict(l=0,r=0,t=0,b=0)
    )
    st.plotly_chart(fig_map, use_container_width=True)
else:
    st.info("Sem rotas a plotar no mapa (dados insuficientes).")

# ---------------------------------------------
# 4) COMPONENTE CENTRAL: ORIGEM → DESTINO → Previsões mensais 2026
# ---------------------------------------------
st.markdown("---")
st.header("🔮 Previsão mensal 2026 — escolha Origem e Destino")

col1, col2, col3 = st.columns([3,3,2])
with col1:
    origem = st.selectbox("Origem", sorted(dff["ORIG"].unique()), index=0)
with col2:
    destino = st.selectbox("Destino", sorted(dff["DEST"].unique()), index=1)
with col3:
    btn_pred = st.button("📈 Gerar previsão 2026 para essa rota")

rota_sel = f"{origem} → {destino}"

def forecast_route(rota, df_all, use_prophet_if_possible=True):
    # agrupa por DATA e gera média e temp média (se disponível)
    sub = df_all[df_all["ROTA"]==rota].groupby("DATA").agg(tar_media=("TARIFA","mean"), temp=("TEMP_MEDIA","mean")).reset_index().sort_values("DATA")
    if sub.shape[0] < 6:
        return None, "Histórico insuficiente (mínimo 6 meses) para esta rota."
    # 1) Regressão linear simples para previsão rápida (apenas como fallback/benchmark)
    start = sub["DATA"].min()
    sub = sub.copy()
    sub["t"] = ((sub["DATA"].dt.year - start.year) * 12 + (sub["DATA"].dt.month - start.month)).astype(int)
    X = sub[["t"]].values
    y = sub["tar_media"].values
    lr = LinearRegression().fit(X,y)
    # previsão média 2026 via regressão (12 meses)
    t_2026 = []
    for m in range(1,13):
        dt = pd.Timestamp(year=2026, month=m, day=1)
        t_val = (dt.year - start.year) * 12 + (dt.month - start.month)
        t_2026.append(t_val)
    preds_lr = lr.predict(np.array(t_2026).reshape(-1,1))
    df_lr_2026 = pd.DataFrame({"ds":[pd.Timestamp(year=2026,month=m,day=1) for m in range(1,13)], "yhat_lr": preds_lr})
    # 2) Se possível, rodar Prophet para essa rota (mais preciso)
    df_prophet_out = None
    if use_prophet_if_possible and sub.shape[0] >= 12:
        dfp = sub.rename(columns={"DATA":"ds","tar_media":"y"})
        m = Prophet(yearly_seasonality=True)
        if sub["temp"].notna().any():
            try:
                m.add_regressor("temp")
            except Exception:
                pass
        try:
            m.fit(dfp)
            future = m.make_future_dataframe(periods=12,freq="MS")
            if "temp" in dfp.columns:
                future["temp"] = dfp["temp"].mean()
            fc = m.predict(future)
            df_prophet_out = fc[fc["ds"].dt.year==2026][["ds","yhat"]].rename(columns={"yhat":"yhat_prophet"})
        except Exception as e:
            df_prophet_out = None
    # merge results: prefer Prophet where available, else LR
    merged = df_lr_2026.copy()
    if df_prophet_out is not None:
        merged = merged.merge(df_prophet_out, on="ds", how="left")
        merged["yhat_final"] = merged["yhat_prophet"].fillna(merged["yhat_lr"])
    else:
        merged["yhat_final"] = merged["yhat_lr"]
    merged["Mes"] = merged["ds"].dt.month.map(MES_NAME)
    merged["Tarifa Prevista (R$)"] = merged["yhat_final"].round(0)
    return merged[["ds","Mes","Tarifa Prevista (R$)"]], None

# Quando o usuário clica em gerar
if btn_pred:
    with st.spinner("Gerando previsão — regressão + Prophet (Misto)..."):
        table_2026, err = forecast_route(rota_sel, dff, use_prophet_if_possible=True)
    if err:
        st.warning(err)
    else:
        st.markdown(f"### Resultado — Previsão Mensal 2026 para **{rota_sel}**")
        # tabela
        st.dataframe(table_2026.reset_index(drop=True).assign(ds=lambda df: df["ds"].dt.strftime("%Y-%m-%d")))
        # gráfico
        fig = px.line(table_2026, x="Mes", y="Tarifa Prevista (R$)", markers=True, title=f"📈 Previsão Mensal 2026 — {rota_sel}")
        fig.update_layout(yaxis_title="Tarifa média prevista (R$)", xaxis_title="Mês")
        st.plotly_chart(fig, use_container_width=True)
        # download CSV
        csv_out = table_2026.to_csv(index=False, encoding="utf-8")
        st.download_button("⬇️ Baixar CSV da previsão (2026)", csv_out, file_name=f"previsao_2026_{origem}_{destino}.csv", mime="text/csv")

# ---------------------------------------------
# 5) Ranking final "Evite essas rotas em 2026" (SR2 deliverable)
# ---------------------------------------------
st.markdown("---")
st.header("🏆 Ranking SR2 — Evite essas rotas em 2026")

if rank_reg.empty:
    st.info("Sem ranking calculado.")
else:
    # mostrar top 25 com sinal
    top_display = rank_reg.copy()
    top_display["pred_2026_mean"] = top_display["pred_2026_mean"].round(0)
    top_display["current_mean"] = top_display["current_mean"].round(0)
    top_display = top_display[["ROTA","current_mean","pred_2026_mean","pct_change","SINAL","n_obs"]].rename(
        columns={"current_mean":"Atual (R$)","pred_2026_mean":"Prev 2026 (R$)","pct_change":"Δ relativo","n_obs":"Obs"}
    )
    st.dataframe(top_display.head(25).style.format({"Δ relativo":"{:.2%}"}))
    st.markdown("**Legendas:** 🛑 Forte alta → Evitar; ⚠️ Atenção → Planejar com cautela; 📉 Queda → Boa oportunidade.")

# ---------------------------------------------
# 6) Export completo do ranking
# ---------------------------------------------
if not rank_reg.empty:
    csv_rank = rank_reg.to_csv(index=False)
    st.download_button("⬇️ Baixar Ranking SR2 (CSV)", csv_rank, file_name="ranking_sr2_rotas_prev_2026.csv", mime="text/csv")

# ---------------------------------------------
# RODAPÉ / INFO
# ---------------------------------------------
st.markdown("---")
st.caption("Bora Alí — SR2 • Tema ROXO — Misto: regressão + Prophet. Visual e prático — pronto para apresentação.")
