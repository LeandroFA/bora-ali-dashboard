# app_streamlit_boraali_sr2.py
# Requisitos: streamlit, pandas, numpy, scikit-learn, plotly, pydeck, folium(optional), joblib
# pip install streamlit pandas numpy scikit-learn plotly pydeck joblib

import streamlit as st
st.set_page_config(page_title="Bora Alí — SR2", layout="wide", initial_sidebar_state="expanded")

# ---------- Estilo (neon) ----------
st.markdown(
    """
    <style>
    .main .block-container{padding:1.5rem 2rem;}
    .neon-title { font-size:34px; font-weight:700; color: #E7C6FF; text-shadow: 0 0 8px #C874FF, 0 0 20px #9B4DFF; }
    .neon-kpi { font-size:28px; font-weight:600; color:#7FFFD4; text-shadow:0 0 6px #2EE2A8; }
    .neon-accent { color: #FFA24D; text-shadow:0 0 8px #FF7A00; font-weight:600; }
    .card { padding:14px; border-radius:12px; box-shadow: 0 8px 20px rgba(0,0,0,0.08); background: linear-gradient(135deg, rgba(255,255,255,0.02), rgba(255,255,255,0.01)); }
    .small { font-size:12px; color: #bdbdbd; }
    </style>
    """,
    unsafe_allow_html=True
)

# ---------- Helpers ----------
import pandas as pd, numpy as np, joblib, os
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import GroupKFold, train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error, mean_squared_error
import plotly.express as px
import plotly.graph_objects as go
import pydeck as pdk
from datetime import datetime

@st.cache_data
def load_data_local(path_csv):
    df = pd.read_csv(path_csv, parse_dates=["DATA"], dayfirst=False, infer_datetime_format=True)
    # expected columns: ORIGEM, DESTINO, ROTA, TARIFA, TARIFA_MEDIA, TEMP_MEDIA, IPCA_MES, MES, ANO, COMPANHIA
    return df

def month_to_season(month):
    # Brazil seasons (DJF = Verão, MAM = Outono, JJA = Inverno, SON = Primavera)
    if month in [12,1,2]:
        return "VERÃO"
    if month in [3,4,5]:
        return "OUTONO"
    if month in [6,7,8]:
        return "INVERNO"
    return "PRIMAVERA"

def prepare_model_data(df):
    # Aggregate to monthly route-level (origin-dest-company-month-year)
    df["MES"] = df["DATA"].dt.month
    df["ANO"] = df["DATA"].dt.year
    df["SEASON"] = df["MES"].apply(month_to_season)
    group_cols = ["ORIGEM","DESTINO","COMPANHIA","ANO","MES"]
    agg = df.groupby(group_cols).agg({
        "TARIFA":"mean",
        "TEMP_MEDIA":"mean",
        "IPCA_MES":"mean",
        "ROTA":"first"
    }).reset_index().rename(columns={"TARIFA":"TARIFA_MEDIA"})
    # features
    agg["YM"] = agg["ANO"]*100 + agg["MES"]
    return agg

# ---------- Layout ----------
st.title("Bora Alí — SR2 (Dashboard Interativo)")
st.markdown('<div class="neon-title">Insights, Previsão 2026 e História para SR2</div>', unsafe_allow_html=True)

# Sidebar: carga de dados
st.sidebar.header("Dados & Configurações")
st.sidebar.markdown("Base: ANAC + INMET + IPCA (2023–2025). Use CSV no repo ou local.")
data_path = st.sidebar.text_input("Caminho do CSV (ex: data/boraali_clean.csv)", value="data/boraali_clean.csv")
if not os.path.exists(data_path):
    st.sidebar.error("Arquivo não encontrado no caminho informado. Faça upload ou coloque o CSV em 'data/'.")
    uploaded = st.sidebar.file_uploader("Ou envie o CSV aqui", type=["csv"])
    if uploaded is not None:
        df_raw = pd.read_csv(uploaded, parse_dates=["DATA"])
        st.success("CSV carregado via upload.")
    else:
        st.stop()
else:
    df_raw = load_data_local(data_path)

# quick KPIs
col1, col2, col3, col4 = st.columns([1.2,1,1,1])
with col1:
    st.markdown("<div class='card'><div class='neon-kpi'>Registros</div><div class='small'>Filtrados</div></div>", unsafe_allow_html=True)
    st.metric("Registros", f"{len(df_raw):,}")
with col2:
    st.markdown("<div class='card'><div class='neon-kpi'>Tarifa média (R$)</div></div>", unsafe_allow_html=True)
    st.metric("", f"{df_raw['TARIFA'].mean():.2f}")
with col3:
    st.markdown("<div class='card'><div class='neon-kpi'>Temp média (°C)</div></div>", unsafe_allow_html=True)
    st.metric("", f"{df_raw['TEMP_MEDIA'].mean():.1f}")
with col4:
    st.markdown("<div class='card'><div class='neon-kpi'>Rotas únicas</div></div>", unsafe_allow_html=True)
    st.metric("", f"{df_raw['ROTA'].nunique()}")

# ---------- Filters (inteligentes) ----------
st.sidebar.markdown("### Filtros Inteligentes")
years = sorted(df_raw["DATA"].dt.year.unique().tolist())
sel_years = st.sidebar.multiselect("Ano(s)", options=years, default=years)
months = list(range(1,13))
sel_months = st.sidebar.multiselect("Mês(es)", options=months, default=months)
capitais = sorted(df_raw["DESTINO"].unique().tolist())
sel_capitais = st.sidebar.multiselect("Capitais (Destino)", options=capitais, default=capitais)
companhias = sorted(df_raw["COMPANHIA"].unique().tolist())
sel_comp = st.sidebar.multiselect("Companhia", options=companhias, default=companhias)
season_filter = st.sidebar.multiselect("Estação(s)", options=["VERÃO","OUTONO","INVERNO","PRIMAVERA"], default=["VERÃO","OUTONO","INVERNO","PRIMAVERA"])

# Apply filters
df_raw["ANO"] = df_raw["DATA"].dt.year
df_raw["MES"] = df_raw["DATA"].dt.month
df_raw["SEASON"] = df_raw["MES"].apply(month_to_season)
df = df_raw[
    (df_raw["ANO"].isin(sel_years)) &
    (df_raw["MES"].isin(sel_months)) &
    (df_raw["DESTINO"].isin(sel_capitais)) &
    (df_raw["COMPANHIA"].isin(sel_comp)) &
    (df_raw["SEASON"].isin(season_filter))
].copy()

st.markdown(f"### Visão filtrada — {len(df):,} registros")
st.write("Use os controles à esquerda para explorar. A paleta neon orienta a leitura por estação (lilás, verde, laranja).")

# ---------- Time series / Seasonal insights ----------
st.subheader("Sazonalidade e Insights por Estação")
ts = df.groupby(["ANO","MES"])["TARIFA"].mean().reset_index()
ts["DATA"] = pd.to_datetime(ts["ANO"].astype(str) + "-" + ts["MES"].astype(str) + "-01")
fig_ts = px.line(ts.sort_values("DATA"), x="DATA", y="TARIFA", title="Tarifa média ao longo do tempo", markers=True)
st.plotly_chart(fig_ts, use_container_width=True)

# Station breakdown (interactive)
season_summary = df.groupby("SEASON").agg({"TARIFA":"mean","TEMP_MEDIA":"mean","ROTA":"nunique"}).reset_index().rename(columns={"ROTA":"Rotas"})
fig_season = px.bar(season_summary, x="SEASON", y="TARIFA", text="Rotas", title="Tarifa média por Estação")
st.plotly_chart(fig_season, use_container_width=True)

# ---------- Map: capitais (pydeck) ----------
st.subheader("Mapa — Capitais (Tarifa média por ponto)")
# expects df to have columns DESTINO_LAT, DESTINO_LON or we'll use a small built-in dict (example)
# try to use columns if present
if 'DEST_LAT' in df.columns and 'DEST_LON' in df.columns:
    map_df = df.groupby(['DESTINO','DEST_LAT','DEST_LON']).agg({"TARIFA":"mean"}).reset_index().rename(columns={"TARIFA":"TARIFA_MEDIA"})
    midpoint = (map_df['DEST_LAT'].mean(), map_df['DEST_LON'].mean())
    layer = pdk.Layer(
        "ScatterplotLayer",
        data=map_df,
        get_position='[DEST_LON, DEST_LAT]',
        get_fill_color='[255-(TARIFA_MEDIA%255), 0, TARIFA_MEDIA%255, 180]',
        get_radius=50000,
        pickable=True
    )
    deck = pdk.Deck(
        initial_view_state=pdk.ViewState(latitude=midpoint[0], longitude=midpoint[1], zoom=4),
        layers=[layer],
        tooltip={"text":"{DESTINO}\nTarifa média: {TARIFA_MEDIA:.2f}"}
    )
    st.pydeck_chart(deck)
else:
    st.info("Mapa requer colunas DEST_LAT e DEST_LON ou dicionário de coordenadas. Adicione no dataset ou no repo o mapeamento lat/lon das capitais.")

# ---------- Top rotas e comparação por companhia ----------
st.subheader("Top rotas — frequência e tarifa média")
route_stats = df.groupby("ROTA").agg({"TARIFA":"mean","DATA":"count"}).rename(columns={"DATA":"freq","TARIFA":"tarifa_media"}).reset_index().sort_values("freq", ascending=False)
st.dataframe(route_stats.head(20))

st.subheader("Comparação por Companhia")
comp_stats = df.groupby("COMPANHIA").agg({"TARIFA":"mean","ROTA":"nunique","DATA":"count"}).rename(columns={"ROTA":"rotas_unicas","DATA":"registros"}).reset_index()
fig_comp = px.bar(comp_stats, x="COMPANHIA", y="TARIFA", title="Tarifa média por companhia")
st.plotly_chart(fig_comp, use_container_width=True)
st.dataframe(comp_stats)

# ---------- Modelagem / Previsão 2026 ----------
st.markdown("---")
st.subheader("Modelagem: previsão de tarifas (2026) por ROTA")

# Prepare modeling dataset
agg = prepare_model_data(df_raw)  # use full 2023-2025
st.write("Dados agregados (rota-mês) usados no modelo:")
st.dataframe(agg.head())

# Select route for forecast
selected_rota = st.selectbox("Escolha rota (origem - destino) para previsão 2026", options=sorted(agg["ROTA"].unique()))
route_df = agg[agg["ROTA"]==selected_rota].sort_values(["ANO","MES"])
st.write(f"Histórico da rota: {selected_rota}")
st.line_chart(route_df.set_index(pd.to_datetime(route_df["ANO"].astype(str) + "-" + route_df["MES"].astype(str) + "-01"))["TARIFA_MEDIA"])

# Modeling pipeline (tabbed explanation)
st.write("Modelo: GradientBoostingRegressor com features temporais + temperatura + ipca + one-hot de ORIGEM/DESTINO/COMPANHIA.")
if st.button("Treinar modelo e gerar previsão 2026 para rota selecionada"):
    # features
    df_model = agg.copy()
    # use 2023-2025 as training; we'll create synthetic 2026 rows for months 1..12 and predict
    X = df_model[["ANO","MES","ORIGEM","DESTINO","COMPANHIA","TEMP_MEDIA","IPCA_MES"]].copy()
    y = df_model["TARIFA_MEDIA"].values
    # create features: cyclical month
    X["month_sin"] = np.sin(2*np.pi*X["MES"]/12)
    X["month_cos"] = np.cos(2*np.pi*X["MES"]/12)
    cat_cols = ["ORIGEM","DESTINO","COMPANHIA"]
    num_cols = ["ANO","TEMP_MEDIA","IPCA_MES","month_sin","month_cos"]
    preproc = ColumnTransformer(transformers=[
        ("cat", OneHotEncoder(handle_unknown="ignore", sparse=False), cat_cols),
    ], remainder="passthrough")
    model = Pipeline([
        ("pre", preproc),
        ("gbr", GradientBoostingRegressor(n_estimators=300, learning_rate=0.05, max_depth=4, random_state=42))
    ])
    # train-test split by route group to avoid leakage
    groups = df_model["ROTA"].values
    gkf = GroupKFold(n_splits=5)
    # simple train on everything (production) and later evaluate via CV
    model.fit(X, y)
    # Evaluate CV MAE
    from sklearn.model_selection import cross_val_score
    import warnings
    warnings.filterwarnings("ignore")
    cv_mae = -cross_val_score(model, X, y, cv=gkf.split(X,y,groups=groups), scoring="neg_mean_absolute_error", n_jobs=1).mean()
    st.success(f"Modelo treinado. CV MAE médio (por rota-groups): {cv_mae:.2f} R$")

    # Build 2026 features for the selected route
    sel = route_df.copy()
    # We'll use average TEMP_MEDIA and IPCA per month from historical same month (simple approach)
    months_2026 = list(range(1,13))
    future_rows = []
    for m in months_2026:
        avg_temp = agg[(agg["ANO"].isin([2023,2024,2025])) & (agg["MES"]==m) & (agg["ROTA"]==selected_rota)]["TEMP_MEDIA"].mean()
        avg_ipca = agg[(agg["ANO"].isin([2023,2024,2025])) & (agg["MES"]==m) & (agg["ROTA"]==selected_rota)]["IPCA_MES"].mean()
        origem = sel["ORIGEM"].iloc[0]
        destino = sel["DESTINO"].iloc[0]
        companhia = sel["COMPANHIA"].mode().iloc[0] if not sel["COMPANHIA"].isnull().all() else agg["COMPANHIA"].mode().iloc[0]
        future_rows.append({"ANO":2026,"MES":m,"ORIGEM":origem,"DESTINO":destino,"COMPANHIA":companhia,"TEMP_MEDIA":avg_temp if not np.isnan(avg_temp) else agg["TEMP_MEDIA"].mean(),"IPCA_MES":avg_ipca if not np.isnan(avg_ipca) else agg["IPCA_MES"].mean()})
    X_future = pd.DataFrame(future_rows)
    X_future["month_sin"] = np.sin(2*np.pi*X_future["MES"]/12)
    X_future["month_cos"] = np.cos(2*np.pi*X_future["MES"]/12)
    X_pred = X_future[["ANO","MES","ORIGEM","DESTINO","COMPANHIA","TEMP_MEDIA","IPCA_MES","month_sin","month_cos"]]
    preds = model.predict(X_pred)
    X_future["PRED_TARIFA_2026"] = preds
    st.write("Previsão mensal 2026 (R$):")
    st.dataframe(X_future[["MES","PRED_TARIFA_2026"]].rename(columns={"MES":"Mês","PRED_TARIFA_2026":"Tarifa prevista (R$)"}))
    fig_forecast = px.line(X_future, x="MES", y="PRED_TARIFA_2026", markers=True, title=f"Previsão 2026 — {selected_rota}")
    st.plotly_chart(fig_forecast, use_container_width=True)

# ---------- Storytelling / SR2 checklist ----------
st.markdown("---")
st.subheader("Narrativa e checklist para SR2 (como apresentar/comprovar)")
st.markdown("""
1. **Contexto (entendimento do negócio):** problema do viajante (alto custo + falta de previsibilidade). (Use slides do SR1 como base). :contentReference[oaicite:3]{index=3}  
2. **Dados & ETL (entendimento + preparação):** descreva as fontes (ANAC, INMET, IBGE), limpeza, padronização, tratamento de outliers e colunas criadas (ROTA, UF, MES, ANO). (Veja cronograma e SR1).   
3. **Protótipos interativos:** demonstre o dashboard com filtros, KPIs, mapas e rotas — explique escolhas visuais (neon pra estações). (Protótipo disponível). :contentReference[oaicite:5]{index=5}  
4. **Modelagem e Avaliação:** explique escolha do modelo (GradientBoosting para regressão), features e CV (GroupKFold por rota). Mostre métricas (MAE).  
5. **Resultados & Insights:** sazonalidade, regiões mais caras, influência da temperatura, e previsão 2026 por rota.  
6. **Implementação:** link para app Streamlit online, repositório com scripts e CSV limpo, e instruções de deploy (Heroku/Streamlit Cloud).  
7. **Pontos de melhoria:** robustez do modelo (usar LightGBM/XGBoost ou modelos de séries temporais como Prophet/SARIMAX), incorporar promoções/ calendário de feriados, e enriquecer dados de demanda.
""")

# ---------- Entregáveis (links locais / arquivos enviados) ----------
st.markdown("### Arquivos de apoio (cronograma / SR1 / protótipos)")
st.markdown(f"- Cronograma e processo detalhado: `/mnt/data/Processos de Acompanhamento - BORA ALÍ (Cronograma) - Página1 (1).pdf`  :contentReference[oaicite:6]{index=6}")
st.markdown(f"- SR1 (sumário e datasets usados): `/mnt/data/BORA ALÍ - SR1 (1).pdf`  :contentReference[oaicite:7]{index=7}")
st.markdown(f"- Protótipo dashboard (Capitais): `/mnt/data/Bora Alí — Dashboard (Capitais) · Streamlit.pdf`  :contentReference[oaicite:8]{index=8}")

st.markdown("---")
st.info("Checklist para entrega SR2: app funcional + notebook com ETL + script de modelagem + slides com narrativa e métricas. Boa sorte — e me diga se quer que eu gere os slides em PowerPoint automaticamente.")
