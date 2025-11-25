# app.py — Bora Alí (Versão Profissional SR2)
# Dataset esperado (na mesma pasta): INMET_ANAC_EXTREMAMENTE_REDUZIDO.csv
# Colunas esperadas: COMPANHIA, ANO, MES, ORIGEM, DESTINO, TARIFA, TEMP_MEDIA
# Opcional: DEST_LAT, DEST_LON para mapa interativo
#
# Dependências:
# pip install streamlit pandas numpy scikit-learn plotly pydeck joblib python-pptx openpyxl

import streamlit as st
import pandas as pd
import numpy as np
import os
from datetime import datetime
import joblib

# ML
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import GroupKFold, cross_val_score
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error, mean_squared_error

# Visuals
import plotly.express as px
import plotly.graph_objects as go
import pydeck as pdk

# ----------------- CONFIG PÁGINA -----------------
st.set_page_config(page_title="Bora Alí — SR2 (Profissional)", layout="wide", initial_sidebar_state="expanded")

# ----------------- ESTILO NEON -----------------
st.markdown(
    """
    <style>
    .neon-title { font-size:36px; font-weight:800; color:#D9B3FF;
                 text-shadow: 0 0 8px #C07CFF, 0 0 20px #8C4CFF; margin-bottom:6px;}
    .neon-sub { font-size:14px; color:#BFFFD9; text-shadow:0 0 6px #6EF0B0; }
    .card { padding:12px; border-radius:12px; background:linear-gradient(180deg, rgba(255,255,255,0.02), rgba(255,255,255,0.01)); box-shadow:0 6px 20px rgba(0,0,0,0.18); }
    .small { font-size:12px; color:#bdbdbd; }
    </style>
    """,
    unsafe_allow_html=True
)

# ----------------- HELPERS -----------------
def month_to_season(m: int) -> str:
    if m in [12,1,2]: return "VERÃO"
    if m in [3,4,5]: return "OUTONO"
    if m in [6,7,8]: return "INVERNO"
    return "PRIMAVERA"

@st.cache_data
def load_dataset(path: str):
    """Carrega CSV sem parse_dates (não há coluna DATA)."""
    df = pd.read_csv(path)
    # Garantir colunas mínimas
    required = {"COMPANHIA","ANO","MES","ORIGEM","DESTINO","TARIFA","TEMP_MEDIA"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"CSV está faltando colunas obrigatórias: {missing}")
    # Tipagem
    df["ANO"] = df["ANO"].astype(int)
    df["MES"] = df["MES"].astype(int)
    df["COMPANHIA"] = df["COMPANHIA"].astype(str)
    df["ORIGEM"] = df["ORIGEM"].astype(str)
    df["DESTINO"] = df["DESTINO"].astype(str)
    df["TARIFA"] = pd.to_numeric(df["TARIFA"], errors="coerce")
    df["TEMP_MEDIA"] = pd.to_numeric(df["TEMP_MEDIA"], errors="coerce")
    df["SEASON"] = df["MES"].apply(month_to_season)
    # Criar um campo de período legível (YYYY-MM)
    df["PERIODO"] = df["ANO"].astype(str) + "-" + df["MES"].astype(str).str.zfill(2)
    # ROTA (origem → destino)
    if "ROTA" not in df.columns:
        df["ROTA"] = df["ORIGEM"] + " → " + df["DESTINO"]
    # Optional lat/lon may exist
    return df

# ----------------- CARREGAMENTO -----------------
DATA_PATH = "INMET_ANAC_EXTREMAMENTE_REDUZIDO.csv"
if not os.path.exists(DATA_PATH):
    st.error(f"Arquivo não encontrado: {DATA_PATH}. Coloque o CSV na raiz do app ou envie via sidebar.")
    uploaded = st.sidebar.file_uploader("Envie o CSV (INMET_ANAC_EXTREMAMENTE_REDUZIDO.csv)", type=["csv"])
    if uploaded is not None:
        df = pd.read_csv(uploaded)
        # tentar normalizar muito rapidamente (mesma rotina)
        df.to_csv(DATA_PATH, index=False)
        st.success("CSV salvo temporariamente. Recarregue a página.")
        st.stop()
    else:
        st.stop()
else:
    try:
        df = load_dataset(DATA_PATH)
    except Exception as e:
        st.error(f"Erro ao carregar o dataset: {e}")
        st.stop()

# ----------------- HEADER -----------------
st.markdown('<div class="neon-title">Bora Alí — Dashboard (Versão Profissional SR2)</div>', unsafe_allow_html=True)
st.markdown('<div class="neon-sub">Dashboard interativo | filtros inteligentes | previsão 2026 por rota | narrativa SR2</div>', unsafe_allow_html=True)
st.markdown("---")

# ----------------- SIDEBAR (Filtros inteligentes) -----------------
st.sidebar.header("Filtros Inteligentes")
years = sorted(df["ANO"].unique())
sel_years = st.sidebar.multiselect("Ano(s)", options=years, default=years)
sel_months = st.sidebar.multiselect("Mês(es)", options=list(range(1,13)), default=list(range(1,13)))
capitais = sorted(df["DESTINO"].unique())
sel_capitais = st.sidebar.multiselect("Destino (Capitais)", options=capitais, default=capitais)
companhias = sorted(df["COMPANHIA"].unique())
sel_comp = st.sidebar.multiselect("Companhia(s)", options=companhias, default=companhias)
sel_season = st.sidebar.multiselect("Estação(s)", options=["VERÃO","OUTONO","INVERNO","PRIMAVERA"], default=["VERÃO","OUTONO","INVERNO","PRIMAVERA"])

# quick option: focar apenas SR2 (se você quiser)
st.sidebar.markdown("---")
if st.sidebar.checkbox("Focar somente nas regras SR2 (pré-configurado)", value=True):
    # aqui você pode aplicar qualquer regra adicional necessária para SR2
    pass

# ----------------- APLICAR FILTROS -----------------
df_filtered = df[
    (df["ANO"].isin(sel_years)) &
    (df["MES"].isin(sel_months)) &
    (df["DESTINO"].isin(sel_capitais)) &
    (df["COMPANHIA"].isin(sel_comp)) &
    (df["SEASON"].isin(sel_season))
].copy()

# ----------------- KPIs -----------------
k1, k2, k3, k4 = st.columns([1.2,1,1,1])
k1.markdown(f"<div class='card'>📊 <b>Registros</b><div class='small'>{len(df_filtered):,} registros filtrados</div></div>", unsafe_allow_html=True)
k2.markdown(f"<div class='card'>💸 <b>Tarifa média (R$)</b><div class='small'>{df_filtered['TARIFA'].mean():.2f}</div></div>", unsafe_allow_html=True)
k3.markdown(f"<div class='card'>🌡️ <b>Temp. média (°C)</b><div class='small'>{df_filtered['TEMP_MEDIA'].mean():.1f}</div></div>", unsafe_allow_html=True)
k4.markdown(f"<div class='card'>✈️ <b>Rotas únicas</b><div class='small'>{df_filtered['ROTA'].nunique()}</div></div>", unsafe_allow_html=True)

st.markdown("---")

# ----------------- Painel 1: Sazonalidade e Séries -----------------
st.header("📈 Sazonalidade & Séries Temporais")
# série temporal agregada por período
ts = df_filtered.groupby("PERIODO").agg({"TARIFA":"mean","TEMP_MEDIA":"mean"}).reset_index()
# converte PERIODO para data-tempo para ordenação:
ts["DATA_ORD"] = pd.to_datetime(ts["PERIODO"] + "-01")
fig_ts = px.line(ts.sort_values("DATA_ORD"), x="DATA_ORD", y="TARIFA", title="Tarifa média por período", markers=True)
fig_ts.update_layout(yaxis_title="Tarifa (R$)", xaxis_title="Período")
st.plotly_chart(fig_ts, use_container_width=True)

# breakdown por estação
st.subheader("Tarifa média por estação")
season_stats = df_filtered.groupby("SEASON").agg(TARIFA=("TARIFA","mean"), ROTAS=("ROTA","nunique")).reset_index()
fig_season = px.bar(season_stats, x="SEASON", y="TARIFA", text="ROTAS",
                    title="Tarifa média por estação (média dos registros filtrados)",
                    color="SEASON",
                    color_discrete_map={"VERÃO":"#C77DFF","OUTONO":"#FFA24D","INVERNO":"#7FFFD4","PRIMAVERA":"#FFD97B"})
st.plotly_chart(fig_season, use_container_width=True)

st.markdown("---")

# ----------------- Painel 2: Mapas (se lat/lon presente) -----------------
st.header("🗺️ Mapa de Capitais (Tarifa média por ponto)")
if {"DEST_LAT","DEST_LON"} <= set(df_filtered.columns):
    map_df = df_filtered.groupby(["DESTINO","DEST_LAT","DEST_LON"]).agg(TARIFA_MEDIA=("TARIFA","mean"), REG=("ROTA","nunique")).reset_index()
    midpoint = (map_df["DEST_LAT"].mean(), map_df["DEST_LON"].mean())
    layer = pdk.Layer("ScatterplotLayer",
                      data=map_df,
                      get_position='[DEST_LON, DEST_LAT]',
                      get_fill_color='[255-(TARIFA_MEDIA%255), 100, TARIFA_MEDIA%255, 180]',
                      get_radius=45000,
                      pickable=True)
    deck = pdk.Deck(initial_view_state=pdk.ViewState(latitude=midpoint[0], longitude=midpoint[1], zoom=4), layers=[layer],
                    tooltip={"text":"{DESTINO}\nTarifa média: {TARIFA_MEDIA:.2f} R$"})
    st.pydeck_chart(deck)
else:
    st.info("Mapa desativado: arquivo não contém colunas DEST_LAT e DEST_LON. Adicione essas colunas para mapa interativo.")

st.markdown("---")

# ----------------- Painel 3: Top Rotas & Comparações -----------------
st.header("🔍 Top rotas & Comparações")
top_routes = df_filtered.groupby("ROTA").agg(TARIFA_MEDIA=("TARIFA","mean"), FREQ=("ROTA","count")).reset_index().sort_values("FREQ", ascending=False)
st.subheader("Top 20 rotas por registros")
st.dataframe(top_routes.head(20), use_container_width=True)

st.subheader("Tarifa média por Companhia")
comp_stats = df_filtered.groupby("COMPANHIA").agg(TARIFA_MEDIA=("TARIFA","mean"), ROTAS=("ROTA","nunique"), REG=("ROTA","count")).reset_index()
fig_comp = px.bar(comp_stats.sort_values("TARIFA_MEDIA", ascending=False), x="COMPANHIA", y="TARIFA_MEDIA", title="Tarifa média por companhia")
st.plotly_chart(fig_comp, use_container_width=True)

st.markdown("---")

# ----------------- Painel 4: Modelagem e Previsão 2026 -----------------
st.header("🤖 Modelagem & Previsão de Tarifas — 2026")
st.write("""
Modelo padrão: GradientBoostingRegressor com engenharia de features temporal.
Validação: GroupKFold por ROTA (evita leakage entre rotas).
Saída: previsão mensal 2026 por rota selecionada.
""")

# Selecionar rota para previsão
rota_options = sorted(df["ROTA"].unique())
rota_escolhida = st.selectbox("Escolha a ROTA (origem → destino) para previsão 2026", options=rota_options)

# Botões de controle
colA, colB, colC = st.columns([1,1,1])
with colA:
    treinar = st.button("Treinar modelo (usar todo 2023-2025)")
with colB:
    gerar = st.button("Gerar previsão 2026 para rota selecionada")
with colC:
    salvar = st.button("Salvar modelo treinado (model.joblib)")

# Preparo dos dados para modelagem (usar todo df para treinar)
df_model = df.copy()
# features temporais
df_model["month_sin"] = np.sin(2*np.pi*df_model["MES"]/12)
df_model["month_cos"] = np.cos(2*np.pi*df_model["MES"]/12)
# lag / média histórica por rota (3 meses média) - só se suficiente histórico
df_model = df_model.sort_values(["ROTA","ANO","MES"])
# criar média movel 3 meses de tarifa por rota (pode gerar NaNs para primeiras linhas)
df_model["tarifa_roll3"] = df_model.groupby("ROTA")["TARIFA"].transform(lambda x: x.rolling(3, min_periods=1).mean())

# Selecionar colunas
FEATURES = ["ANO","MES","ORIGEM","DESTINO","COMPANHIA","TEMP_MEDIA","month_sin","month_cos","tarifa_roll3"]
TARGET = "TARIFA"

# Treinar modelo
model = None
cv_mae = None
if treinar:
    X = df_model[FEATURES].copy()
    y = df_model[TARGET].values
    # Preproc: one-hot categories
    cat_cols = ["ORIGEM","DESTINO","COMPANHIA"]
    preproc = ColumnTransformer([("cat", OneHotEncoder(handle_unknown="ignore", sparse=False), cat_cols)], remainder="passthrough")
    model = Pipeline([("pre", preproc), ("gbr", GradientBoostingRegressor(n_estimators=500, learning_rate=0.05, max_depth=4, random_state=42))])
    # Grupo: rota
    groups = df_model["ROTA"].values
    gkf = GroupKFold(n_splits=5)
    # CV (MAE)
    st.info("Rodando validação cruzada (GroupKFold) — isso pode demorar alguns segundos...")
    try:
        scores = -cross_val_score(model, X, y, cv=gkf.split(X, y, groups=groups), scoring="neg_mean_absolute_error", n_jobs=1)
        cv_mae = scores.mean()
        # Treinar final no dataset completo
        model.fit(X, y)
        st.success(f"Modelo treinado. CV MAE médio (GroupKFold): {cv_mae:.2f} R$")
    except Exception as e:
        st.error(f"Erro durante treinamento/CV: {e}")

# Gerar previsão 2026 para rota selecionada
if gerar:
    if model is None:
        # tentar carregar modelo salvo se existir
        if os.path.exists("model.joblib"):
            model = joblib.load("model.joblib")
            st.info("Modelo carregado de model.joblib")
        else:
            st.error("Modelo não treinado. Clique em 'Treinar modelo' primeiro.")
    if model is not None:
        # construir dataframe future 2026 meses 1..12 para a rota escolhida
        base = df_model[df_model["ROTA"]==rota_escolhida]
        if base.shape[0] == 0:
            st.error("Não há histórico para a rota selecionada.")
        else:
            future = pd.DataFrame({"ANO":2026, "MES": list(range(1,13))})
            future["ORIGEM"] = base["ORIGEM"].iloc[0]
            future["DESTINO"] = base["DESTINO"].iloc[0]
            # companhia mais frequente na rota (fallback)
            future["COMPANHIA"] = base["COMPANHIA"].mode().iloc[0]
            # TEMP_MEDIA: usar média histórica por mês (meses 1..12) da rota; preencher com média global se NaN
            month_temp = base.groupby("MES")["TEMP_MEDIA"].mean().reindex(range(1,13))
            future["TEMP_MEDIA"] = future["MES"].map(lambda m: month_temp.get(m, np.nan))
            future["TEMP_MEDIA"].fillna(df["TEMP_MEDIA"].mean(), inplace=True)
            future["month_sin"] = np.sin(2*np.pi*future["MES"]/12)
            future["month_cos"] = np.cos(2*np.pi*future["MES"]/12)
            # tarifa_roll3: usar média móvel histórica do mesmo mês se possível, senão média rota
            rota_roll = base.groupby("MES")["TARIFA"].mean().reindex(range(1,13))
            future["tarifa_roll3"] = future["MES"].map(lambda m: rota_roll.get(m, np.nan))
            future["tarifa_roll3"].fillna(base["TARIFA"].mean(), inplace=True)
            # Prever
            X_future = future[["ANO","MES","ORIGEM","DESTINO","COMPANHIA","TEMP_MEDIA","month_sin","month_cos","tarifa_roll3"]]
            preds = model.predict(X_future)
            future["PRED_TARIFA_2026"] = preds
            # mostrar tabela e gráfico
            st.subheader(f"Previsão mensal 2026 — {rota_escolhida}")
            st.dataframe(future[["MES","PRED_TARIFA_2026"]].rename(columns={"MES":"Mês","PRED_TARIFA_2026":"Tarifa Prevista (R$)"}), use_container_width=True)
            figf = px.line(future, x="MES", y="PRED_TARIFA_2026", markers=True, title=f"Previsão 2026 — {rota_escolhida}")
            st.plotly_chart(figf, use_container_width=True)

# Salvar modelo
if salvar:
    if model is None:
        st.error("Nenhum modelo para salvar. Treine primeiro.")
    else:
        joblib.dump(model, "model.joblib")
        st.success("Modelo salvo em model.joblib no diretório do app.")

st.markdown("---")

# ----------------- Painel 5: Narrativa SR2 & Checklist -----------------
st.header("📚 Narrativa e checklist prático para SR2")
st.markdown("""
**Sugestão de roteiro de apresentação (SR2)**

1. **Contexto & Pergunta:** por que prever tarifas? impacto no planejamento de viagens e em políticas de precificação.
2. **Dados & ETL:** explicar origem do dataset (INMET + ANAC reduzido), colunas, limpeza, tratamento de outliers, criação de `ROTA`, `PERIODO` e `SEASON`.
3. **Protótipo & Visualização:** mostrar filtros, mapas (se houver lat/lon), sazonalidade e top rotas.
4. **Modelagem:** justificar escolha do modelo (GradientBoosting), features (temperatura, cyclical month, média móvel), validação (GroupKFold por rota).
5. **Métricas & Interpretação:** apresentar MAE (CV) e exemplos de previsões 2026.
6. **Entregáveis SR2:** app funcional (Streamlit), notebook com ETL e modelagem (Jupyter/Colab), slides com narrativa, repositório com código e CSV.
""")

st.markdown("**Arquivos locais enviados (referência):**")
st.markdown("- Cronograma / Processo: `/mnt/data/Processos de Acompanhamento - BORA ALÍ (Cronograma) - Página1 (1).pdf`")
st.markdown("- Documento SR1: `/mnt/data/BORA ALÍ - SR1 (1).pdf`")
st.markdown("- Protótipo Dashboard (Capitais): `/mnt/data/Bora Alí — Dashboard (Capitais) · Streamlit.pdf`")
st.markdown("- Home - Análise de Tarifas: `/mnt/data/Home - Análise de Tarifas · Streamlit.pdf`")

st.markdown("---")

# ----------------- Rodapé: instruções de entrega e deploy -----------------
st.subheader("📦 Entrega & Deploy (passo-a-passo)")
st.markdown("""
1. Suba `app.py` e `INMET_ANAC_EXTREMAMENTE_REDUZIDO.csv` no repositório `LeandroFA/bora-ali-dashboard`.
2. Adicione um `requirements.txt` com as dependências (streamlit, pandas, numpy, scikit-learn, plotly, pydeck, joblib, python-pptx, openpyxl).
3. Conecte no Streamlit Cloud (ou Render) para deploy automático a partir do GitHub.
4. Na apresentação SR2 leve: link do app, notebook .ipynb com ETL, slides (.pptx) e relatório curto (1–2 páginas).
""")

st.success("App pronto — versão profissional carregada. Se quiser, eu já gero o notebook ETL e os slides PPTX automaticamente com os gráficos e narrativa (diga 'gerar notebook' ou 'gerar slides').")
