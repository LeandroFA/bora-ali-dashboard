# =============================================================
# app.py — Bora Alí — SR2 (VERSÃO FINAL, ROBUSTA)
# - Não salva/alterar CSV
# - Trata NaNs
# - Compatível com diferentes sklearn
# - Fluxo: ORIGEM -> DESTINO -> COMPANHIA
# - Mapa em memória com coords das capitais (se não houver coords no CSV)
# =============================================================

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import pydeck as pdk
import math
import warnings
warnings.filterwarnings("ignore")

# ML
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import GroupKFold, cross_val_score

st.set_page_config(page_title="Bora Alí — SR2 (Final)", layout="wide", initial_sidebar_state="expanded")

# -------------------- PALETA (centralizada) --------------------
COLOR_ORANGE = "#FF8A33"
COLOR_LILAC   = "#C77DFF"
COLOR_LIME    = "#8BFF66"
COLOR_AZUL    = "#28F8FF"

COLORS_SEASON = {
    "VERÃO": COLOR_ORANGE,
    "OUTONO": COLOR_LILAC,
    "INVERNO": COLOR_LIME,
    "PRIMAVERA": COLOR_AZUL
}

# -------------------- COORDENADAS DAS CAPITAIS (em memória) --------------------
CAPITAL_COORDS = {
    "Rio Branco": (-9.97499, -67.8243), "Maceió": (-9.64985, -35.70895),
    "Macapá": (0.034934, -51.0694),"Manaus": (-3.11903, -60.0217),
    "Salvador": (-12.9718, -38.5011),"Fortaleza": (-3.71664, -38.5423),
    "Brasília": (-15.7934, -47.8823),"Vitória": (-20.3155, -40.3128),
    "Goiânia": (-16.6864, -49.2643),"São Luís": (-2.53874, -44.2825),
    "Cuiabá": (-15.5989, -56.0949),"Campo Grande": (-20.4697, -54.6201),
    "Belo Horizonte": (-19.9167, -43.9345),"Belém": (-1.45583, -48.5039),
    "João Pessoa": (-7.11509, -34.8641),"Curitiba": (-25.4284, -49.2733),
    "Recife": (-8.04756, -34.8771),"Teresina": (-5.08921, -42.8016),
    "Rio de Janeiro": (-22.9068, -43.1729),"Natal": (-5.79448, -35.211),
    "Porto Alegre": (-30.0346, -51.2177),"Porto Velho": (-8.76077, -63.8999),
    "Boa Vista": (2.82384, -60.6753),"Florianópolis": (-27.5945, -48.5477),
    "São Paulo": (-23.5505, -46.6333),"Aracaju": (-10.9472, -37.0731),
    "Palmas": (-10.184, -48.3336)
}

# -------------------- UTILS --------------------
def month_to_season(m:int)->str:
    if m in (12,1,2): return "VERÃO"
    if m in (3,4,5): return "OUTONO"
    if m in (6,7,8): return "INVERNO"
    return "PRIMAVERA"

def safe_onehot(**kwargs):
    """
    Create OneHotEncoder compatible with multiple sklearn versions.
    Tries 'sparse_output' first, falls back to 'sparse' if needed.
    """
    try:
        return OneHotEncoder(**{**kwargs, **{"sparse_output": False}})
    except TypeError:
        return OneHotEncoder(**{**kwargs, **{"sparse": False}})

def train_model_safe(df_train, features, target="TARIFA"):
    """
    Train GradientBoosting with GroupKFold CV and safe preprocessing.
    Drops rows with missing values in required features/target.
    """
    df_t = df_train.copy()
    req = features + [target]
    df_t = df_t.dropna(subset=req)
    if df_t.shape[0] < 50:
        raise ValueError("Dados insuficientes para treinar (após limpeza).")
    cat_cols = ["ORIGEM","DESTINO","COMPANHIA"]
    preproc = ColumnTransformer([("cat", safe_onehot(handle_unknown="ignore"), cat_cols)], remainder="passthrough")
    model = Pipeline([("pre", preproc), ("gbr", GradientBoostingRegressor(n_estimators=300, learning_rate=0.05, max_depth=4, random_state=42))])
    X = df_t[features]
    y = df_t[target].values
    groups = df_t["ROTA"].values
    gkf = GroupKFold(n_splits=4)
    # use n_jobs=1 for safety in hosted envs
    scores = -cross_val_score(model, X, y, cv=gkf.split(X,y,groups=groups), scoring="neg_mean_absolute_error", n_jobs=1)
    cv_mae = scores.mean()
    model.fit(X,y)
    return model, cv_mae

# -------------------- LOAD DATA (your CSV in repo) --------------------
DATA_PATH = "INMET_ANAC_EXTREMAMENTE_REDUZIDO.csv"
try:
    df = pd.read_csv(DATA_PATH)
except FileNotFoundError:
    st.error(f"Arquivo não encontrado: {DATA_PATH}. Coloque o CSV na pasta do app ou me mande o caminho.")
    st.stop()

# Ensure expected columns exist (robust message if not)
EXPECTED = {"COMPANHIA","ANO","MES","ORIGEM","DESTINO","TARIFA","TEMP_MEDIA"}
missing = EXPECTED - set(df.columns)
if missing:
    st.error(f"CSV está faltando colunas obrigatórias: {missing}")
    st.stop()

# Basic typing + derived cols
df["ANO"] = df["ANO"].astype(int)
df["MES"] = df["MES"].astype(int)
df["TARIFA"] = pd.to_numeric(df["TARIFA"], errors="coerce")
df["TEMP_MEDIA"] = pd.to_numeric(df["TEMP_MEDIA"], errors="coerce")
df["SEASON"] = df["MES"].apply(month_to_season)
df["PERIODO"] = df["ANO"].astype(str) + "-" + df["MES"].astype(str).str.zfill(2)
df["ROTA"] = df["ORIGEM"].astype(str) + " → " + df["DESTINO"].astype(str)

# Add coordinates in memory only (do NOT save CSV)
if "DEST_LAT" not in df.columns or "DEST_LON" not in df.columns:
    df["DEST_LAT"] = df["DESTINO"].apply(lambda x: CAPITAL_COORDS.get(x, (np.nan,np.nan))[0])
    df["DEST_LON"] = df["DESTINO"].apply(lambda x: CAPITAL_COORDS.get(x, (np.nan,np.nan))[1])
# Drop rows without coords to avoid map errors (but keep rest of analysis by copy when needed)
df_map_ready = df.dropna(subset=["DEST_LAT","DEST_LON"]).copy()

# -------------------- SIDEBAR FILTERS --------------------
st.sidebar.header("Filtros Inteligentes")
# months by name
MES_NAMES = {1:"Janeiro",2:"Fevereiro",3:"Março",4:"Abril",5:"Maio",6:"Junho",7:"Julho",8:"Agosto",9:"Setembro",10:"Outubro",11:"Novembro",12:"Dezembro"}
mes_default = list(MES_NAMES.values())
sel_months_names = st.sidebar.multiselect("Mês(es)", options=list(MES_NAMES.values()), default=mes_default)
sel_months = [k for k,v in MES_NAMES.items() if v in sel_months_names]
if not sel_months:
    sel_months = list(MES_NAMES.keys())

sel_years = st.sidebar.multiselect("Ano(s)", options=sorted(df["ANO"].unique()), default=sorted(df["ANO"].unique()))
sel_capitais = st.sidebar.multiselect("Destino (Capitais)", options=sorted(df["DESTINO"].unique()), default=sorted(df["DESTINO"].unique()))
sel_companies = st.sidebar.multiselect("Companhia(s)", options=sorted(df["COMPANHIA"].unique()), default=sorted(df["COMPANHIA"].unique()))
sel_seasons = st.sidebar.multiselect("Estação(s)", options=["VERÃO","OUTONO","INVERNO","PRIMAVERA"], default=["VERÃO","OUTONO","INVERNO","PRIMAVERA"])

df_filtered = df[
    (df["ANO"].isin(sel_years)) &
    (df["MES"].isin(sel_months)) &
    (df["DESTINO"].isin(sel_capitais)) &
    (df["COMPANHIA"].isin(sel_companies)) &
    (df["SEASON"].isin(sel_seasons))
].copy()

# -------------------- HEADER + KPIs --------------------
st.title("✈️ Bora Alí — SR2 (Final & Estável)")
st.markdown("Cores: LARANJA • LILÁS • VERDE-LIMÃO — Sazonalidade e previsão 2026 por rota")

k1,k2,k3 = st.columns(3)
k1.metric("Registros (filtro)", f"{len(df_filtered):,}")
k2.metric("Tarifa média (R$)", f"{df_filtered['TARIFA'].mean():.2f}")
k3.metric("Rotas únicas (filtro)", f"{df_filtered['ROTA'].nunique():,}")

st.markdown("---")

# -------------------- Sazonalidade --------------------
st.subheader("🌦️ Média tarifária por estação (filtro aplicado)")
season_stats = df_filtered.groupby("SEASON")["TARIFA"].mean().reindex(["VERÃO","OUTONO","INVERNO","PRIMAVERA"]).reset_index()
fig_season = px.bar(season_stats, x="SEASON", y="TARIFA", color="SEASON", color_discrete_map=COLORS_SEASON, labels={"TARIFA":"Tarifa média (R$)"})
st.plotly_chart(fig_season, use_container_width=True)

# -------------------- Previsão por rota (ORIGEM->DESTINO->COMPANHIA) --------------------
st.subheader("🔮 Previsão por ROTA — selecione Origem → Destino → Companhia")

origens = sorted(df["ORIGEM"].unique())
origem_sel = st.selectbox("1) Escolha a ORIGEM", options=["-- escolha origem --"] + origens)
if origem_sel == "-- escolha origem --":
    st.info("Selecione a origem para prosseguir")
    st.stop()

destinos_for_origem = sorted(df[df["ORIGEM"]==origem_sel]["DESTINO"].unique())
if not destinos_for_origem:
    st.error("Nenhum destino encontrado para essa origem.")
    st.stop()

dest_sel = st.selectbox("2) Escolha o DESTINO", options=["-- escolha destino --"] + destinos_for_origem)
if dest_sel == "-- escolha destino --":
    st.info("Selecione o destino para prosseguir")
    st.stop()

route_hist = df[(df["ORIGEM"]==origem_sel) & (df["DESTINO"]==dest_sel)]
if route_hist.empty:
    st.error("Não há histórico nesta rota.")
    st.stop()

st.write(f"Tarifa média histórica da rota **{origem_sel} → {dest_sel}**: **R$ {route_hist['TARIFA'].mean():.2f}**")
companies_on_route = sorted(route_hist["COMPANHIA"].unique())
company_sel = st.selectbox("3) Escolha a COMPANHIA (para previsão)", options=companies_on_route, index=0)

# Prepare global model features and safe training (cached)
FEATURES = ["ANO","MES","ORIGEM","DESTINO","COMPANHIA","TEMP_MEDIA","TARIFA"]

# Build features for modeling (drop rows with missing critical fields first)
model_features = ["ORIGEM","DESTINO","COMPANHIA","MES","TEMP_MEDIA","TARIFA"]
df_for_model = df.dropna(subset=["ORIGEM","DESTINO","COMPANHIA","MES","TEMP_MEDIA","TARIFA"]).copy()

# create month cyclic features for extra stability (added to pipeline via passthrough later)
df_for_model["month_sin"] = np.sin(2*np.pi*df_for_model["MES"]/12)
df_for_model["month_cos"] = np.cos(2*np.pi*df_for_model["MES"]/12)
MODEL_FEATURES = ["ORIGEM","DESTINO","COMPANHIA","MES","TEMP_MEDIA","month_sin","month_cos"]

# train model safely (cache resource)
@st.cache_resource
def get_trained_model(df_input):
    model_obj, cv_mae = train_model_safe(df_input, MODEL_FEATURES, target="TARIFA")
    return model_obj, cv_mae

try:
    model, cv_mae = get_trained_model(df_for_model)
    st.success(f"Modelo pronto — CV MAE (GroupKFold por rota): {cv_mae:.2f} R$")
except Exception as e:
    st.error(f"Erro ao treinar modelo: {e}")
    st.stop()

# build future rows for 2026 for the selected pair & company
def build_future_rows(df_hist, origem, destino, companhia):
    months = list(range(1,13))
    pair = df_hist[(df_hist["ORIGEM"]==origem) & (df_hist["DESTINO"]==destino) & (df_hist["COMPANHIA"]==companhia)]
    # fallback: same origin/destination but any company
    if pair.empty:
        pair = df_hist[(df_hist["ORIGEM"]==origem) & (df_hist["DESTINO"]==destino)]
    # fallback global
    if pair.empty:
        temp_month = df_hist.groupby("MES")["TEMP_MEDIA"].mean().reindex(months)
    else:
        temp_month = pair.groupby("MES")["TEMP_MEDIA"].mean().reindex(months)
    temp_month.fillna(df_hist["TEMP_MEDIA"].mean(), inplace=True)
    rows = []
    for m in months:
        rows.append({
            "ANO":2026,
            "MES":m,
            "ORIGEM":origem,
            "DESTINO":destino,
            "COMPANHIA":companhia,
            "TEMP_MEDIA": temp_month.loc[m],
            "month_sin": math.sin(2*math.pi*m/12),
            "month_cos": math.cos(2*math.pi*m/12)
        })
    fut = pd.DataFrame(rows)
    fut["SEASON"] = fut["MES"].apply(month_to_season)
    return fut

future_df = build_future_rows(df, origem_sel, dest_sel, company_sel)

# Predict (safe)
X_future = future_df[["ORIGEM","DESTINO","COMPANHIA","MES","TEMP_MEDIA","month_sin","month_cos"]]
preds = model.predict(X_future)
future_df["PRED"] = preds

st.subheader("Previsão mensal 2026 — rota selecionada")
st.dataframe(future_df[["MES","SEASON","PRED"]].rename(columns={"MES":"Mês","SEASON":"Estação","PRED":"Tarifa prevista (R$)"}), use_container_width=True)

# plot predictions
fig_pred = px.line(future_df, x="MES", y="PRED", markers=True, title=f"Previsão mensal 2026 — {origem_sel} → {dest_sel} ({company_sel})",
                   labels={"PRED":"Tarifa prevista (R$)","MES":"Mês"}, color_discrete_sequence=[COLOR_ORANGE])
st.plotly_chart(fig_pred, use_container_width=True)

# cheapest season
season_means = future_df.groupby("SEASON")["PRED"].mean().reset_index()
cheapest_season = season_means.sort_values("PRED").iloc[0]["SEASON"]
st.markdown(f"**A rota será mais barata (previsão 2026) na estação:** **{cheapest_season}**")
st.dataframe(season_means.rename(columns={"PRED":"Tarifa média prevista (R$)"}), use_container_width=True)

# variation vs historical
hist_mean_route = route_hist["TARIFA"].mean()
pred_mean_route = future_df["PRED"].mean()
pct_change_route = (pred_mean_route - hist_mean_route) / hist_mean_route * 100 if hist_mean_route!=0 else 0.0
st.markdown(f"**Variação média prevista (2026 vs histórico):** {pct_change_route:.2f}% (hist: R$ {hist_mean_route:.2f} → prev: R$ {pred_mean_route:.2f})")

st.markdown("---")

# -------------------- MAP: Capitais QUEDA/ESTÁVEL/ALTA --------------------
st.subheader("🗺️ Mapa: Capitais — Queda / Estável / Alta (Previsão 2026 vs Histórico 2023–2025)")

# historical baseline per destination (2023-2025)
hist_base = df[df["ANO"].isin([2023,2024,2025])].groupby("DESTINO")["TARIFA"].mean().reset_index().rename(columns={"TARIFA":"HIST_MEAN"})
# predicted mean per destination (approx): average predicted means for representative origin/company pairs
pred_list = []
pairs = df[["ORIGEM","DESTINO"]].drop_duplicates()
for _, r in pairs.iterrows():
    o = r["ORIGEM"]; d = r["DESTINO"]
    # choose most frequent company for pair if available
    comp_mode = df[(df["ORIGEM"]==o)&(df["DESTINO"]==d)]["COMPANHIA"]
    if comp_mode.empty:
        comp = df["COMPANHIA"].mode().iloc[0]
    else:
        comp = comp_mode.mode().iloc[0]
    fut = build_future_rows(df, o, d, comp)
    try:
        p = model.predict(fut[["ORIGEM","DESTINO","COMPANHIA","MES","TEMP_MEDIA","month_sin","month_cos"]])
        pred_list.append({"ORIGEM":o,"DESTINO":d,"MEAN_PRED": np.mean(p)})
    except Exception:
        # fallback: use historical mean
        pd_mean = df[(df["ORIGEM"]==o)&(df["DESTINO"]==d)]["TARIFA"].mean() if not df[(df["ORIGEM"]==o)&(df["DESTINO"]==d)].empty else df["TARIFA"].mean()
        pred_list.append({"ORIGEM":o,"DESTINO":d,"MEAN_PRED": pd_mean})

pred_df = pd.DataFrame(pred_list).groupby("DESTINO")["MEAN_PRED"].mean().reset_index().rename(columns={"MEAN_PRED":"PRED_MEAN_2026"})
map_df = hist_base.merge(pred_df, on="DESTINO", how="left")
map_df = map_df.merge(df_map_ready[["DESTINO","DEST_LAT","DEST_LON"]].drop_duplicates(), on="DESTINO", how="left")

# classify by threshold ±5%
thresh = 0.05
map_df["PCT_CHANGE"] = (map_df["PRED_MEAN_2026"] - map_df["HIST_MEAN"]) / map_df["HIST_MEAN"]
def status_from_pct(p):
    if pd.isna(p): return "SEM DADO"
    if p <= -thresh: return "QUEDA"
    if p >= thresh: return "ALTA"
    return "ESTAVEL"
map_df["STATUS"] = map_df["PCT_CHANGE"].apply(status_from_pct)
color_map = {"QUEDA": COLOR_LIME, "ESTAVEL": COLOR_LILAC, "ALTA": COLOR_ORANGE, "SEM DADO": "#888888"}
map_df["COLOR_RGB"] = map_df["STATUS"].map(lambda s: [int(color_map.get(s,"#888888")[i:i+2],16) for i in (1,3,5)] if isinstance(color_map.get(s), str) else [136,136,136])

# compute impacted season per dest (approx)
impacts = []
for dest in map_df["DESTINO"].dropna().unique():
    # gather per-month preds across origins that go to this dest
    per_months = []
    for _, r in pairs[pairs["DESTINO"]==dest].iterrows():
        o = r["ORIGEM"]
        comp_mode = df[(df["ORIGEM"]==o)&(df["DESTINO"]==dest)]["COMPANHIA"]
        comp = comp_mode.mode().iloc[0] if not comp_mode.empty else df["COMPANHIA"].mode().iloc[0]
        fut = build_future_rows(df, o, dest, comp)
        try:
            p = model.predict(fut[["ORIGEM","DESTINO","COMPANHIA","MES","TEMP_MEDIA","month_sin","month_cos"]])
            per_months.append(p)
        except Exception:
            continue
    if per_months:
        avg_month = np.mean(per_months, axis=0)
        dfm = pd.DataFrame({"MES": list(range(1,13)), "PRED": avg_month})
        dfm["SEASON"] = dfm["MES"].apply(month_to_season)
        season_mean = dfm.groupby("SEASON")["PRED"].mean().reset_index()
        # choose season with largest abs difference vs historical season mean for that dest
        hist_dest = df[df["DESTINO"]==dest]
        if hist_dest.empty:
            impacted = season_mean.sort_values("PRED").iloc[0]["SEASON"]
        else:
            hist_season = hist_dest.groupby("SEASON")["TARIFA"].mean().reset_index()
            merged = season_mean.merge(hist_season, on="SEASON", how="left").fillna(0)
            merged["DIFF"] = merged["PRED"] - merged["TARIFA"]
            impacted = merged.loc[merged["DIFF"].abs().idxmax()]["SEASON"]
    else:
        impacted = None
    impacts.append({"DESTINO": dest, "IMPACT_SEASON": impacted})

imp_df = pd.DataFrame(impacts)
map_df = map_df.merge(im p_df if False else imp_df, on="DESTINO", how="left")  # safe merge (no crash if empty)

# show map if coords present
if map_df[["DEST_LAT","DEST_LON"]].dropna().shape[0] > 0:
    # use pydeck for interactive markers with pick tooltip
    layer = pdk.Layer(
        "ScatterplotLayer",
        data=map_df.dropna(subset=["DEST_LAT","DEST_LON"]),
        get_position='[DEST_LON, DEST_LAT]',
        get_fill_color='[COLOR_RGB[0], COLOR_RGB[1], COLOR_RGB[2], 180]',
        get_radius=60000,
        pickable=True
    )
    view = pdk.ViewState(latitude=map_df["DEST_LAT"].mean(), longitude=map_df["DEST_LON"].mean(), zoom=4)
    st.pydeck_chart(pdk.Deck(layers=[layer], initial_view_state=view,
                             tooltip={"text":"{DESTINO}\nStatus: {STATUS}\nHistórico: R${HIST_MEAN}\nPrevisto: R${PRED_MEAN_2026}\nEstação impactada: {IMPACT_SEASON}"}))
    st.dataframe(map_df[["DESTINO","STATUS","HIST_MEAN","PRED_MEAN_2026","PCT_CHANGE","IMPACT_SEASON"]].sort_values("PCT_CHANGE", ascending=False), use_container_width=True)
else:
    st.info("Mapa desativado: não há coordenadas válidas em memória para as capitais (verifique nomes das capitais no CSV).")

st.markdown("---")
st.markdown("✅ App final carregado. Se quiser, posso: (a) limpar/otimizar ainda mais o código, (b) gerar um notebook `.ipynb` com ETL e modelagem, ou (c) apenas ajustar paleta de cores. Diga qual opção (escreva: `notebook`, `otimizar`, `cores` ou `nada`).")
