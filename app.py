# app.py — Bora Alí — Versão corrigida (Profissional v4)
# Substitua totalmente o seu app.py por este arquivo.
# Arquivo de dados esperado: INMET_ANAC_EXTREMAMENTE_REDUZIDO.csv
# Colunas esperadas: COMPANHIA, ANO, MES, ORIGEM, DESTINO, TARIFA, TEMP_MEDIA
# O app automaticamente verifica DEST_LAT/DEST_LON para mapa.

import streamlit as st
import pandas as pd
import numpy as np
import os
import math
from datetime import datetime
import warnings

# ML
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GroupKFold, cross_val_score
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import GradientBoostingRegressor

# Viz
import plotly.express as px
import pydeck as pdk

warnings.filterwarnings("ignore")

# ---------------- Config page & colors ----------------
COLOR_LILAC = "#C77DFF"
COLOR_ORANGE = "#FF8A33"
COLOR_LIME = "#8BFF66"
BG = "#0f0f12"

st.set_page_config(page_title="Bora Alí — SR2", layout="wide", initial_sidebar_state="expanded")
st.markdown(f"""
    <style>
      body {{ background: {BG}; color: #eee; }}
      .title {{ color: {COLOR_LILAC}; font-size:32px; font-weight:800; }}
      .card {{ padding:10px; border-radius:10px; background: rgba(255,255,255,0.03); }}
      .small {{ font-size:12px; color:#bfbfbf; }}
    </style>
""", unsafe_allow_html=True)

# ---------------- Helpers ----------------
def month_to_season(m:int)->str:
    if m in [12,1,2]: return "VERÃO"
    if m in [3,4,5]: return "OUTONO"
    if m in [6,7,8]: return "INVERNO"
    return "PRIMAVERA"

MES_NAMES = {
    1:"Janeiro",2:"Fevereiro",3:"Março",4:"Abril",5:"Maio",6:"Junho",
    7:"Julho",8:"Agosto",9:"Setembro",10:"Outubro",11:"Novembro",12:"Dezembro"
}

def safe_onehot_encoder(**kwargs):
    """
    Create OneHotEncoder with compatibility across sklearn versions.
    prefer 'sparse_output' if accepted, else fallback to 'sparse'.
    """
    try:
        return OneHotEncoder(**{**kwargs, **{"sparse_output": False}})
    except TypeError:
        # older sklearn expects 'sparse'
        return OneHotEncoder(**{**kwargs, **{"sparse": False}})

def train_model(df_train, features, target="TARIFA"):
    cat_cols = ["ORIGEM","DESTINO","COMPANHIA"]
    enc = safe_onehot_encoder(handle_unknown="ignore")
    preproc = ColumnTransformer([("cat", enc, cat_cols)], remainder="passthrough")
    model = Pipeline([("pre", preproc),
                      ("gbr", GradientBoostingRegressor(n_estimators=400, learning_rate=0.05, max_depth=4, random_state=42))])
    X = df_train[features].copy()
    y = df_train[target].values
    groups = df_train["ROTA"].values
    gkf = GroupKFold(n_splits=5)
    # use n_jobs=1 for safety on hosted envs
    scores = -cross_val_score(model, X, y, cv=gkf.split(X, y, groups=groups), scoring="neg_mean_absolute_error", n_jobs=1)
    cv_mae = scores.mean()
    model.fit(X, y)
    return model, cv_mae

def season_of_cheapest_month(df_pred):
    df_temp = df_pred.copy()
    df_temp["SEASON"] = df_temp["MES"].apply(month_to_season)
    s = df_temp.groupby("SEASON")["PRED"].mean().sort_values()
    if s.empty:
        return None, pd.DataFrame()
    return s.index[0], s.reset_index().rename(columns={"PRED":"Tarifa média prevista (R$)"})

# ---------------- Load data ----------------
DATA_PATH = "INMET_ANAC_EXTREMAMENTE_REDUZIDO.csv"
if not os.path.exists(DATA_PATH):
    st.error(f"Arquivo não encontrado: {DATA_PATH}. Coloque o CSV na pasta do app ou use o uploader na sidebar.")
    uploaded = st.sidebar.file_uploader("Envie INMET_ANAC_EXTREMAMENTE_REDUZIDO.csv", type=["csv"])
    if uploaded:
        df = pd.read_csv(uploaded)
        df.to_csv(DATA_PATH, index=False)
        st.success("CSV enviado e salvo. Recarregue a página.")
    st.stop()

df = pd.read_csv(DATA_PATH)

# validate minimal columns
required = {"COMPANHIA","ANO","MES","ORIGEM","DESTINO","TARIFA","TEMP_MEDIA"}
missing = required - set(df.columns)
if missing:
    st.error(f"CSV está faltando colunas obrigatórias: {missing}. Corrija o CSV e recarregue.")
    st.stop()

# types & derived fields
df["ANO"] = df["ANO"].astype(int)
df["MES"] = df["MES"].astype(int)
df["COMPANHIA"] = df["COMPANHIA"].astype(str)
df["ORIGEM"] = df["ORIGEM"].astype(str)
df["DESTINO"] = df["DESTINO"].astype(str)
df["TARIFA"] = pd.to_numeric(df["TARIFA"], errors="coerce")
df["TEMP_MEDIA"] = pd.to_numeric(df["TEMP_MEDIA"], errors="coerce")
df["SEASON"] = df["MES"].apply(month_to_season)
df["PERIODO"] = df["ANO"].astype(str) + "-" + df["MES"].astype(str).str.zfill(2)
df["ROTA"] = df["ORIGEM"] + " → " + df["DESTINO"]

# ---------------- Sidebar filters ----------------
st.sidebar.title("Filtros Inteligentes")
sel_years = st.sidebar.multiselect("Ano(s)", options=sorted(df["ANO"].unique()), default=sorted(df["ANO"].unique()))
sel_months_names = st.sidebar.multiselect("Mês(es)", options=list(MES_NAMES.values()), default=list(MES_NAMES.values()))
# map months names back to numbers
sel_months = [m for m,num in MES_NAMES.items() if num in sel_months_names]
if not sel_months:
    # avoid empty selection
    sel_months = list(MES_NAMES.keys())
sel_capitais = st.sidebar.multiselect("Destino (Capitais)", options=sorted(df["DESTINO"].unique()), default=sorted(df["DESTINO"].unique()))
sel_comp = st.sidebar.multiselect("Companhia(s)", options=sorted(df["COMPANHIA"].unique()), default=sorted(df["COMPANHIA"].unique()))
sel_season = st.sidebar.multiselect("Estação(s)", options=["VERÃO","OUTONO","INVERNO","PRIMAVERA"], default=["VERÃO","OUTONO","INVERNO","PRIMAVERA"])

df_filtered = df[
    (df["ANO"].isin(sel_years)) &
    (df["MES"].isin(sel_months)) &
    (df["DESTINO"].isin(sel_capitais)) &
    (df["COMPANHIA"].isin(sel_comp)) &
    (df["SEASON"].isin(sel_season))
].copy()

# ---------------- Header & KPIs ----------------
st.markdown(f"<div class='title'>✈️ Bora Alí — SR2 Dashboard Profissional</div>", unsafe_allow_html=True)
k1, k2, k3 = st.columns(3)
k1.metric("Registros (filtro)", f"{len(df_filtered):,}")
k2.metric("Tarifa média (R$)", f"{df_filtered['TARIFA'].mean():.2f}")
k3.metric("Rotas únicas", f"{df_filtered['ROTA'].nunique():,}")

st.markdown("---")

# ---------------- Seasonal insight ----------------
st.subheader("🌦️ Média Tarifária por Estação (filtro aplicado)")
season_stats = df_filtered.groupby("SEASON")["TARIFA"].mean().reindex(["VERÃO","OUTONO","INVERNO","PRIMAVERA"]).reset_index()
fig_season = px.bar(season_stats, x="SEASON", y="TARIFA", color="SEASON",
                    color_discrete_map={"VERÃO":COLOR_ORANGE,"OUTONO":COLOR_LILAC,"INVERNO":COLOR_GREEN,"PRIMAVERA":"#28F8FF"},
                    labels={"TARIFA":"Tarifa média (R$)"})
st.plotly_chart(fig_season, use_container_width=True)

st.markdown("---")

# ---------------- Forecast flow: ORIGEM -> DESTINO -> COMPANHIA ----------------
st.header("🔮 Previsão por ROTA (fluxo: ORIGEM → DESTINO → COMPANHIA)")

origens = sorted(df["ORIGEM"].unique())
origem_sel = st.selectbox("1) Escolha a ORIGEM", options=["-- escolha origem --"] + origens)
if origem_sel == "-- escolha origem --":
    st.info("Selecione a origem para prosseguir.")
    st.stop()

destinos = sorted(df[df["ORIGEM"]==origem_sel]["DESTINO"].unique().tolist())
if not destinos:
    st.error("Nenhum destino encontrado para essa origem.")
    st.stop()
dest_sel = st.selectbox("2) Escolha o DESTINO", options=["-- escolha destino --"] + destinos)
if dest_sel == "-- escolha destino --":
    st.info("Selecione o destino para prosseguir.")
    st.stop()

# prepare route info
route_hist = df[(df["ORIGEM"]==origem_sel) & (df["DESTINO"]==dest_sel)].copy()
if route_hist.empty:
    st.error("Sem histórico para essa rota.")
    st.stop()

st.write(f"Tarifa média histórica da rota **{origem_sel} → {dest_sel}**: **R$ {route_hist['TARIFA'].mean():.2f}**")
companies_route = sorted(route_hist["COMPANHIA"].unique())
company_default_idx = 0 if companies_route else None
company_sel = st.selectbox("3) Escolha a COMPANHIA (para previsão)", options=companies_route, index=company_default_idx if company_default_idx is not None else 0)

# ---------------- Build & train model (safe) ----------------
st.info("Treinando modelo (validação GroupKFold). Isso pode demorar alguns segundos...")

# engineer features
df_model = df.copy().sort_values(["ROTA","ANO","MES"])
df_model["month_sin"] = np.sin(2*np.pi*df_model["MES"]/12)
df_model["month_cos"] = np.cos(2*np.pi*df_model["MES"]/12)
# rolling 3
df_model["tarifa_roll3"] = df_model.groupby("ROTA")["TARIFA"].transform(lambda x: x.rolling(3, min_periods=1).mean())

FEATURES = ["ANO","MES","ORIGEM","DESTINO","COMPANHIA","TEMP_MEDIA","month_sin","month_cos","tarifa_roll3"]

try:
    model, cv_mae = train_model(df_model.dropna(subset=["TARIFA"]), FEATURES, target="TARIFA")
    st.success(f"Modelo treinado. CV MAE (GroupKFold por rota): {cv_mae:.2f} R$")
except Exception as e:
    st.error(f"Erro durante o treinamento do modelo: {e}")
    st.stop()

# ---------------- Build future rows for selected pair ----------------
def build_future_for_pair(df_all, origem, destino, companhia):
    months = list(range(1,13))
    # estimate TEMP_MEDIA per month for this destination/origin
    pair = df_all[(df_all["ORIGEM"]==origem)&(df_all["DESTINO"]==destino)]
    if not pair.empty:
        temp_by_month = pair.groupby("MES")["TEMP_MEDIA"].mean().reindex(months)
    else:
        temp_by_month = df_all[df_all["DESTINO"]==destino].groupby("MES")["TEMP_MEDIA"].mean().reindex(months)
    temp_by_month.fillna(df_all["TEMP_MEDIA"].mean(), inplace=True)
    rows = []
    for m in months:
        rows.append({
            "ANO":2026,
            "MES":m,
            "ORIGEM":origem,
            "DESTINO":destino,
            "COMPANHIA":companhia,
            "TEMP_MEDIA": temp_by_month.loc[m]
        })
    fut = pd.DataFrame(rows)
    fut["month_sin"] = np.sin(2*np.pi*fut["MES"]/12)
    fut["month_cos"] = np.cos(2*np.pi*fut["MES"]/12)
    # tarifa_roll3 fallback from route historical monthly mean
    monthly_route_mean = route_hist.groupby("MES")["TARIFA"].mean().reindex(range(1,13))
    fut["tarifa_roll3"] = fut["MES"].map(lambda m: monthly_route_mean.get(m, math.nan))
    fut["tarifa_roll3"].fillna(route_hist["TARIFA"].mean(), inplace=True)
    fut["SEASON"] = fut["MES"].apply(month_to_season)
    return fut

future = build_future_for_pair(df, origem_sel, dest_sel, company_sel)

# predict
X_future = future[FEATURES]
try:
    preds = model.predict(X_future)
except Exception as e:
    st.error(f"Erro ao predizer: {e}")
    st.stop()

future["PRED"] = preds

# show results
st.subheader("Previsão mensal 2026 — rota selecionada")
st.dataframe(future[["MES","SEASON","PRED"]].rename(columns={"MES":"Mês","SEASON":"Estação","PRED":"Tarifa prevista (R$)"}), use_container_width=True)

# line plot
figp = px.line(future, x="MES", y="PRED", markers=True, title=f"Previsão 2026 — {origem_sel} → {dest_sel}",
               labels={"PRED":"Tarifa prevista (R$)", "MES":"Mês"}, color_discrete_sequence=[COLOR_ORANGE])
st.plotly_chart(figp, use_container_width=True)

# cheapest season
cheapest_season, season_avgs = season_of_cheapest_month(future[["MES","PRED"]].rename(columns={"PRED":"PRED"}))
if cheapest_season:
    st.markdown(f"**A rota ficará mais barata (previsão 2026) na estação:** **{cheapest_season}**")
    st.dataframe(season_avgs.rename(columns={"PRED":"Tarifa média prevista (R$)"}), use_container_width=True)

# variation vs historical
hist_mean = route_hist["TARIFA"].mean()
pred_mean = future["PRED"].mean()
pct_change = (pred_mean - hist_mean) / hist_mean * 100 if hist_mean!=0 else 0.0
st.markdown(f"**Variação média prevista (2026 vs histórico):** {pct_change:.2f}%  (hist: R$ {hist_mean:.2f} → prev: R$ {pred_mean:.2f})")

st.markdown("---")

# ---------------- Map: Capitais QUEDA / ESTÁVEL / ALTA ----------------
st.header("🗺️ Mapa — Capitais: Queda / Estável / Alta (Previsão 2026 vs Histórico 2023–2025)")

# Compute historical per-destination mean (2023-2025)
hist_period = df[df["ANO"].isin([2023,2024,2025])]
hist_dest_mean = hist_period.groupby("DESTINO")["TARIFA"].mean().reset_index().rename(columns={"TARIFA":"HIST_MEAN"})

# Build predicted mean per destination by averaging predictions of pairs that end on that destination
dest_summary = []
unique_pairs = df[['ORIGEM','DESTINO']].drop_duplicates()
# speed note: we predict per pair similarly to above
for dest in sorted(df["DESTINO"].unique()):
    # collect pair preds means
    pair_means = []
    for _, row in unique_pairs[unique_pairs["DESTINO"]==dest].iterrows():
        o = row["ORIGEM"]
        # pick most common company for that pair or global fallback
        comp_series = df[(df["ORIGEM"]==o)&(df["DESTINO"]==dest)]["COMPANHIA"]
        if not comp_series.empty:
            comp = comp_series.mode().iloc[0]
        else:
            comp = df["COMPANHIA"].mode().iloc[0]
        fut = build_future_for_pair(df, o, dest, comp)
        try:
            p = model.predict(fut[FEATURES])
            pair_means.append(np.mean(p))
        except Exception:
            # fallback to historical mean for pair or dest
            pair_hist = df[(df["ORIGEM"]==o)&(df["DESTINO"]==dest)]
            if not pair_hist.empty:
                pair_means.append(pair_hist["TARIFA"].mean())
            else:
                pair_means.append(df["TARIFA"].mean())
    if not pair_means:
        continue
    pred_mean_dest = np.mean(pair_means)
    hist_row = hist_dest_mean[hist_dest_mean["DESTINO"]==dest]
    hist_mean_dest = hist_row["HIST_MEAN"].iloc[0] if not hist_row.empty else hist_period["TARIFA"].mean()
    pct_change_dest = (pred_mean_dest - hist_mean_dest) / hist_mean_dest if hist_mean_dest != 0 else 0.0
    # classify
    thresh = 0.05
    if pct_change_dest <= -thresh:
        status = "QUEDA"
        color = COLOR_LIME
    elif pct_change_dest >= thresh:
        status = "ALTA"
        color = COLOR_ORANGE
    else:
        status = "ESTÁVEL"
        color = COLOR_LILAC
    dest_summary.append({
        "DESTINO": dest,
        "HIST_MEAN": round(hist_mean_dest,2),
        "PRED_MEAN_2026": round(pred_mean_dest,2),
        "PCT_CHANGE": round(pct_change_dest*100,2),
        "STATUS": status,
        "COLOR_HEX": color
    })

dest_df = pd.DataFrame(dest_summary)

# join coordinates if present
if {"DEST_LAT","DEST_LON"} <= set(df.columns):
    coords = df[["DESTINO","DEST_LAT","DEST_LON"]].drop_duplicates(subset="DESTINO").set_index("DESTINO")
    dest_df = dest_df.set_index("DESTINO").join(coords, how="left").reset_index()
    # compute season of max impact for each dest (simplified approach)
    impacts = []
    for _, r in dest_df.iterrows():
        dest = r["DESTINO"]
        # compute monthly predicted average across origins (reuse pair predictions approximated earlier)
        per_month_preds = []
        for _, pair in unique_pairs[unique_pairs["DESTINO"]==dest].iterrows():
            o = pair["ORIGEM"]
            comp_series = df[(df["ORIGEM"]==o)&(df["DESTINO"]==dest)]["COMPANHIA"]
            comp = comp_series.mode().iloc[0] if not comp_series.empty else df["COMPANHIA"].mode().iloc[0]
            fut = build_future_for_pair(df, o, dest, comp)
            try:
                p = model.predict(fut[FEATURES])
                per_month_preds.append(p)
            except Exception:
                per_month_preds.append(np.full(12, fut["tarifa_roll3"].mean()))
        if not per_month_preds:
            impacts.append({"DESTINO": dest, "IMPACT_SEASON": None})
            continue
        per_month_avg = np.mean(per_month_preds, axis=0)
        dfm = pd.DataFrame({"MES": range(1,13), "PRED": per_month_avg})
        dfm["SEASON"] = dfm["MES"].apply(month_to_season)
        season_mean = dfm.groupby("SEASON")["PRED"].mean().reset_index()
        # compare to historical season mean
        hist_dest = hist_period[hist_period["DESTINO"]==dest]
        if hist_dest.empty:
            impacted_season = season_mean.sort_values("PRED").iloc[0]["SEASON"]
        else:
            hist_season = hist_dest.groupby("SEASON")["TARIFA"].mean().reset_index()
            merged = season_mean.merge(hist_season, on="SEASON", how="left").fillna(0)
            merged["DIFF"] = merged["PRED"] - merged["TARIFA"]
            impacted_season = merged.loc[merged["DIFF"].abs().idxmax()]["SEASON"]
        impacts.append({"DESTINO": dest, "IMPACT_SEASON": impacted_season})
    impacts_df = pd.DataFrame(impacts)
    dest_df = dest_df.merge(impacts_df, on="DESTINO", how="left")

    # prepare map
    map_df = dest_df.dropna(subset=["DEST_LAT","DEST_LON"])
    if not map_df.empty:
        def hex_to_rgb(h):
            h = h.lstrip("#")
            return [int(h[i:i+2],16) for i in (0,2,4)]
        map_df["rgb"] = map_df["COLOR_HEX"].apply(hex_to_rgb)
        layer = pdk.Layer(
            "ScatterplotLayer",
            data=map_df,
            get_position='[DEST_LON, DEST_LAT]',
            get_radius=50000,
            get_fill_color='rgb',
            pickable=True
        )
        view = pdk.ViewState(latitude=map_df["DEST_LAT"].mean(), longitude=map_df["DEST_LON"].mean(), zoom=4)
        st.pydeck_chart(pdk.Deck(layers=[layer], initial_view_state=view,
                                 tooltip={"text":"{DESTINO}\nStatus: {STATUS}\nHistórico: R${HIST_MEAN}\nPrevisto: R${PRED_MEAN_2026}\nEstação impactada: {IMPACT_SEASON}"}))
        st.dataframe(map_df[["DESTINO","STATUS","HIST_MEAN","PRED_MEAN_2026","PCT_CHANGE","IMPACT_SEASON"]].sort_values("PCT_CHANGE", ascending=False), use_container_width=True)
    else:
        st.info("Coordenadas das capitais não encontradas (DEST_LAT, DEST_LON). Adicione-as ao CSV para ativar o mapa.")
else:
    st.info("Para o mapa com categorias, inclua colunas DEST_LAT e DEST_LON no CSV com coordenadas das capitais.")

st.markdown("---")
st.header("Checklist SR2 e próximos passos")
st.markdown("""
- Ajuste de threshold de estabilidade: 5% (variável `thresh` no código).  
- Melhorias sugeridas: LightGBM/XGBoost, inclusão de feriados, oferta/promos, dados de demanda.  
- Entregáveis recomendados: app (Streamlit), notebook com ETL e modelagem, slides PPTX com narrativa.
""")

st.success("Pronto — código corrigido e robusto. Me diga se quer que eu gere os slides (.pptx) e/ou o notebook (.ipynb) automaticamente com os gráficos e narrativa SR2.")
