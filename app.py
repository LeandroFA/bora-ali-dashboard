# app.py — Bora Alí (Versão Profissional 2.0)
# Atualizações: paleta LARANJA / LILÁS / VERDE-LIMÃO
# Fluxo de previsão: ORIGEM -> DESTINO -> (mostrar tarifa média histórica e previsão 2026)
# Mapa: indica capitais em Queda / Estável / Alta e mostra estação mais impactada
#
# Arquivo esperado: INMET_ANAC_EXTREMAMENTE_REDUZIDO.csv
# Colunas obrigatórias: COMPANHIA, ANO, MES, ORIGEM, DESTINO, TARIFA, TEMP_MEDIA
# Colunas opcionais: DEST_LAT, DEST_LON (para mapa)
#
# Dependências:
# pip install streamlit pandas numpy scikit-learn plotly pydeck joblib openpyxl python-pptx

import streamlit as st
import pandas as pd
import numpy as np
import os
import joblib
import warnings
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import GroupKFold, cross_val_score
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import plotly.express as px
import pydeck as pdk

warnings.filterwarnings("ignore")

# ---------------- page config & styles ----------------
st.set_page_config(page_title="Bora Alí — SR2 (Profissional v2)", layout="wide")
st.markdown("""
<style>
.neon-title { font-size:34px; font-weight:800; color:#A94EFF; }
.sub { color:#6d6d6d; }
.card { padding:12px; border-radius:10px; background:linear-gradient(180deg, rgba(255,255,255,0.02), rgba(255,255,255,0.01)); box-shadow:0 6px 20px rgba(0,0,0,0.12); }
</style>
""", unsafe_allow_html=True)

# ---------------- helpers ----------------
def month_to_season(m:int)->str:
    if m in [12,1,2]: return "VERÃO"
    if m in [3,4,5]: return "OUTONO"
    if m in [6,7,8]: return "INVERNO"
    return "PRIMAVERA"

@st.cache_data
def load_dataset(path="INMET_ANAC_EXTREMAMENTE_REDUZIDO.csv"):
    if not os.path.exists(path):
        raise FileNotFoundError(f"{path} não encontrado. Coloque o CSV na raiz do app ou faça upload pela sidebar.")
    df = pd.read_csv(path)
    required = {"COMPANHIA","ANO","MES","ORIGEM","DESTINO","TARIFA","TEMP_MEDIA"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"CSV está faltando colunas: {missing}")
    # Tipagem segura
    df["ANO"] = df["ANO"].astype(int)
    df["MES"] = df["MES"].astype(int)
    df["COMPANHIA"] = df["COMPANHIA"].astype(str)
    df["ORIGEM"] = df["ORIGEM"].astype(str)
    df["DESTINO"] = df["DESTINO"].astype(str)
    df["TARIFA"] = pd.to_numeric(df["TARIFA"], errors="coerce")
    df["TEMP_MEDIA"] = pd.to_numeric(df["TEMP_MEDIA"], errors="coerce")
    df["SEASON"] = df["MES"].apply(month_to_season)
    df["PERIODO"] = df["ANO"].astype(str) + "-" + df["MES"].astype(str).str.zfill(2)
    if "ROTA" not in df.columns:
        df["ROTA"] = df["ORIGEM"] + " → " + df["DESTINO"]
    return df

def train_model(df_train, features, target="TARIFA"):
    # Preproc
    cat_cols = ["ORIGEM","DESTINO","COMPANHIA"]
    preproc = ColumnTransformer([("cat", OneHotEncoder(handle_unknown="ignore", sparse=False), cat_cols)], remainder="passthrough")
    model = Pipeline([("pre", preproc), ("gbr", GradientBoostingRegressor(n_estimators=400, learning_rate=0.05, max_depth=4, random_state=42))])
    X = df_train[features].copy()
    y = df_train[target].values
    groups = df_train["ROTA"].values
    gkf = GroupKFold(n_splits=5)
    st.info("Executando validação cruzada (GroupKFold) — aguarde...")
    scores = -cross_val_score(model, X, y, cv=gkf.split(X,y,groups=groups), scoring="neg_mean_absolute_error", n_jobs=1)
    cv_mae = scores.mean()
    # treina no dataset completo
    model.fit(X,y)
    return model, cv_mae

def create_future_rows_for_pair(df, origem, destino, companhia, months=range(1,13)):
    # TEMP_MEDIA per month from historical pair fallback to overall destination mean
    pair = df[(df["ORIGEM"]==origem) & (df["DESTINO"]==destino)]
    dest_month_temp = pair.groupby("MES")["TEMP_MEDIA"].mean().reindex(months)
    if dest_month_temp.isnull().all():
        # fallback to destination aggregated
        dest_month_temp = df[df["DESTINO"]==destino].groupby("MES")["TEMP_MEDIA"].mean().reindex(months)
    dest_month_temp.fillna(df["TEMP_MEDIA"].mean(), inplace=True)
    rows = []
    for m in months:
        rows.append({
            "ANO":2026,
            "MES":m,
            "ORIGEM":origem,
            "DESTINO":destino,
            "COMPANHIA":companhia,
            "TEMP_MEDIA": dest_month_temp.loc[m] if m in dest_month_temp.index else df["TEMP_MEDIA"].mean()
        })
    fut = pd.DataFrame(rows)
    fut["month_sin"] = np.sin(2*np.pi*fut["MES"]/12)
    fut["month_cos"] = np.cos(2*np.pi*fut["MES"]/12)
    return fut

def season_of_cheapest_month(pred_months):
    # pred_months: dict month->predvalue or dataframe with MES and PRED
    dfm = pd.DataFrame(pred_months)
    dfm["SEASON"] = dfm["MES"].apply(month_to_season)
    s = dfm.groupby("SEASON")["PRED"].mean().reset_index()
    s = s.sort_values("PRED")
    return s.iloc[0]["SEASON"], s

# ---------------- load data ----------------
try:
    df = load_dataset()
except Exception as e:
    st.error(str(e))
    st.stop()

# --------------- HEADER & filtros globais ---------------
st.title("Bora Alí — SR2 (Profissional v2)")
st.caption("Cores: LARANJA • LILÁS • VERDE-LIMÃO — análises sazonais e previsão 2026 por rota")

# sidebar upload / overrides
st.sidebar.header("Configurações")
if st.sidebar.checkbox("Subir outro CSV (sobrescrever)", value=False):
    uploaded = st.sidebar.file_uploader("Enviar INMET_ANAC_EXTREMAMENTE_REDUZIDO.csv", type=["csv"])
    if uploaded:
        uploaded.save("INMET_ANAC_EXTREMAMENTE_REDUZIDO.csv")
        st.experimental_rerun()

# global filters quick (keeps app responsivo)
years = sorted(df["ANO"].unique())
sel_years = st.sidebar.multiselect("Ano(s) (filtro exploratório)", options=years, default=years)
sel_seasons = st.sidebar.multiselect("Estação(s)", options=["VERÃO","OUTONO","INVERNO","PRIMAVERA"], default=["VERÃO","OUTONO","INVERNO","PRIMAVERA"])
df_filtered = df[(df["ANO"].isin(sel_years)) & (df["SEASON"].isin(sel_seasons))]

# colors
COLOR_ORANGE = "#FF8A33"   # alta
COLOR_LILAC  = "#A94EFF"   # estável
COLOR_LIME   = "#8BFF66"   # queda

# ---------------- KPIs ----------------
c1,c2,c3 = st.columns(3)
c1.metric("Registros (filtro)", f"{len(df_filtered):,}")
c2.metric("Tarifa média (filtro) R$", f"{df_filtered['TARIFA'].mean():.2f}")
c3.metric("Rotas únicas (filtro)", f"{df_filtered['ROTA'].nunique():,}")

st.markdown("---")

# ---------------- Previsão interativa: origem -> destino -> rota ----------------
st.header("🔮 Previsão por ROTA (fluxo: ORIGEM → DESTINO → Análise)")
# Step 1: escolher origem
origens = sorted(df["ORIGEM"].unique())
origem_sel = st.selectbox("1) Escolha a ORIGEM", options=["-- escolha origem --"] + origens)
if origem_sel == "-- escolha origem --":
    st.info("Selecione a origem para prosseguir.")
    st.stop()

# Step 2: destinos disponíveis para essa origem
destinos_for_origem = sorted(df[df["ORIGEM"]==origem_sel]["DESTINO"].unique())
if len(destinos_for_origem) == 0:
    st.error("Nenhum destino encontrado para essa origem.")
    st.stop()
dest_sel = st.selectbox("2) Escolha o DESTINO", options=["-- escolha destino --"] + destinos_for_origem)
if dest_sel == "-- escolha destino --":
    st.info("Selecione o destino para prosseguir.")
    st.stop()

# Show basic route metrics
route_df = df[(df["ORIGEM"]==origem_sel) & (df["DESTINO"]==dest_sel)].copy()
if route_df.empty:
    st.error("Não há histórico para essa rota.")
    st.stop()

st.subheader(f"Rota: {origem_sel} → {dest_sel}")
st.write("Tarifa média histórica (todos anos disponíveis):", f"R$ {route_df['TARIFA'].mean():.2f}")
st.write("Companhias presentes na rota:", ", ".join(sorted(route_df["COMPANHIA"].unique())))

# Select companhia to model for this route (default: most common)
companias_route = sorted(route_df["COMPANHIA"].unique())
companhia_default = route_df["COMPANHIA"].mode().iloc[0]
companhia_sel = st.selectbox("3) Escolha a COMPANHIA (para previsão) — default = mais frequente", options=companias_route, index=companias_route.index(companhia_default))

# Build and train model on entire dataset (robust) once; cache to speed up
FEATURES = ["ANO","MES","ORIGEM","DESTINO","COMPANHIA","TEMP_MEDIA","month_sin","month_cos","tarifa_roll3"]

@st.cache_resource
def build_and_train_global_model(df_full):
    df_m = df_full.copy().sort_values(["ROTA","ANO","MES"])
    df_m["month_sin"] = np.sin(2*np.pi*df_m["MES"]/12)
    df_m["month_cos"] = np.cos(2*np.pi*df_m["MES"]/12)
    df_m["tarifa_roll3"] = df_m.groupby("ROTA")["TARIFA"].transform(lambda x: x.rolling(3, min_periods=1).mean())
    # drop rows with NaN TARIFA
    df_m = df_m[~df_m["TARIFA"].isna()]
    model, cv_mae = train_model(df_m, FEATURES, target="TARIFA")
    return model, cv_mae, df_m

with st.spinner("Construindo e treinando modelo global (uma vez)..."):
    model, cv_mae, df_model_ready = build_and_train_global_model(df)

st.success(f"Modelo pronto — CV MAE (GroupKFold por rota): {cv_mae:.2f} R$")

# Create future rows for the selected pair & company
future_rows = create_future_rows_for_pair(df, origem_sel, dest_sel, companhia_sel, months=range(1,13))
# add tarifa_roll3 fallback using historical monthly avg for route
route_month_mean = route_df.groupby("MES")["TARIFA"].mean().reindex(range(1,13))
future_rows["tarifa_roll3"] = future_rows["MES"].map(lambda m: route_month_mean.get(m, np.nan))
future_rows["tarifa_roll3"].fillna(route_df["TARIFA"].mean(), inplace=True)

X_future = future_rows[["ANO","MES","ORIGEM","DESTINO","COMPANHIA","TEMP_MEDIA","month_sin","month_cos","tarifa_roll3"]]
preds = model.predict(X_future)
future_rows["PRED"] = preds

# show predicted table and plot
st.subheader("Previsão mensal 2026 (R$) — rota selecionada")
st.dataframe(future_rows[["MES","PRED"]].rename(columns={"MES":"Mês","PRED":"Tarifa Prevista (R$)"}), use_container_width=True)

fig_pred = px.line(future_rows, x="MES", y="PRED", markers=True, title=f"Previsão mensal 2026 — {origem_sel} → {dest_sel}",
                   labels={"PRED":"Tarifa prevista (R$)","MES":"Mês"},
                   color_discrete_sequence=[COLOR_ORANGE])
st.plotly_chart(fig_pred, use_container_width=True)

# cheapest season in prediction
months_df = future_rows[["MES","PRED"]].rename(columns={"PRED":"PRED"})
cheapest_season, season_avgs = season_of_cheapest_month(months_df)
st.markdown(f"**A rota será mais barata em (segundo previsão 2026):** **{cheapest_season}**")
st.write("Média prevista por estação (2026):")
st.dataframe(season_avgs.rename(columns={"PRED":"Tarifa média prevista (R$)"}), use_container_width=True)

st.markdown("---")

# ---------------- Map of Capitals: comparar previsão 2026 vs histórico 2023-2025 ----------------
st.header("🗺️ Mapa: Capitais — Queda / Estável / Alta (Previsão 2026 vs Histórico 2023–2025)")

# Prepare per-destination historical mean (2023-2025)
hist_period = df[df["ANO"].isin([2023,2024,2025])]
hist_dest_mean = hist_period.groupby("DESTINO")["TARIFA"].mean().rename("HIST_MEAN").reset_index()

# For every (ORIGEM, DESTINO) present, build future predictions and then aggregate per DESTINO
# We'll average predicted monthly mean across distinct origin pairs to compute DESTINO predicted mean
unique_pairs = df[['ORIGEM','DESTINO']].drop_duplicates()

dest_preds = {}
for dest in df["DESTINO"].unique():
    dest_preds[dest] = []

# For performance: determine most common company per pair (fallback overall mode)
pair_comp_mode = df.groupby(["ORIGEM","DESTINO"])["COMPANHIA"].agg(lambda x: x.mode().iloc[0] if not x.mode().empty else df["COMPANHIA"].mode().iloc[0]).reset_index()

for _, row in unique_pairs.iterrows():
    o = row["ORIGEM"]; d = row["DESTINO"]
    comp_row = pair_comp_mode[(pair_comp_mode["ORIGEM"]==o) & (pair_comp_mode["DESTINO"]==d)]
    comp = comp_row["COMPANHIA"].iloc[0] if not comp_row.empty else df["COMPANHIA"].mode().iloc[0]
    fut = create_future_rows_for_pair(df, o, d, comp, months=range(1,13))
    # tarifa_roll3
    pair_hist = df[(df["ORIGEM"]==o)&(df["DESTINO"]==d)].sort_values(["ANO","MES"])
    pair_roll = pair_hist.groupby("MES")["TARIFA"].mean().reindex(range(1,13))
    fut["tarifa_roll3"] = fut["MES"].map(lambda m: pair_roll.get(m, np.nan))
    fut["tarifa_roll3"].fillna(pair_hist["TARIFA"].mean() if not pair_hist.empty else df["TARIFA"].mean(), inplace=True)
    Xf = fut[["ANO","MES","ORIGEM","DESTINO","COMPANHIA","TEMP_MEDIA","month_sin","month_cos","tarifa_roll3"]]
    try:
        pf = model.predict(Xf)
    except Exception:
        pf = np.full(len(Xf), Xf["tarifa_roll3"].mean())  # fallback
    # mean predicted for this pair (average monthly)
    mean_pair_pred = pf.mean()
    dest_preds[d].append(mean_pair_pred)

# Aggregate per destination: mean of pair means
dest_summary = []
for dest, preds_list in dest_preds.items():
    if len(preds_list)==0:
        continue
    mean_pred_dest = np.mean(preds_list)
    # historical mean (if missing, fallback to overall)
    hist_val = hist_dest_mean[hist_dest_mean["DESTINO"]==dest]["HIST_MEAN"]
    hist_val = hist_val.iloc[0] if not hist_val.empty else hist_period["TARIFA"].mean()
    # percent change
    pct_change = (mean_pred_dest - hist_val) / hist_val if hist_val != 0 else 0.0
    # classify with threshold 5%
    thresh = 0.05
    if pct_change <= -thresh:
        status = "QUEDA"
        color = COLOR_LIME
    elif pct_change >= thresh:
        status = "ALTA"
        color = COLOR_ORANGE
    else:
        status = "ESTÁVEL"
        color = COLOR_LILAC
    dest_summary.append({
        "DESTINO": dest,
        "HIST_MEAN": round(hist_val,2),
        "PRED_MEAN_2026": round(mean_pred_dest,2),
        "PCT_CHANGE": round(pct_change*100,2),
        "STATUS": status,
        "COLOR": color
    })

dest_summary_df = pd.DataFrame(dest_summary).sort_values("PCT_CHANGE", ascending=False)

# if lat/lon present, join coords for map
if {"DEST_LAT","DEST_LON"} <= set(df.columns):
    coords = df[["DESTINO","DEST_LAT","DEST_LON"]].drop_duplicates(subset=["DESTINO"]).set_index("DESTINO")
    dest_summary_df = dest_summary_df.set_index("DESTINO").join(coords, how="left").reset_index()
    # Also compute which season shows largest increase/decrease per destination (based on predicted months aggregated by season)
    season_impacts = []
    for idx, row in dest_summary_df.iterrows():
        dest = row["DESTINO"]
        # Build predicted months for dest by averaging pair predictions per month (approx)
        per_month_preds = []
        # recompute monthly predictions across pairs to detect season of max change — simplified by using future for each origin pair
        for _, pair in unique_pairs[unique_pairs["DESTINO"]==dest].iterrows():
            o = pair["ORIGEM"]
            comp = pair_comp_mode[(pair_comp_mode["ORIGEM"]==o)&(pair_comp_mode["DESTINO"]==dest)]["COMPANHIA"]
            comp = comp.iloc[0] if not comp.empty else df["COMPANHIA"].mode().iloc[0]
            fut = create_future_rows_for_pair(df, o, dest, comp, months=range(1,13))
            pair_roll = df[(df["ORIGEM"]==o)&(df["DESTINO"]==dest)].groupby("MES")["TARIFA"].mean().reindex(range(1,13))
            fut["tarifa_roll3"] = fut["MES"].map(lambda m: pair_roll.get(m, np.nan))
            fut["tarifa_roll3"].fillna(df["TARIFA"].mean(), inplace=True)
            try:
                pvals = model.predict(fut[["ANO","MES","ORIGEM","DESTINO","COMPANHIA","TEMP_MEDIA","month_sin","month_cos","tarifa_roll3"]])
            except Exception:
                pvals = np.full(12, fut["tarifa_roll3"].mean())
            per_month_preds.append(pvals)
        if len(per_month_preds)==0:
            continue
        per_month_avg = np.mean(per_month_preds, axis=0)
        # map to seasons
        months = np.arange(1,13)
        df_m = pd.DataFrame({"MES":months, "PRED":per_month_avg})
        df_m["SEASON"] = df_m["MES"].apply(month_to_season)
        season_mean = df_m.groupby("SEASON")["PRED"].mean().reset_index()
        # pick season with highest predicted increase or lowest value depending — user asked "indique em qual estação do ano isso vai ocorrer"
        # we'll report season with largest absolute change relative to historical season mean
        # compute historical season means for this dest
        hist_dest = hist_period[hist_period["DESTINO"]==dest]
        if hist_dest.empty:
            impacted_season = season_mean.sort_values("PRED").iloc[0]["SEASON"]
        else:
            hist_season = hist_dest.groupby("SEASON")["TARIFA"].mean().reset_index()
            merged = season_mean.merge(hist_season, on="SEASON", how="left").fillna(method="ffill")
            merged["DIFF"] = merged["PRED"] - merged["TARIFA"]
            # pick season with max absolute diff
            impacted_season = merged.loc[merged["DIFF"].abs().idxmax()]["SEASON"]
        season_impacts.append({"DESTINO":dest, "IMPACT_SEASON": impacted_season})
    season_impacts_df = pd.DataFrame(season_impacts)
    dest_summary_df = dest_summary_df.merge(season_impacts_df, on="DESTINO", how="left")
    # Map rendering
    st.subheader("Mapa — categorias por capital (previsão 2026 vs histórico 2023-2025)")
    map_df = dest_summary_df.dropna(subset=["DEST_LAT","DEST_LON"])
    if map_df.empty:
        st.info("Não foram encontradas coordenadas únicas para as capitais.")
    else:
        # create color by status
        def hex_to_rgb(h):
            h = h.lstrip("#")
            return [int(h[i:i+2],16) for i in (0,2,4)]
        map_df["color_rgb"] = map_df["COLOR"].apply(hex_to_rgb)
        layer = pdk.Layer(
            "ScatterplotLayer",
            data=map_df,
            get_position='[DEST_LON, DEST_LAT]',
            get_fill_color='color_rgb',
            get_radius=60000,
            pickable=True
        )
        deck = pdk.Deck(
            initial_view_state=pdk.ViewState(latitude=map_df["DEST_LAT"].mean(), longitude=map_df["DEST_LON"].mean(), zoom=4),
            layers=[layer],
            tooltip={"text":"{DESTINO}\nStatus: {STATUS}\nHistórico: R${HIST_MEAN}\nPrevisto 2026: R${PRED_MEAN_2026}\nImpacto na estação: {IMPACT_SEASON}"}
        )
        st.pydeck_chart(deck)

        st.markdown("Legenda: **QUEDA** (verde-limão) — **ESTÁVEL** (lilás) — **ALTA** (laranja).")
        st.dataframe(map_df[["DESTINO","STATUS","HIST_MEAN","PRED_MEAN_2026","PCT_CHANGE","IMPACT_SEASON"]].sort_values("PCT_CHANGE", ascending=False), use_container_width=True)
else:
    st.info("Para gerar mapa com categorias, adicione colunas `DEST_LAT` e `DEST_LON` no CSV com coordenadas das capitais.")

st.markdown("---")
st.header("Checklist e observações finais")
st.markdown("""
- A classificação QUEDA/ESTÁVEL/ALTA usa threshold padrão de 5% (pode ser ajustado no código - variável `thresh`).  
- A estação mais barata por rota foi calculada com base na média das previsões mensais de 2026 por estação.  
- O mapa agrega previsões por destino (média entre origens) para dar uma visão de tendência por capital.  
- Para melhorar: incluir feriados, eventos e preço por cabine (promoções) e usar LightGBM/XGBoost para acelerar.
""")

st.success("Atualização concluída — paleta e UX ajustadas. Diga se quer que eu gere agora os slides PPTX ou o notebook (.ipynb) com toda a ETL e modelagem (posso incluir gráficos e texto pronto para SR2).")
