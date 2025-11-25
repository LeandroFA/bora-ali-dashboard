# =========================================================
# ✈️ Bora Alí — SR2 Dashboard Profissional (v5 FINAL)
# =========================================================

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score
import warnings
warnings.filterwarnings("ignore")

# ---------------------------------------------------------
# 🎨 PALETA DE CORES
# ---------------------------------------------------------
COLOR_ORANGE = "#FF6B00"
COLOR_LILAC  = "#A259FF"
COLOR_LIME   = "#9EFF00"
COLOR_BLUE   = "#28F8FF"

COLORS_SEASON = {
    "VERÃO": COLOR_ORANGE,
    "OUTONO": COLOR_LILAC,
    "INVERNO": COLOR_LIME,
    "PRIMAVERA": COLOR_BLUE
}

# ---------------------------------------------------------
# 📌 FUNÇÃO PARA CARREGAR DATASET
# ---------------------------------------------------------
@st.cache_data
def load_data():
    df = pd.read_csv("INMET_ANAC_EXTREMAMENTE_REDUZIDO.csv")

    # Tipagem
    df["ANO"] = df["ANO"].astype(int)
    df["MES"] = df["MES"].astype(int)

    # Criar Estações
    df["SEASON"] = df["MES"].apply(lambda m:
        "VERÃO" if m in [12,1,2] else
        "OUTONO" if m in [3,4,5] else
        "INVERNO" if m in [6,7,8] else
        "PRIMAVERA"
    )
    return df

df = load_data()

# ---------------------------------------------------------
# 📍 ADICIONAR COORDENADAS (AUTOMATICAMENTE)
# ---------------------------------------------------------
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

if "DEST_LAT" not in df.columns or "DEST_LON" not in df.columns:
    df["DEST_LAT"] = df["DESTINO"].apply(lambda x: CAPITAL_COORDS.get(x, (np.nan,np.nan))[0])
    df["DEST_LON"] = df["DESTINO"].apply(lambda x: CAPITAL_COORDS.get(x, (np.nan,np.nan))[1])
    df.to_csv("INMET_ANAC_EXTREMAMENTE_REDUZIDO.csv", index=False)
    st.warning("📍 Coordenadas adicionadas! Atualize a página (Ctrl + F5).")

# ---------------------------------------------------------
# 🤖 TREINO GLOBAL — Previsão 2026
# ---------------------------------------------------------
@st.cache_resource
def build_model(df):
    df_m = df.copy()
    FEATURES = ["ORIGEM", "DESTINO", "COMPANHIA", "MES", "TEMP_MEDIA"]
    target = "TARIFA"

    cat_cols = ["ORIGEM", "DESTINO", "COMPANHIA"]
    preproc = ColumnTransformer([("cat", OneHotEncoder(handle_unknown="ignore", sparse=False), cat_cols)],
                                remainder="passthrough")

    model = Pipeline([
        ("prep", preproc),
        ("rf", RandomForestRegressor(n_estimators=300, random_state=42))
    ])
    model.fit(df_m[FEATURES], df_m[target])
    cv = cross_val_score(model, df_m[FEATURES], df_m[target], cv=4, scoring="neg_mean_absolute_error")
    return model, -cv.mean()

model, mae_cv = build_model(df)

# ---------------------------------------------------------
# 🧮 FUNÇÃO DE PREVISÃO POR ROTA
# ---------------------------------------------------------
def predict_route(ori, dst, cia):
    base = df[(df["ORIGEM"] == ori) & (df["DESTINO"] == dst) & (df["COMPANHIA"] == cia)]
    if base.empty:
        return None
    future = pd.DataFrame({
        "ORIGEM":[ori]*12, "DESTINO":[dst]*12, "COMPANHIA":[cia]*12,
        "MES": list(range(1,13)), "TEMP_MEDIA": base["TEMP_MEDIA"].mean()
    })
    future["PREV"] = model.predict(future)
    return future

# =========================================================
# 🖥️ LAYOUT DO DASHBOARD
# =========================================================
st.title("✈️ Bora Alí — SR2 Dashboard Profissional")
st.caption("🔎 Tarifa aérea por capitais • Sazonalidade • Previsão 2026 por rota")

# KPIs
col1, col2, col3 = st.columns(3)
col1.metric("Registros", f"{df.shape[0]:,}".replace(",","."))
col2.metric("Tarifa média (R$)", f"{df['TARIFA'].mean():.2f}")
col3.metric("Rotas únicas", df.groupby(["ORIGEM","DESTINO"]).ngroups)

# ---------------------------------------------------------
# 🌦️ GRÁFICO POR ESTAÇÃO
# ---------------------------------------------------------
season_stats = df.groupby("SEASON")["TARIFA"].mean().reset_index()
fig_season = px.bar(season_stats, x="SEASON", y="TARIFA",
                    color="SEASON", color_discrete_map=COLORS_SEASON,
                    title="🌦️ Média Tarifária por Estação")
st.plotly_chart(fig_season, use_container_width=True)

# ---------------------------------------------------------
# 🔮 PREVISÃO POR ROTA
# ---------------------------------------------------------
st.subheader("🔮 Previsão por ROTA (ORIGEM → DESTINO → COMPANHIA)")

ori = st.selectbox("Escolha a ORIGEM", sorted(df["ORIGEM"].unique()))
dst = st.selectbox("Escolha o DESTINO", sorted(df[df["ORIGEM"] == ori]["DESTINO"].unique()))
rot_df = df[(df["ORIGEM"]==ori)&(df["DESTINO"]==dst)]
cia = st.selectbox("Escolha a COMPANHIA", sorted(rot_df["COMPANHIA"].unique()))

st.info(f"Tarifa média histórica da rota **{ori} → {dst}**: **R$ {rot_df['TARIFA'].mean():.2f}**")

future = predict_route(ori, dst, cia)
if future is not None:
    future["SEASON"] = future["MES"].apply(lambda m:
        "VERÃO" if m in [12,1,2] else
        "OUTONO" if m in [3,4,5] else
        "INVERNO" if m in [6,7,8] else
        "PRIMAVERA"
    )
    best = future.loc[future["PREV"].idxlowest()]
    st.success(f"💰 **Estação mais barata em 2026:** {best['SEASON']} • **R$ {best['PREV']:.2f}**")

    fig_fore = px.line(future, x="MES", y="PREV", color="SEASON",
                       color_discrete_map=COLORS_SEASON,
                       markers=True, title=f"📈 Previsão 2026 — {ori} → {dst} ({cia})")
    st.plotly_chart(fig_fore, use_container_width=True)

# ---------------------------------------------------------
# 🗺️ MAPA DE VALORAÇÃO DAS CAPITAIS
# ---------------------------------------------------------
st.subheader("🗺️ Tendência nas Capitais (2026)")

tend = df.groupby("DESTINO")["TARIFA"].mean().reset_index()
tend.rename(columns={"TARIFA":"BASE"}, inplace=True)

# Prever média para 2026 (todas capitais)
all_preds = []
for dest in df["DESTINO"].unique():
    ori_temp = df[df["DESTINO"]==dest]["ORIGEM"].mode()[0]
    cia_temp = df[df["DESTINO"]==dest]["COMPANHIA"].mode()[0]
    pred_tmp = predict_route(ori_temp, dest, cia_temp)
    if pred_tmp is not None:
        all_preds.append({"DESTINO":dest,"PREV_2026":pred_tmp["PREV"].mean()})

pred_df = pd.DataFrame(all_preds)
tend = tend.merge(pred_df, on="DESTINO", how="inner")
tend["STATUS"] = tend.apply(lambda r: "📉 Queda" if r["PREV_2026"]<r["BASE"]-10
                            else ("📈 Alta" if r["PREV_2026"]>r["BASE"]+10 else "🔁 Estável"), axis=1)

tend = tend.merge(df[["DESTINO","DEST_LAT","DEST_LON"]].drop_duplicates(), on="DESTINO")

fig_map = px.scatter_mapbox(tend, lat="DEST_LAT", lon="DEST_LON", hover_name="DESTINO",
                            hover_data=["STATUS","BASE","PREV_2026"],
                            color="STATUS", size=np.abs(tend["PREV_2026"]-tend["BASE"])+1,
                            color_discrete_map={"📉 Queda":COLOR_LIME,"📈 Alta":COLOR_ORANGE,"🔁 Estável":COLOR_LILAC},
                            zoom=3, height=600)
fig_map.update_layout(mapbox_style="carto-positron")
st.plotly_chart(fig_map, use_container_width=True)

# =========================================================
# 📌 FIM
# =========================================================
