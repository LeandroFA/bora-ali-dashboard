# ================================================================
# Bora Alí — SR2 (Profissional v3)
# Dashboard com previsão 2026, sazonalidade e mapa de tendência
# Dataset: INMET_ANAC_EXTREMAMENTE_REDUZIDO.csv
# ================================================================

# -------------------- IMPORTS --------------------
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import pydeck as pdk
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GroupKFold, cross_val_score
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import GradientBoostingRegressor

# -------------------- CONFIG DE TEMA --------------------
COLOR_LILAC = "#C77DFF"
COLOR_ORANGE = "#FFA24D"
COLOR_GREEN = "#7BFF00"
COLOR_BG = "#22172B"

st.set_page_config(
    page_title="Bora Alí — SR2",
    page_icon="✈️",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown(f"""
    <style>
    body {{
        background-color: {COLOR_BG};
    }}
    .metric-card {{
        padding: 10px;
        border-radius: 12px;
        background: rgba(255,255,255,0.06);
        color: white;
        font-weight: bold;
        text-align: center;
        box-shadow: 0 4px 15px rgba(0,0,0,0.3);
    }}
    .title-main {{
        color: {COLOR_LILAC};
        font-size: 36px;
        font-weight: 800;
        text-shadow: 0 0 10px {COLOR_LILAC};
    }}
    </style>
""", unsafe_allow_html=True)

# -------------------- FUNÇÕES AUXILIARES --------------------
def month_to_season(m):
    if m in [12,1,2]: return "VERÃO"
    if m in [3,4,5]: return "OUTONO"
    if m in [6,7,8]: return "INVERNO"
    return "PRIMAVERA"

@st.cache_data
def load_data():
    df = pd.read_csv("INMET_ANAC_EXTREMAMENTE_REDUZIDO.csv")
    df["SEASON"] = df["MES"].apply(month_to_season)
    return df

def train_model(df_train, features, target="TARIFA"):
    cat_cols = ["ORIGEM","DESTINO","COMPANHIA"]
    preproc = ColumnTransformer(
        [("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), cat_cols)],
        remainder="passthrough"
    )
    model = Pipeline([
        ("pre", preproc),
        ("gbr", GradientBoostingRegressor(n_estimators=400, learning_rate=0.05, max_depth=4))
    ])
    X = df_train[features].copy()
    y = df_train[target].values
    groups = df_train["ROTA"].values
    gkf = GroupKFold(n_splits=5)
    scores = -cross_val_score(model, X, y, cv=gkf.split(X, y, groups), scoring="neg_mean_absolute_error")
    cv_mae = scores.mean()
    model.fit(X, y)
    return model, cv_mae

def season_of_cheapest(future):
    sgroup = future.groupby("SEASON")["PRED"].mean().sort_values()
    return sgroup.index[0], sgroup

# -------------------- CARREGAR DATA --------------------
df = load_data()
df["ROTA"] = df["ORIGEM"] + " → " + df["DESTINO"]

# -------------------- FILTROS --------------------
st.sidebar.title("Filtros Inteligentes")

MES_NAMES = {
    1:"Janeiro",2:"Fevereiro",3:"Março",4:"Abril",5:"Maio",6:"Junho",
    7:"Julho",8:"Agosto",9:"Setembro",10:"Outubro",11:"Novembro",12:"Dezembro"
}
sel_months_names = st.sidebar.multiselect("Mês", list(MES_NAMES.values()), default=list(MES_NAMES.values()))
sel_months = [k for k,v in MES_NAMES.items() if v in sel_months_names]

sel_years = st.sidebar.multiselect("Ano", sorted(df["ANO"].unique()), default=sorted(df["ANO"].unique()))
sel_capitais = st.sidebar.multiselect("Destino (Capitais)", sorted(df["DESTINO"].unique()), default=sorted(df["DESTINO"].unique()))
sel_comp = st.sidebar.multiselect("Companhias", sorted(df["COMPANHIA"].unique()), default=sorted(df["COMPANHIA"].unique()))
sel_season = st.sidebar.multiselect("Estação", ["VERÃO","OUTONO","INVERNO","PRIMAVERA"], default=["VERÃO","OUTONO","INVERNO","PRIMAVERA"])

df_filtered = df[
    (df["ANO"].isin(sel_years)) & (df["MES"].isin(sel_months)) &
    (df["DESTINO"].isin(sel_capitais)) & (df["COMPANHIA"].isin(sel_comp)) &
    (df["SEASON"].isin(sel_season))
]

# -------------------- KPI --------------------
st.markdown("<div class='title-main'>✈️ Bora Alí — SR2 Dashboard Profissional</div>", unsafe_allow_html=True)
col1, col2, col3 = st.columns(3)
col1.markdown(f"<div class='metric-card'>Registros<br>{len(df_filtered):,}</div>", unsafe_allow_html=True)
col2.markdown(f"<div class='metric-card'>Tarifa média (R$)<br>{df_filtered['TARIFA'].mean():.2f}</div>", unsafe_allow_html=True)
col3.markdown(f"<div class='metric-card'>Rotas únicas<br>{df_filtered['ROTA'].nunique()}</div>", unsafe_allow_html=True)

# -------------------- SAZONALIDADE --------------------
st.subheader("🌦️ Média Tarifária por Estação")
season_stats = df_filtered.groupby("SEASON")["TARIFA"].mean().reset_index()
fig_season = px.bar(season_stats, x="SEASON", y="TARIFA", color="SEASON",
                    color_discrete_sequence=[COLOR_LILAC, COLOR_ORANGE, COLOR_GREEN, "#28F8FF"])
st.plotly_chart(fig_season, use_container_width=True)

# -------------------- PREVISÃO 2026 --------------------
st.subheader("🔮 Previsão por ROTA (ORIGEM → DESTINO → COMPANHIA)")
origem_sel = st.selectbox("Escolha a ORIGEM", sorted(df["ORIGEM"].unique()))
dest_sel = st.selectbox("Escolha o DESTINO", sorted(df[df["ORIGEM"]==origem_sel]["DESTINO"].unique()))
rota_df = df[(df["ORIGEM"]==origem_sel)&(df["DESTINO"]==dest_sel)]
st.write(f"Tarifa média histórica da rota **{origem_sel} → {dest_sel}**: **R$ {rota_df['TARIFA'].mean():.2f}**")
sel_company = st.selectbox("Escolha a COMPANHIA (para previsão)", sorted(rota_df["COMPANHIA"].unique()), index=0)

# -------------------- TREINAR MODELO --------------------
df_m = df.copy()
FEATURES = ["ANO","MES","ORIGEM","DESTINO","COMPANHIA","TEMP_MEDIA"]
model, cv_mae = train_model(df_m, FEATURES)

future = pd.DataFrame({"ANO":2026,"MES":range(1,13)})
future["ORIGEM"]=origem_sel
future["DESTINO"]=dest_sel
future["COMPANHIA"]=sel_company
temp_month = df_m[df_m["DESTINO"]==dest_sel].groupby("MES")["TEMP_MEDIA"].mean().reindex(range(1,13)).bfill()
future["TEMP_MEDIA"]=temp_month.values
future["PRED"]=model.predict(future)
future["SEASON"]=future["MES"].apply(month_to_season)

cheapest_season, season_values = season_of_cheapest(future)

st.success(f"💸 **Estação mais barata prevista em 2026 para {origem_sel} → {dest_sel}: _{cheapest_season}_**")

fig_pred = px.line(future, x="MES", y="PRED", markers=True,
                   title=f"📈 Tarifas previstas (2026) — {origem_sel} → {dest_sel}",
                   labels={"PRED":"Tarifa (R$)","MES":"Mês"},
                   color_discrete_sequence=[COLOR_ORANGE])
st.plotly_chart(fig_pred, use_container_width=True)

# -------------------- MAPA --------------------
st.subheader("🗺️ Tendência por Capital em 2026 (queda, estável, alta)")
if "DEST_LAT" in df.columns:
    pred_map = df_m.groupby(["DESTINO","DEST_LAT","DEST_LON"])["TARIFA"].mean().reset_index()
    pred_map["PRED2026"] = model.predict(df_m.drop(columns=["ROTA"])[FEATURES].sample(n=len(pred_map), replace=True))
    pred_map["VAR"] = pred_map["PRED2026"] - pred_map["TARIFA"]
    pred_map["STATUS"] = pred_map["VAR"].apply(lambda x: "📉 Queda" if x<0 else ("📈 Alta" if x>50 else "➖ Estável"))
    COLOR_MAP = {"📉 Queda":COLOR_GREEN,"➖ Estável":"#FFFFFF","📈 Alta":COLOR_ORANGE}
    pred_map["COLOR"]=pred_map["STATUS"].map(COLOR_MAP)

    layer = pdk.Layer(
        "ScatterplotLayer",
        pred_map,
        get_position='[DEST_LON, DEST_LAT]',
        get_radius=45000,
        get_fill_color="COLOR",
        pickable=True
    )
    view = pdk.ViewState(latitude=-14, longitude=-51, zoom=3.3)
    st.pydeck_chart(pdk.Deck(layers=[layer], initial_view_state=view))
else:
    st.warning("⚠ Seu dataset não possui DEST_LAT e DEST_LON. Adicione coordenadas para ativar o mapa.")

# ================================================================
# FIM DO APP
# ================================================================
