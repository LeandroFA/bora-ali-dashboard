# =============================================================
# ✈️ Bora Alí — SR2 Dashboard Profissional (Versão Final)
# =============================================================

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import pydeck as pdk
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import GroupKFold, cross_val_score
import math, warnings
warnings.filterwarnings("ignore")

# -------------------- OneHot UNIVERSAL (não quebra) --------------------
def OneHotSafe():
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False)  # sklearn >=1.2
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", sparse=False)         # sklearn <1.2

# -------------------- PALETA --------------------
COLOR_ORANGE = "#FF8A33"
COLOR_LILAC   = "#C77DFF"
COLOR_LIME    = "#8BFF66"
COLOR_BLUE    = "#28F8FF"

COLORS_SEASON = {"VERÃO":COLOR_ORANGE,"OUTONO":COLOR_LILAC,"INVERNO":COLOR_LIME,"PRIMAVERA":COLOR_BLUE}

# -------------------- Coordenadas memorizadas --------------------
CAPITAL_COORDS = {
"Rio Branco":(-9.97499,-67.8243),"Maceió":(-9.64985,-35.70895),"Macapá":(0.034934,-51.0694),
"Manaus":(-3.11903,-60.0217),"Salvador":(-12.9718,-38.5011),"Fortaleza":(-3.71664,-38.5423),
"Brasília":(-15.7934,-47.8823),"Vitória":(-20.3155,-40.3128),"Goiânia":(-16.6864,-49.2643),
"São Luís":(-2.53874,-44.2825),"Cuiabá":(-15.5989,-56.0949),"Campo Grande":(-20.4697,-54.6201),
"Belo Horizonte":(-19.9167,-43.9345),"Belém":(-1.45583,-48.5039),"João Pessoa":(-7.11509,-34.8641),
"Curitiba":(-25.4284,-49.2733),"Recife":(-8.04756,-34.8771),"Teresina":(-5.08921,-42.8016),
"Rio de Janeiro":(-22.9068,-43.1729),"Natal":(-5.79448,-35.211),"Porto Alegre":(-30.0346,-51.2177),
"Porto Velho":(-8.76077,-63.8999),"Boa Vista":(2.82384,-60.6753),"Florianópolis":(-27.5945,-48.5477),
"São Paulo":(-23.5505,-46.6333),"Aracaju":(-10.9472,-37.0731),"Palmas":(-10.184,-48.3336)
}

# -------------------- Funções --------------------
def month_to_season(m):
    if m in (12,1,2): return "VERÃO"
    if m in (3,4,5): return "OUTONO"
    if m in (6,7,8): return "INVERNO"
    return "PRIMAVERA"

def train_model_safe(df):
    FEATURES = ["ORIGEM","DESTINO","COMPANHIA","MES","TEMP_MEDIA","SIN","COS"]
    df = df.dropna(subset=FEATURES+["TARIFA"])
    cat = ["ORIGEM","DESTINO","COMPANHIA"]
    prep = ColumnTransformer([("cat", OneHotSafe(), cat)], remainder="passthrough")
    model = Pipeline([("prep",prep),("gbr",GradientBoostingRegressor(n_estimators=300,learning_rate=0.05,max_depth=4))])
    X = df[FEATURES]; y = df["TARIFA"].values
    groups = df["ROTA"]
    cv = GroupKFold(n_splits=4)
    scores = -cross_val_score(model,X,y,cv=cv.split(X,y,groups),scoring="neg_mean_absolute_error")
    model.fit(X,y)
    return model, scores.mean()

def build_future(df, o, d, c):
    months=range(1,13)
    base=df[(df["ORIGEM"]==o)&(df["DESTINO"]==d)&(df["COMPANHIA"]==c)]
    if base.empty: base=df[(df["ORIGEM"]==o)&(df["DESTINO"]==d)]
    temp = base.groupby("MES")["TEMP_MEDIA"].mean().reindex(months)
    temp.fillna(df["TEMP_MEDIA"].mean(), inplace=True)
    rows=[]
    for m in months:
        rows.append({"ANO":2026,"MES":m,"ORIGEM":o,"DESTINO":d,"COMPANHIA":c,
                     "TEMP_MEDIA":temp.loc[m],
                     "SIN":math.sin(2*math.pi*m/12),
                     "COS":math.cos(2*math.pi*m/12)})
    f=pd.DataFrame(rows); f["SEASON"]=f["MES"].apply(month_to_season)
    return f

# -------------------- LOAD CSV --------------------
st.set_page_config(page_title="Bora Alí — SR2",layout="wide")
CSV="INMET_ANAC_EXTREMAMENTE_REDUZIDO.csv"
df=pd.read_csv(CSV)

df["MES"]=df["MES"].astype(int)
df["ANO"]=df["ANO"].astype(int)
df["TEMP_MEDIA"]=pd.to_numeric(df["TEMP_MEDIA"],errors="coerce")
df["TARIFA"]=pd.to_numeric(df["TARIFA"],errors="coerce")
df["SEASON"]=df["MES"].apply(month_to_season)
df["ROTA"]=df["ORIGEM"]+" → "+df["DESTINO"]
df["SIN"]=np.sin(2*np.pi*df["MES"]/12)
df["COS"]=np.cos(2*np.pi*df["MES"]/12)
df["DEST_LAT"]=df["DESTINO"].apply(lambda x:CAPITAL_COORDS.get(x,(np.nan,np.nan))[0])
df["DEST_LON"]=df["DESTINO"].apply(lambda x:CAPITAL_COORDS.get(x,(np.nan,np.nan))[1])
df_map=df.dropna(subset=["DEST_LAT","DEST_LON"])

# -------------------- KPIs --------------------
st.title("✈️ Bora Alí — SR2 Dashboard")
col1,col2,col3=st.columns(3)
col1.metric("Registros",f"{df.shape[0]:,}")
col2.metric("Tarifa Média",f"R$ {df['TARIFA'].mean():.2f}")
col3.metric("Rotas Únicas",df["ROTA"].nunique())

# -------------------- Gráfico Estação --------------------
st.subheader("🌦️ Média Tarifária por Estação")
s=df.groupby("SEASON")["TARIFA"].mean().reset_index()
fig=px.bar(s,x="SEASON",y="TARIFA",color="SEASON",color_discrete_map=COLORS_SEASON)
st.plotly_chart(fig,use_container_width=True)

# -------------------- Previsão --------------------
st.subheader("🔮 Previsão da Rota (ORIGEM → DESTINO → COMPANHIA)")
o=st.selectbox("Origem",sorted(df["ORIGEM"].unique()))
d=st.selectbox("Destino",sorted(df[df["ORIGEM"]==o]["DESTINO"].unique()))
c=st.selectbox("Companhia",sorted(df[(df["ORIGEM"]==o)&(df["DESTINO"]==d)]["COMPANHIA"].unique()))

df_model=df.dropna(subset=["TARIFA","TEMP_MEDIA"])
model,mae=train_model_safe(df_model)

fut=build_future(df,o,d,c)
fut["PRED"]=model.predict(fut[["ORIGEM","DESTINO","COMPANHIA","MES","TEMP_MEDIA","SIN","COS"]])

st.success(f"Rota: {o} → {d} ({c}) | Estação mais barata: **{fut.groupby('SEASON')['PRED'].mean().idxmin()}**")

fig2=px.line(fut,x="MES",y="PRED",markers=True,color_discrete_sequence=[COLOR_ORANGE])
st.plotly_chart(fig2,use_container_width=True)

# -------------------- MAPA --------------------
st.subheader("🗺️ Mapa de Tendência — Capitais (2026 vs Histórico)")
hist=df.groupby("DESTINO")["TARIFA"].mean().reset_index().rename(columns={"TARIFA":"HIST"})
pred=[]
for dest in df["DESTINO"].unique():
    o_tmp=df[df["DESTINO"]==dest]["ORIGEM"].mode().iloc[0]
    c_tmp=df[df["DESTINO"]==dest]["COMPANHIA"].mode().iloc[0]
    f=build_future(df,o_tmp, dest, c_tmp)
    f["P"]=model.predict(f[["ORIGEM","DESTINO","COMPANHIA","MES","TEMP_MEDIA","SIN","COS"]])
    pred.append({"DESTINO":dest,"PRED":f["P"].mean()})
pred=pd.DataFrame(pred)

m=hist.merge(pred,on="DESTINO")
m["VAR"]=(m["PRED"]-m["HIST"])/m["HIST"]
def status(v):
    if v<=-0.05: return "QUEDA"
    if v>=0.05: return "ALTA"
    return "ESTÁVEL"
m["STATUS"]=m["VAR"].apply(status)
m=m.merge(df_map[["DESTINO","DEST_LAT","DEST_LON"]].drop_duplicates(),on="DESTINO")

color_map={"QUEDA":COLOR_LIME,"ALTA":COLOR_ORANGE,"ESTÁVEL":COLOR_LILAC}

layer=pdk.Layer(
"ScatterplotLayer",
data=m.dropna(subset=["DEST_LAT","DEST_LON"]),
get_position='[DEST_LON, DEST_LAT]',
get_fill_color=f"""[
    (PRED>HIST ? 255 : 0),
    (PRED<HIST ? 255 : 0),
    120, 200 ]""",
get_radius=60000,
pickable=True)

view=pdk.ViewState(latitude=-15,longitude=-55,zoom=3)
st.pydeck_chart(pdk.Deck(layers=[layer],initial_view_state=view))
st.dataframe(m[["DESTINO","STATUS","HIST","PRED","VAR"]])
