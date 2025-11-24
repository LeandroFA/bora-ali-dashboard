# =============================================================================
#  Bora Alí — Dashboard Urbano (Laranja Sunset) FINAL COMPLETO
#  SR2 — 7 visualizações + Previsão 2026 + Ranking por Destino
# =============================================================================

# ---------------------------------------------
# IMPORTS
# ---------------------------------------------
import unicodedata
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from prophet import Prophet
from prophet.plot import plot_plotly

# ---------------------------------------------
# CONFIGURAÇÃO DA TELA
# ---------------------------------------------
st.set_page_config(
    page_title="Bora Alí — Dashboard Urbano",
    layout="wide",
    page_icon="🧳"
)

CSV_FILE = "INMET_ANAC_ROTAS_APENAS_CAPITAIS.csv"

# 🎨 PALETA LARANJA SUNSET
ORANGE = "#FF6A00"
PURPLE = "#6328E0"
SOFT = "#FFD199"
BG = "#FDFBFA"

# ---------------------------------------------
# ESTILO CSS GLOBAL
# ---------------------------------------------
st.markdown(f"""
<style>
body {{
    background-color:{BG};
}}
h1,h2,h3,h4,h5 {{
    color:{PURPLE};
    font-weight:800;
}}
.stButton>button {{
    background:{ORANGE};
    color:white;
    border-radius:8px;
    font-weight:700;
}}
</style>
""", unsafe_allow_html=True)

st.title("🧳 Bora Alí — Dashboard Urbano (Laranja Sunset)")
st.caption("Capitais do Brasil • Tarifas Aéreas • Temperatura • Previsões 2026")

# ---------------------------------------------
# FUNÇÕES AUXILIARES
# ---------------------------------------------
def normalize_city(city):
    if pd.isna(city): return city
    c = "".join(ch for ch in unicodedata.normalize("NFKD", str(city)) if not unicodedata.combining(ch))
    return " ".join(c.replace("_"," ").replace("-"," ").split()).title()

def parse_route(r):
    if pd.isna(r): return (None,None)
    for sep in ["→","-","/"]:
        if sep in str(r):
            pts=[x.strip() for x in str(r).split(sep)]
            if len(pts)>=2: return pts[0],pts[-1]
    return (None,None)

# ---------------------------------------------
# LEITURA + TRATAMENTO DO CSV
# ---------------------------------------------
@st.cache_data
def load_csv():
    try:
        df = pd.read_csv(CSV_FILE, low_memory=False)
    except:
        st.error("⛔ CSV NÃO ENCONTRADO! Coloque **INMET_ANAC_ROTAS_APENAS_CAPITAIS.csv** na raiz.")
        st.stop()

    df.columns=[c.upper().strip() for c in df.columns]

    for c in ["TARIFA","TEMP_MEDIA","TEMP_MIN","TEMP_MAX"]:
        df[c]=pd.to_numeric(df.get(c),errors="coerce")
    if df["TEMP_MEDIA"].isna().all():
        df["TEMP_MEDIA"]=(df["TEMP_MIN"]+df["TEMP_MAX"])/2

    parsed=df["ROTA"].apply(lambda r: pd.Series(parse_route(r),index=["_o","_d"]))
    df["ORIG"]=df.get("ORIGEM",parsed["_o"]).fillna(parsed["_o"]).apply(normalize_city)
    df["DEST"]=df.get("DESTINO",parsed["_d"]).fillna(parsed["_d"]).apply(normalize_city)

    df["ANO"]=pd.to_numeric(df["ANO"],errors="coerce").fillna(0).astype(int)
    df["MES"]=pd.to_numeric(df["MES"],errors="coerce").fillna(0).astype(int)

    MESES={1:"Janeiro",2:"Fevereiro",3:"Março",4:"Abril",5:"Maio",6:"Junho",
           7:"Julho",8:"Agosto",9:"Setembro",10:"Outubro",11:"Novembro",12:"Dezembro"}
    df["MES_NOME"]=df["MES"].map(MESES)

    def est(m):
        return "Verão" if m in [12,1,2] else "Outono" if m in [3,4,5] else "Inverno" if m in [6,7,8] else "Primavera"
    df["ESTACAO"]=df["MES"].apply(est)

    df["DATA"]=pd.to_datetime(df["ANO"].astype(str)+"-"+df["MES"].astype(str).str.zfill(2)+"-01")
    df["ROTA"]=df["ORIG"]+" → "+df["DEST"]
    return df

df = load_csv()

# ---------------------------------------------
# COORDENADAS DAS CAPITAIS
# ---------------------------------------------
COORDS={'Rio Branco':(-9.97,-67.82),'Maceió':(-9.64,-35.70),'Macapá':(0.03,-51.06),'Manaus':(-3.11,-60.02),
'Salvador':(-12.97,-38.50),'Fortaleza':(-3.71,-38.54),'Brasília':(-15.79,-47.88),'Vitória':(-20.31,-40.31),
'Goiânia':(-16.68,-49.26),'São Luís':(-2.52,-44.30),'Cuiabá':(-15.60,-56.09),'Campo Grande':(-20.44,-54.64),
'Belo Horizonte':(-19.91,-43.93),'Belém':(-1.45,-48.50),'João Pessoa':(-7.11,-34.84),'Curitiba':(-25.42,-49.27),
'Recife':(-8.04,-34.87),'Teresina':(-5.08,-42.80),'Rio de Janeiro':(-22.90,-43.17),'Natal':(-5.79,-35.20),
'Porto Alegre':(-30.03,-51.21),'Porto Velho':(-8.76,-63.90),'Boa Vista':(2.81,-60.67),'Florianópolis':(-27.59,-48.54),
'Aracaju':(-10.94,-37.07),'São Paulo':(-23.55,-46.63),'Palmas':(-10.18,-48.33)}

# ---------------------------------------------
# SIDEBAR (FILTROS)
# ---------------------------------------------
st.sidebar.header("🎯 Filtros Inteligentes")

anos=sorted(df["ANO"].unique())
meses=list(df["MES_NOME"].dropna().unique())  # mantém ordem original
companias=sorted(df["COMPANHIA"].dropna().unique())
estacoes=["Verão","Outono","Inverno","Primavera"]
caps=list(COORDS.keys())

sel_ano=st.sidebar.multiselect("Ano",anos,default=anos)
sel_mes=st.sidebar.multiselect("Mês",meses,default=meses)
sel_comp=st.sidebar.multiselect("Companhia",companias,default=companias)
sel_est=st.sidebar.multiselect("Estação",estacoes,default=estacoes)
sel_cap=st.sidebar.multiselect("Capitais",caps,default=caps)

dff=df[(df["ANO"].isin(sel_ano))&(df["MES_NOME"].isin(sel_mes))&(df["COMPANHIA"].isin(sel_comp))&
       (df["ESTACAO"].isin(sel_est))&(df["ORIG"].isin(sel_cap))&(df["DEST"].isin(sel_cap))]

if dff.empty:
    st.error("⛔ Nenhum registro com esses filtros!")
    st.stop()

# ---------------------------------------------
# KPIs
# ---------------------------------------------
st.markdown("---")
c1,c2,c3,c4=st.columns(4)
c1.metric("📊 Registros",f"{len(dff):,}")
c2.metric("💰 Tarifa Média",f"R$ {dff['TARIFA'].mean():.0f}")
c3.metric("🌡 Temp Média",f"{dff['TEMP_MEDIA'].mean():.1f} °C")
c4.metric("✈️ Rotas Únicas",dff["ROTA"].nunique())

# ---------------------------------------------
# 1) MAPA CAPITAIS
# ---------------------------------------------
st.markdown("---")
st.subheader("🗺️ 1) Capitais — Tarifas vs Temperatura")
agg=dff.groupby("DEST").agg(tar=("TARIFA","mean"),temp=("TEMP_MEDIA","mean"),regs=("TARIFA","count")).reset_index()
agg["lat"]=agg["DEST"].map(lambda x:COORDS[x][0])
agg["lon"]=agg["DEST"].map(lambda x:COORDS[x][1])

fig1=px.scatter_mapbox(agg,lat="lat",lon="lon",size="tar",color="temp",
                       hover_name="DEST",
                       hover_data={"tar":":.0f","temp":":.1f","regs":True,"lat":False,"lon":False},
                       color_continuous_scale=[SOFT,ORANGE,PURPLE],size_max=45,zoom=3.1)
fig1.update_layout(mapbox_style="carto-positron",margin=dict(l=0,r=0,t=0,b=0))
st.plotly_chart(fig1,use_container_width=True)

# ---------------------------------------------
# 2) MAPA ROTAS (Texto para Leigos)
# ---------------------------------------------
st.markdown("---")
st.subheader("🛫 2) Rotas Mais Caras do Brasil (quanto mais grossa, maior o preço da passagem)")
st.caption("✈️ As linhas mais espessas indicam rotas com valores médios de passagens mais altos.")

routes=dff.groupby("ROTA").agg(m=("TARIFA","mean"),regs=("TARIFA","count")).reset_index()
routes[["O","D"]]=routes["ROTA"].apply(lambda r: pd.Series(parse_route(r)))
routes["olat"]=routes["O"].map(lambda x:COORDS.get(normalize_city(x),(np.nan,np.nan))[0])
routes["olon"]=routes["O"].map(lambda x:COORDS.get(normalize_city(x),(np.nan,np.nan))[1])
routes["dlat"]=routes["D"].map(lambda x:COORDS.get(normalize_city(x),(np.nan,np.nan))[0])
routes["dlon"]=routes["D"].map(lambda x:COORDS.get(normalize_city(x),(np.nan,np.nan))[1])
routes=routes.dropna()

if not routes.empty:
    q=routes["m"].quantile([.25,.50,.75])
    def width(x): return 1.5 if x<=q[.25] else 3 if x<=q[.50] else 5 if x<=q[.75] else 7
    routes["w"]=routes["m"].apply(width)

    fig2=go.Figure()
    for _,r in routes.iterrows():
        fig2.add_trace(go.Scattermapbox(lat=[r["olat"],r["dlat"]],lon=[r["olon"],r["dlon"]],
                                        mode="lines",
                                        line=dict(width=r["w"],color=ORANGE),
                                        hoverinfo="text",
                                        text=f"<b>{r['ROTA']}</b><br>💰 R$ {r['m']:.0f}<br>📌 {int(r['regs'])} registros"))
    fig2.update_layout(mapbox_style="carto-positron",
                       mapbox_center={"lat":-14.2,"lon":-51.9},
                       mapbox_zoom=3.1,
                       height=540,
                       margin=dict(l=0,r=0,t=0,b=0))
    st.plotly_chart(fig2,use_container_width=True)
else:
    st.info("⚠ Não há rotas com esses filtros.")

# ---------------------------------------------
# 3) Ranking — Destinos
# ---------------------------------------------
st.markdown("---")
st.subheader("🏆 3) Destinos Mais Caros")
rank=dff.groupby("DEST").agg(m=("TARIFA","mean")).reset_index().sort_values("m",ascending=False)
fig3=px.bar(rank,x="DEST",y="m",text=rank["m"].round(0),
            color="m",color_continuous_scale=[SOFT,ORANGE,PURPLE])
fig3.update_traces(textposition="outside")
fig3.update_layout(yaxis_title="Tarifa média (R$)")
st.plotly_chart(fig3,use_container_width=True)

# ---------------------------------------------
# 4) Temporal
# ---------------------------------------------
st.markdown("---")
st.subheader("📈 4) Variação de Tarifas ao Longo do Tempo")
ts=dff.groupby("DATA").agg(m=("TARIFA","mean")).reset_index()
st.plotly_chart(px.line(ts,x="DATA",y="m",markers=True,color_discrete_sequence=[ORANGE])
                .update_layout(yaxis_title="Tarifa média (R$)"),use_container_width=True)

# ---------------------------------------------
# 5) Estações
# ---------------------------------------------
st.markdown("---")
st.subheader("🌦 5) Tarifas por Estação do Ano")
est=dff.groupby("ESTACAO").agg(m=("TARIFA","mean")).reset_index()
st.plotly_chart(px.bar(est,x="ESTACAO",y="m",text=est["m"].round(0),
    color="ESTACAO",color_discrete_sequence=[SOFT,ORANGE,PURPLE,"#FFC872"])
    .update_traces(textposition="outside"),use_container_width=True)

# ---------------------------------------------
# 6) Regiões
# ---------------------------------------------
st.markdown("---")
st.subheader("🌎 6) Tarifas por Região do Brasil")
REG={"Norte":["Belém","Macapá","Manaus","Boa Vista","Rio Branco","Porto Velho","Palmas"],
"Nordeste":["São Luís","Teresina","Fortaleza","Natal","João Pessoa","Recife","Maceió","Aracaju","Salvador"],
"Centro-Oeste":["Brasília","Goiânia","Campo Grande","Cuiabá"],
"Sudeste":["São Paulo","Rio de Janeiro","Belo Horizonte","Vitória"],
"Sul":["Curitiba","Florianópolis","Porto Alegre"]}
def reg(x):
    for k,v in REG.items():
        if x in v:return k
    return "Outro"
dff["REGIAO"]=dff["DEST"].apply(reg)
regm=dff.groupby("REGIAO").agg(m=("TARIFA","mean")).reset_index()
st.plotly_chart(px.bar(regm,x="REGIAO",y="m",text=regm["m"].round(0),
    color="REGIAO",color_discrete_sequence=[ORANGE,PURPLE,SOFT,"#A6E3E9","#FF8C42"])
    .update_traces(textposition="outside")
    .update_layout(yaxis_title="Tarifa média (R$)"),use_container_width=True)

# ---------------------------------------------
# 7) PREVISÃO 2026 — DESTINOS
# ---------------------------------------------
st.markdown("---")
st.subheader("🔮 7) Previsão 2026 — Destinos mais caros do Brasil")

dest_pred=[]
for destino in dff["DEST"].unique():
    tmp=dff[dff["DEST"]==destino].groupby("DATA").agg(tar=("TARIFA","mean")).reset_index()
    if tmp.shape[0]<12: continue
    tmp=tmp.rename(columns={"DATA":"ds","tar":"y"})
    model=Prophet(yearly_seasonality=True)
    model.fit(tmp)
    fut=model.make_future_dataframe(periods=12,freq="MS")
    forecast=model.predict(fut)
    pred=forecast[forecast["ds"].dt.year==2026]["yhat"].mean()
    dest_pred.append((destino,pred))

df_pred=pd.DataFrame(dest_pred,columns=["Destino","Tarifa Prevista"]).dropna().sort_values("Tarifa Prevista",ascending=False)

fig7=px.bar(df_pred,x="Destino",y="Tarifa Prevista",
            text=df_pred["Tarifa Prevista"].round(0),
            color="Tarifa Prevista",color_continuous_scale=[SOFT,ORANGE,PURPLE])
fig7.update_traces(textposition="outside")
fig7.update_layout(yaxis_title="Preço previsto (R$)")
st.plotly_chart(fig7,use_container_width=True)

# Insight automático
top=df_pred.head(3)["Destino"].tolist()
st.success(f"🔎 Em 2026, os destinos mais caros para viajar devem ser: **{', '.join(top)}**.")

st.caption("🌇 Bora Alí © Laranja Sunset — SR2")

