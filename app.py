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
from sklearn.model_selection import GroupKfold, cross_val_score
import math, warnings
warnings.filterwarnings("ignore")

# -------------------- OneHot UNIVERSAL (não quebra) --------------------
def OneHotSafe():
    """Retorna um OneHotEncoder com manuseio seguro de sparse_output/sparse."""
    try:
        # Tenta usar a versão mais recente (sklearn >=1.2)
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        # Fallback para versões mais antigas (sklearn <1.2)
        return OneHotEncoder(handle_unknown="ignore", sparse=False)

# -------------------- PALETA --------------------
COLOR_ORANGE = "#FF8A33" # Laranja
COLOR_LILAC  = "#C77DFF" # Lilás
COLOR_LIME   = "#8BFF66" # Limão
COLOR_BLUE   = "#28F8FF" # Ciano

COLORS_SEASON = {"VERÃO":COLOR_ORANGE, "OUTONO":COLOR_LILAC, "INVERNO":COLOR_LIME, "PRIMAVERA":COLOR_BLUE}

# -------------------- Coordenadas memorizadas (Capitais do Brasil) --------------------
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
    """Converte o número do mês para a estação brasileira correspondente."""
    if m in (12,1,2): return "VERÃO"
    if m in (3,4,5): return "OUTONO"
    if m in (6,7,8): return "INVERNO"
    return "PRIMAVERA"

@st.cache_resource(show_spinner="Treinando modelo de previsão...")
def train_model_safe(df):
    """
    Treina o modelo de Gradient Boosting Regressor com Cross-Validation por Grupo.
    """
    FEATURES = ["ORIGEM","DESTINO","COMPANHIA","MES","TEMP_MEDIA","SIN","COS"]
    df = df.dropna(subset=FEATURES+["TARIFA"])
    
    # Pré-processamento
    cat = ["ORIGEM","DESTINO","COMPANHIA"]
    prep = ColumnTransformer([("cat", OneHotSafe(), cat)], remainder="passthrough")
    
    # Pipeline: Pré-processamento + Modelo GBR
    model = Pipeline([
        ("prep", prep),
        ("gbr", GradientBoostingRegressor(n_estimators=300, learning_rate=0.05, max_depth=4))
    ])
    
    X = df[FEATURES]
    y = df["TARIFA"].values
    groups = df["ROTA"] 

    # Avaliação do modelo usando Cross-Validation com GroupKFold
    cv = GroupKfold(n_splits=4)
    scores = -cross_val_score(
        model, 
        X, 
        y, 
        cv=cv.split(X, y, groups), 
        scoring="neg_mean_absolute_error"
    )
    
    # Treinamento final do modelo
    model.fit(X, y)
    
    return model, scores.mean()

def build_future(df, o, d, c):
    """
    Cria um DataFrame de 12 meses (2026) para previsão, 
    usando a temperatura média histórica para a rota.
    """
    months=range(1,13)
    base = df[(df["ORIGEM"]==o)&(df["DESTINO"]==d)&(df["COMPANHIA"]==c)]
    if base.empty: 
        base = df[(df["ORIGEM"]==o)&(df["DESTINO"]==d)]
    
    temp = base.groupby("MES")["TEMP_MEDIA"].mean().reindex(months)
    temp.fillna(df["TEMP_MEDIA"].mean(), inplace=True)
    
    rows=[]
    for m in months:
        rows.append({
            "ANO":2026, 
            "MES":m, 
            "ORIGEM":o, 
            "DESTINO":d, 
            "COMPANHIA":c,
            "TEMP_MEDIA":temp.loc[m],
            "SIN":math.sin(2*math.pi*m/12),
            "COS":math.cos(2*math.pi*m/12)
        })
    
    f = pd.DataFrame(rows)
    f["SEASON"] = f["MES"].apply(month_to_season)
    return f

# -------------------- LOAD CSV & PRÉ-PROCESSAMENTO --------------------
st.set_page_config(page_title="Bora Alí — SR2", layout="wide")

# ATENÇÃO: Verifique se este nome de arquivo está correto
CSV = "INMET_ANAC_EXTREMAMENTE_REDUZIDO.csv" 

@st.cache_data
def load_data(file_path):
    try:
        df = pd.read_csv(file_path)
    except FileNotFoundError:
        st.error(f"Arquivo CSV '{file_path}' não encontrado. Por favor, carregue o arquivo para rodar o dashboard.")
        st.stop()
        
    df["MES"] = df["MES"].astype(int)
    df["ANO"] = df["ANO"].astype(int)
    df["TEMP_MEDIA"] = pd.to_numeric(df["TEMP_MEDIA"], errors="coerce")
    df["TARIFA"] = pd.to_numeric(df["TARIFA"], errors="coerce")
    
    df["SEASON"] = df["MES"].apply(month_to_season)
    df["ROTA"] = df["ORIGEM"] + " → " + df["DESTINO"]
    df["SIN"] = np.sin(2 * np.pi * df["MES"] / 12)
    df["COS"] = np.cos(2 * np.pi * df["MES"] / 12)
    
    df["DEST_LAT"] = df["DESTINO"].apply(lambda x: CAPITAL_COORDS.get(x, (np.nan, np.nan))[0])
    df["DEST_LON"] = df["DESTINO"].apply(lambda x: CAPITAL_COORDS.get(x, (np.nan, np.nan))[1])
    
    return df

df = load_data(CSV)
df_map = df.dropna(subset=["DEST_LAT", "DEST_LON"])


# -------------------- LAYOUT E COMPONENTES STREAMLIT --------------------

st.title("✈️ Bora Alí — SR2 Dashboard")

# -------------------- KPIs --------------------
st.header("Metadados e Visão Geral")
col1, col2, col3 = st.columns(3)
col1.metric("Registros Históricos", f"{df.shape[0]:,}")
col2.metric("Tarifa Média Histórica", f"R$ {df['TARIFA'].mean():.2f}")
col3.metric("Rotas Únicas Monitoradas", df["ROTA"].nunique())

st.divider()

# -------------------- Gráfico Estação --------------------
st.subheader("🌦️ Média Tarifária por Estação")
s = df.groupby("SEASON")["TARIFA"].mean().reset_index()
season_order = ["VERÃO", "OUTONO", "INVERNO", "PRIMAVERA"]
s["SEASON"] = pd.Categorical(s["SEASON"], categories=season_order, ordered=True)
s = s.sort_values("SEASON")

fig = px.bar(
    s, 
    x="SEASON", 
    y="TARIFA", 
    color="SEASON", 
    color_discrete_map=COLORS_SEASON,
    title="Preço Médio Histórico de Passagens por Estação"
)
fig.update_layout(xaxis_title="Estação", yaxis_title="Tarifa Média (R$)", showlegend=False)
st.plotly_chart(fig, use_container_width=True)

st.divider()

# -------------------- Previsão (Machine Learning) --------------------
st.subheader("🔮 Previsão da Tarifa Aérea para 2026")

# Filtros para seleção da rota
col_o, col_d, col_c = st.columns(3)

with col_o:
    o = st.selectbox("Origem", sorted(df["ORIGEM"].unique()))

destinos_validos = sorted(df[df["ORIGEM"] == o]["DESTINO"].unique())
with col_d:
    d = st.selectbox("Destino", destinos_validos)

companhias_validas = sorted(df[(df["ORIGEM"] == o) & (df["DESTINO"] == d)]["COMPANHIA"].unique())
with col_c:
    c = st.selectbox("Companhia", companhias_validas)


# Treinamento do modelo
df_model = df.dropna(subset=["TARIFA", "TEMP_MEDIA"])

# BLCO DE SEGURANÇA: Só executa previsão e mapa se houver dados suficientes
if not df_model.empty and df_model.shape[0] > 10: 
    model, mae = train_model_safe(df_model)

    # -------------------- PLOT DA PREVISÃO --------------------
    fut = build_future(df, o, d, c)
    fut["PRED"] = model.predict(fut[["ORIGEM", "DESTINO", "COMPANHIA", "MES", "TEMP_MEDIA", "SIN", "COS"]])

    estacao_mais_barata = fut.groupby('SEASON')['PRED'].mean().idxmin()
    mae_formatado = f"R$ {mae:.2f}"
    
    st.success(f"**Resultado da Previsão:** A rota mais barata (em média) para 2026 é na estação **{estacao_mais_barata}**. (Erro Absoluto Médio do Modelo: {mae_formatado})")
    
    meses_nomes = ["Jan", "Fev", "Mar", "Abr", "Mai", "Jun", "Jul", "Ago", "Set", "Out", "Nov", "Dez"]
    fut["MÊS_NOME"] = fut["MES"].apply(lambda x: meses_nomes[x-1])
    
    fig2 = px.line(
        fut, 
        x="MÊS_NOME", 
        y="PRED", 
        markers=True, 
        color_discrete_sequence=[COLOR_ORANGE],
        title=f"Previsão de Tarifa Média (2026): {o} → {d} ({c})"
    )
    fig2.update_layout(xaxis_title="Mês", yaxis_title="Tarifa Prevista (R$)", hovermode="x unified")
    fig2.update_xaxes(tickangle=45)
    
    st.plotly_chart(fig2, use_container_width=True)

    st.divider() 

    # -------------------- MAPA (Pydeck) - SEÇÃO SEGURA --------------------
    st.subheader("🗺️ Mapa de Tendência — Capitais (2026 vs Histórico)")

    hist = df.groupby("DESTINO")["TARIFA"].mean().reset_index().rename(columns={"TARIFA":"HIST"})
    pred_list = []

    with st.spinner("Calculando variação de tendência para o mapa..."):
        for dest in df["DESTINO"].unique():
            route_data = df[df["DESTINO"] == dest]
            if route_data.empty: continue

            o_tmp = route_data["ORIGEM"].mode().iloc[0] if not route_data["ORIGEM"].mode().empty else 'Origem Padrão'
            c_tmp = route_data["COMPANHIA"].mode().iloc[0] if not route_data["COMPANHIA"].mode().empty else 'Companhia Padrão'
            
            f = build_future(df, o_tmp, dest, c_tmp)
            f["P"] = model.predict(f[["ORIGEM","DESTINO","COMPANHIA","MES","TEMP_MEDIA","SIN","COS"]])
            pred_list.append({"DESTINO": dest, "PRED": f["P"].mean()})

    pred = pd.DataFrame(pred_list)
    m = hist.merge(pred, on="DESTINO")
    m["VAR"] = (m["PRED"] - m["HIST"]) / m["HIST"] 

    def status(v):
        """Classifica a variação em Queda, Alta ou Estável."""
        if v <= -0.05: return "QUEDA" 
        if v >= 0.05: return "ALTA"  
        return "ESTÁVEL"

    m["STATUS"] = m["VAR"].apply(status)
    m = m.merge(df_map[["DESTINO", "DEST_LAT", "DEST_LON"]].drop_duplicates(), on="DESTINO")

    # -------------------- Configuração do Pydeck --------------------
    layer = pdk.Layer(
        "ScatterplotLayer",
        data=m.dropna(subset=["DEST_LAT", "DEST_LON"]),
        get_position='[DEST_LON, DEST_LAT]',
        # Cores: PRED > HIST (Vermelho/Laranja), PRED < HIST (Verde/Limão)
        get_fill_color=f"""[
            (PRED > HIST ? 255 : 0),
            (PRED < HIST ? 255 : 0),
            100, 200 ]""", 
        get_radius=60000, 
        pickable=True,
        tooltip={
            "text": "{DESTINO}\nStatus: {STATUS}\nHistórico: R$ {HIST:.2f}\nPrevisto: R$ {PRED:.2f}\nVariação: {VAR:.2%}"
        }
    )

    view = pdk.ViewState(latitude=-15, longitude=-55, zoom=3.5, min_zoom=3, max_zoom=7)

    st.pydeck_chart(pdk.Deck(
        layers=[layer], 
        initial_view_state=view, 
        map_style="mapbox://styles/mapbox/light-v11" 
    ))

    st.markdown("### Tabela de Tendências de Capitais")
    st.dataframe(m[["DESTINO", "STATUS", "HIST", "PRED", "VAR"]].style.format({
        "HIST": "R$ {:.2f}", 
        "PRED": "R$ {:.2f}", 
        "VAR": "{:.2%}"
    }))

else:
    st.warning("Dados insuficientes para treinar o modelo de previsão e gerar o mapa de tendências. Verifique se o arquivo CSV contém dados válidos para 'TARIFA' e 'TEMP_MEDIA'.")
