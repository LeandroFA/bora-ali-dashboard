# app.py — Bora Alí (versão com pydeck + Prophet)
# Rode: streamlit run app.py

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import pydeck as pdk
from prophet import Prophet
from prophet.plot import plot_plotly
from datetime import datetime

st.set_page_config(layout="wide", page_title="Bora Alí — Capitais", page_icon="✈️")

# -------------------------
# CONFIG: default CSV (raw GitHub URL do seu repo)
DEFAULT_CSV = "https://raw.githubusercontent.com/LeandroFA/bora-ali-dashboard/main/INMET_ANAC_ROTAS_APENAS_CAPITAIS.csv"

# -------------------------
@st.cache_data
def read_csv(path_or_buffer):
    try:
        if hasattr(path_or_buffer, "read"):
            df = pd.read_csv(path_or_buffer, low_memory=False)
        else:
            df = pd.read_csv(path_or_buffer, low_memory=False)
    except Exception as e:
        st.error(f"Erro lendo CSV: {e}")
        return pd.DataFrame()
    # normalize column names
    df.columns = [c.strip().lower() for c in df.columns]
    return df

def normalize(df):
    rename_map = {
        "tarifa_":"tarifa","price":"tarifa","valor":"tarifa",
        "temp_media":"temp_media","temp":"temp_media",
        "orig":"origem","dst":"destino",
        "longitude":"lon","latitude":"lat"
    }
    df = df.rename(columns={k:v for k,v in rename_map.items() if k in df.columns})
    if 'tarifa' in df.columns:
        df['tarifa'] = pd.to_numeric(df['tarifa'], errors='coerce')
    if 'temp_media' in df.columns:
        df['temp_media'] = pd.to_numeric(df['temp_media'], errors='coerce')
    if 'data' in df.columns:
        df['data'] = pd.to_datetime(df['data'], dayfirst=True, errors='coerce')
        df['ano'] = df['data'].dt.year
        df['mes'] = df['data'].dt.month
    # build rota if missing
    if 'rota' not in df.columns and 'origem' in df.columns and 'destino' in df.columns:
        df['rota'] = df['origem'].astype(str) + " - " + df['destino'].astype(str)
    # drop rows without tarifa
    if 'tarifa' in df.columns:
        df = df.dropna(subset=['tarifa'])
    return df

def hover_template_destino():
    return "<b>%{properties.name}</b><br>Tarifa média: R$ %{properties.tarifa:.2f}<br>Temp média: %{properties.temp_media:.1f}°C"

# -------------------------
# Sidebar: upload / source / filters
st.sidebar.title("Bora Alí — Controles")
st.sidebar.markdown("**Fonte padrão:** GitHub (raw CSV) — você pode carregar outro CSV abaixo.")
uploaded = st.sidebar.file_uploader("Carregue um CSV (opcional)", type=["csv"])
use_url = st.sidebar.text_input("Ou cole URL raw CSV (opcional)", value=DEFAULT_CSV)

# choose source
source = uploaded if uploaded is not None else use_url
df_raw = read_csv(source)
if df_raw.empty:
    st.warning("CSV vazio ou inválido. Verifique o arquivo/URL e recarregue.")
    st.stop()

df = normalize(df_raw)

# quick col detection
cols = df.columns.tolist()
has_geo = ('lat' in cols and 'lon' in cols)
has_rota = 'rota' in cols or ('origem' in cols and 'destino' in cols)

# Filters
st.sidebar.markdown("---")
st.sidebar.header("Filtros rápidos")
anos = sorted(df['ano'].dropna().unique().tolist()) if 'ano' in df.columns else []
selected_anos = st.sidebar.multiselect("Ano", options=anos, default=anos if anos else None)
selected_comp = st.sidebar.multiselect("Companhia", options=sorted(df['companhia'].dropna().unique().tolist()) if 'companhia' in df.columns else [], default=None)
selected_mes = st.sidebar.multiselect("Mês", options=sorted(df['mes'].dropna().unique().tolist()) if 'mes' in df.columns else list(range(1,13)), default=None)

# apply filters
df_f = df.copy()
if selected_anos:
    if 'ano' in df_f.columns:
        df_f = df_f[df_f['ano'].isin(selected_anos)]
    elif 'data' in df_f.columns:
        df_f = df_f[df_f['data'].dt.year.isin(selected_anos)]
if selected_comp:
    df_f = df_f[df_f['companhia'].isin(selected_comp)]
if selected_mes:
    if 'mes' in df_f.columns:
        df_f = df_f[df_f['mes'].isin(selected_mes)]
    elif 'data' in df_f.columns:
        df_f = df_f[df_f['data'].dt.month.isin(selected_mes)]

# -------------------------
# HEADER & KPIs
st.markdown("<h1 style='margin:0'>✈️ Bora Alí — Capitais</h1>", unsafe_allow_html=True)
st.markdown("Dashboard interativo • filtros à esquerda • mapas e insights abaixo")

k1, k2, k3, k4 = st.columns(4)
k1.metric("Registros (filtrados)", f"{len(df_f):,}")
k2.metric("Tarifa média (R$)", f"{df_f['tarifa'].mean():.2f}" if 'tarifa' in df_f.columns and not df_f['tarifa'].isnull().all() else "—")
k3.metric("Temp média (°C)", f"{df_f['temp_media'].mean():.1f}" if 'temp_media' in df_f.columns and not df_f['temp_media'].isnull().all() else "—")
k4.metric("Rotas únicas", df_f['rota'].nunique() if 'rota' in df_f.columns else (df_f.groupby(['origem','destino']).ngroups if ('origem' in df_f.columns and 'destino' in df_f.columns) else "—"))

# -------------------------
# INSIGHTS (cards)
st.markdown("### Insights automáticos")
col_a, col_b, col_c = st.columns(3)

# top rota
if 'rota' in df_f.columns:
    top_rota = df_f['rota'].value_counts().idxmax()
    top_rota_count = df_f['rota'].value_counts().max()
    col_a.info(f"**Rota mais frequente:** {top_rota} ({top_rota_count} registros)")
else:
    col_a.info("Rota mais frequente: —")

# rota mais cara (média)
if 'rota' in df_f.columns and 'tarifa' in df_f.columns:
    rota_mais_cara = df_f.groupby('rota')['tarifa'].mean().idxmax()
    col_b.info(f"**Rota mais cara (média):** {rota_mais_cara}")
else:
    col_b.info("Rota mais cara: —")

# origem mais movimentada
if 'origem' in df_f.columns:
    origem_top = df_f['origem'].value_counts().idxmax()
    col_c.info(f"**Origem mais movimentada:** {origem_top}")
else:
    col_c.info("Origem mais movimentada: —")

col_d, col_e, col_f = st.columns(3)
# destino mais barato
if 'destino' in df_f.columns and 'tarifa' in df_f.columns:
    destino_barato = df_f.groupby('destino')['tarifa'].mean().idxmin()
    col_d.info(f"**Destino mais barato (média):** {destino_barato}")
else:
    col_d.info("Destino mais barato: —")

# mês com pico
if 'mes' in df_f.columns and 'tarifa' in df_f.columns:
    mes_pico = int(df_f.groupby('mes')['tarifa'].mean().idxmax())
    col_e.info(f"**Mês com pico de tarifa (média):** {mes_pico}")
else:
    col_e.info("Mês com pico: —")

# variação ano anterior (simples)
def pct_vs_prev_year(df_all, df_current):
    if 'ano' in df_all.columns and 'ano' in df_current.columns:
        years = sorted(df_current['ano'].dropna().unique().tolist())
        prev_years = [y-1 for y in years]
        prev = df_all[df_all['ano'].isin(prev_years)]
        if prev.empty or df_current.empty:
            return None
        return (df_current['tarifa'].mean() - prev['tarifa'].mean()) / prev['tarifa'].mean()
    return None

change = pct_vs_prev_year(df, df_f)
col_f.metric("Variação vs ano anterior", f"{change*100:+.1f}%" if change is not None else "—")

# -------------------------
# MAP: pydeck interactive (left) + rota selector (right)
left, right = st.columns([2,1])

with left:
    st.markdown("## Mapa interativo — capitais")
    if has_geo:
        # build GeoJSON-like features for pydeck; hide lat/lon in tooltip by using properties
        points = df_f.dropna(subset=['lat','lon']).groupby('destino').agg(
            tarifa=('tarifa','mean'), temp_media=('temp_media','mean'),
            lat=('lat','first'), lon=('lon','first'), registros=('tarifa','count')
        ).reset_index()
        if points.empty:
            st.info("Sem pontos com lat/lon no dataset filtrado.")
        else:
            # create pydeck datasource
            features = []
            for _, r in points.iterrows():
                features.append({
                    "type":"Feature",
                    "geometry":{"type":"Point","coordinates":[r['lon'], r['lat']]},
                    "properties":{"name": r['destino'], "tarifa": float(r['tarifa']) if not pd.isna(r['tarifa']) else None, "temp_media": float(r['temp_media']) if not pd.isna(r['temp_media']) else None, "registros": int(r['registros'])}
                })
            geojson = {"type":"FeatureCollection", "features": features}
            # deck layer
            layer = pdk.Layer(
                "GeoJsonLayer",
                geojson,
                pickable=True,
                stroked=False,
                filled=True,
                point_radius_min_pixels=5,
                get_fill_color="[255 - properties.registros*2, 100, properties.registros*2, 180]",
                get_radius="properties.registros * 300",
            )
            # tooltip: only show tarifa/temp_media and name
            tooltip = {"html": "<b>{name}</b><br>Tarifa média: R$ {tarifa:.2f}<br>Temp média: {temp_media:.1f}°C", "style": {"color":"#000"}}
            view_state = pdk.ViewState(latitude=points['lat'].mean(), longitude=points['lon'].mean(), zoom=4)
            r = pdk.Deck(layers=[layer], initial_view_state=view_state, tooltip=tooltip)
            st.pydeck_chart(r)
    else:
        st.info("Sem lat/lon — carregue dados com colunas 'lat' e 'lon' para usar o mapa.")

with right:
    st.markdown("### Rotas — destaque")
    if has_rota and has_geo:
        top_n = st.slider("Top N rotas", 5, 40, 12)
        route_counts = df_f['rota'].value_counts().reset_index()
        route_counts.columns = ['rota','freq']
        top_routes = route_counts.head(top_n)['rota'].tolist()
        sel = st.selectbox("Selecione rota", options=["Nenhuma"] + top_routes)
        if sel != "Nenhuma":
            points_map = df_f.dropna(subset=['lat','lon']).groupby('destino').agg(lat=('lat','first'), lon=('lon','first')).reset_index()
            try:
                o,d = [p.strip() for p in sel.split('-')]
                orow = points_map[points_map['destino'].str.lower()==o.lower()]
                drow = points_map[points_map['destino'].str.lower()==d.lower()]
                if not orow.empty and not drow.empty:
                    olat, olon = float(orow['lat'].iloc[0]), float(orow['lon'].iloc[0])
                    dlat, dlon = float(drow['lat'].iloc[0]), float(drow['lon'].iloc[0])
                    # show small map with line (plotly)
                    lats = np.linspace(olat, dlat, 20)
                    lons = np.linspace(olon, dlon, 20)
                    fig = go.Figure()
                    fig.add_trace(go.Scattermapbox(lat=lats, lon=lons, mode='lines+markers', line=dict(width=3), marker=dict(size=6), hoverinfo='none'))
                    fig.update_layout(mapbox_style='open-street-map', mapbox=dict(center=dict(lat=(olat+dlat)/2, lon=(olon+dlon)/2), zoom=4), margin=dict(l=0,r=0,t=0,b=0), height=360)
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.info("Coordenadas não encontradas para origem/destino.")
            except Exception:
                st.info("Formato de rota inesperado.")
    else:
        st.info("Para destacar rotas, é preciso coluna 'rota' e colunas 'lat'/'lon'.")

# -------------------------
# GRÁFICOS: top rotas, série temporal, boxplot, heatmap, histograma, small multiples
st.markdown("## Análises detalhadas")

# top rotas bar
st.markdown("### Top rotas")
if 'rota' in df_f.columns:
    top_table = df_f.groupby('rota').agg(freq=('rota','count'), tarifa_media=('tarifa','mean')).reset_index().sort_values('freq', ascending=False).head(20)
    fig = px.bar(top_table, x='freq', y='rota', orientation='h', text='tarifa_media', height=520)
    fig.update_traces(hovertemplate='Rota: %{y}<br>Freq: %{x}<br>Tarifa média: R$ %{customdata[0]:.2f}', customdata=np.stack([top_table['tarifa_media']], axis=-1))
    st.plotly_chart(fig, use_container_width=True)
else:
    st.info("Sem coluna 'rota'")

# time series tarifa
st.markdown("### Tarifa média — Série temporal (mensal)")
if 'data' in df_f.columns:
    ts = df_f.set_index('data').resample('M').agg(tarifa_media=('tarifa','mean')).reset_index()
    fig = px.line(ts, x='data', y='tarifa_media', markers=True)
    st.plotly_chart(fig, use_container_width=True)
else:
    st.info("Sem coluna 'data' para série temporal.")

# boxplot companhia
st.markdown("### Boxplot: tarifa por companhia")
if 'companhia' in df_f.columns:
    top_comp = df_f['companhia'].value_counts().head(8).index.tolist()
    df_comp = df_f[df_f['companhia'].isin(top_comp)]
    fig = px.box(df_comp, x='companhia', y='tarifa', points='all', height=420)
    st.plotly_chart(fig, use_container_width=True)
else:
    st.info("Sem coluna 'companhia'")

# heatmap
st.markdown("### Heatmap: tarifa média por mês x origem")
if 'mes' in df_f.columns and 'origem' in df_f.columns:
    pivot = df_f.groupby(['mes','origem']).agg(tarifa_media=('tarifa','mean')).reset_index()
    heat = pivot.pivot(index='origem', columns='mes', values='tarifa_media').fillna(0)
    fig = px.imshow(heat, labels=dict(x='Mês', y='Origem', color='Tarifa média (R$)'), aspect='auto', height=520)
    st.plotly_chart(fig, use_container_width=True)
else:
    st.info("Dados insuficientes (mes/origem)")

# histograma
st.markdown("### Distribuição de tarifas")
if 'tarifa' in df_f.columns:
    fig = px.histogram(df_f, x='tarifa', nbins=40, marginal='box', height=360)
    st.plotly_chart(fig, use_container_width=True)
else:
    st.info("Sem tarifa")

# small multiples
st.markdown("### Small multiples — tarifa média por mês por companhia")
if 'companhia' in df_f.columns and 'mes' in df_f.columns:
    comps = df_f['companhia'].value_counts().head(6).index.tolist()
    df_small = df_f[df_f['companhia'].isin(comps)].groupby(['companhia','mes']).agg(tarifa_media=('tarifa','mean')).reset_index()
    fig = px.line(df_small, x='mes', y='tarifa_media', color='companhia', facet_col='companhia', facet_col_wrap=3, markers=True, height=520)
    st.plotly_chart(fig, use_container_width=True)
else:
    st.info("Sem companhia/mes para small multiples")

# -------------------------
# FORECAST (Prophet) — previsão simples de tarifa média mensal
st.markdown("## Previsão rápida: tarifa média (Prophet)")
if 'data' in df_f.columns and 'tarifa' in df_f.columns and len(df_f)>=24:
    # prepare monthly series
    monthly = df_f.set_index('data').resample('M').agg(tarifa_media=('tarifa','mean')).reset_index().dropna()
    monthly = monthly.rename(columns={'data':'ds','tarifa_media':'y'})
    try:
        m = Prophet(yearly_seasonality=True, daily_seasonality=False, weekly_seasonality=False)
        m.fit(monthly)
        future = m.make_future_dataframe(periods=6, freq='M')
        forecast = m.predict(future)
        fig = plot_plotly(m, forecast)
        st.plotly_chart(fig, use_container_width=True)
        last = forecast[['ds','yhat']].tail(6)
        st.table(last.assign(ds=lambda df: df['ds'].dt.strftime('%Y-%m')))
    except Exception as e:
        st.info(f"Erro no Prophet: {e}")
else:
    st.info("Previsão precisa de coluna 'data' e 'tarifa' e pelo menos ~24 registros.")

# -------------------------
# EXPORT
st.sidebar.markdown("---")
st.sidebar.header("Exportar")
if st.sidebar.button("Baixar CSV filtrado"):
    csv = df_f.to_csv(index=False).encode('utf-8')
    st.sidebar.download_button("Download CSV", data=csv, file_name="capitais_filtrado.csv", mime="text/csv")

st.sidebar.markdown("Deploy: push no GitHub e conecte em share.streamlit.io")

st.caption("Versão com pydeck e Prophet — ajuste DEFAULT_CSV para outro raw URL se necessário.")


