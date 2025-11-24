# BoraAli_Streamlit_Perfect.py
# Streamlit app improved: mapas de rotas, 7+ insights interativos, hover limpo (somente tarifa/temp_media)
# Copie este arquivo para o seu projeto e rode: streamlit run BoraAli_Streamlit_Perfect.py

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime

st.set_page_config(layout="wide", page_title="Bora Alí — Capitais (Perfeito)")

# -------------------------
# CONFIG: caminho para arquivos enviados (o sistema transformará esses caminhos em URLs quando necessário)
PDF_SOURCE_1 = "/mnt/data/Bora Alí — Capitais · Streamlit.pdf"
PDF_SOURCE_2 = "/mnt/data/Bora Alí — Dashboard (Capitais) · Streamlit.pdf"
# Caminho default para CSV (se tiver). Substitua pelo seu dataset real ou use o upload abaixo.
DEFAULT_CSV = "data/capitais_clean.csv"
# -------------------------

# UTIL ------------------------------------------------
@st.cache_data
def safe_read_csv(path):
    try:
        df = pd.read_csv(path, parse_dates=True, dayfirst=True, low_memory=False)
    except Exception as e:
        st.error(f"Não foi possível ler {path}: {e}")
        return pd.DataFrame()
    # normalize cols
    df.columns = [c.strip().lower() for c in df.columns]
    return df


def normalize_and_prepare(df):
    # rename typical columns
    rename_map = {
        "tarifa_":"tarifa","price":"tarifa",
        "temp_media":"temp_media","temp":"temp_media",
        "orig":"origem","dst":"destino",
        "longitude":"lon","latitude":"lat"
    }
    df = df.rename(columns={k:v for k,v in rename_map.items() if k in df.columns})

    # guarantee numeric
    if 'tarifa' in df.columns:
        df['tarifa'] = pd.to_numeric(df['tarifa'], errors='coerce')
    if 'temp_media' in df.columns:
        df['temp_media'] = pd.to_numeric(df['temp_media'], errors='coerce')

    # try to ensure a datetime column
    if 'data' in df.columns:
        try:
            df['data'] = pd.to_datetime(df['data'], dayfirst=True, errors='coerce')
            df['ano'] = df['data'].dt.year
            df['mes'] = df['data'].dt.month
        except Exception:
            pass

    # if rota not exists, create from origem/destino
    if 'rota' not in df.columns and 'origem' in df.columns and 'destino' in df.columns:
        df['rota'] = df['origem'].astype(str) + ' - ' + df['destino'].astype(str)

    # drop rows without tarifa
    if 'tarifa' in df.columns:
        df = df.dropna(subset=['tarifa'])

    return df

# Hover template helper: hides lat/lon
def hover_template_destino():
    return '<b>%{customdata[0]}</b><br>Tarifa média: R$ %{customdata[1]:.2f}<br>Temp média: %{customdata[2]:.1f}°C<extra></extra>'

# Smooth line interpolation (linear interpolation used for simplicity)
def smooth_line(lat1, lon1, lat2, lon2, points=10):
    lats = np.linspace(lat1, lat2, points)
    lons = np.linspace(lon1, lon2, points)
    return lats, lons

# -------------------------
# DATA LOAD UI
st.sidebar.header("Carregar dados")
upload = st.sidebar.file_uploader("Carregue seu CSV (ou deixe em branco para usar DEFAULT_CSV)", type=['csv'])
use_pdf_refs = st.sidebar.checkbox("Mostrar referências de PDFs (enviados)", value=True)

if use_pdf_refs:
    st.sidebar.markdown(f"Fonte (PDF): `{PDF_SOURCE_1}`")
    st.sidebar.markdown(f"Fonte (PDF): `{PDF_SOURCE_2}`")

if upload is not None:
    df = safe_read_csv(upload)
else:
    df = safe_read_csv(DEFAULT_CSV)

if df.empty:
    st.warning("Dataset vazio — faça o upload de um CSV válido ou coloque o CSV em data/capitais_clean.csv")

# PREPARE
df = normalize_and_prepare(df)

# Basic columns detection
cols = df.columns.tolist()
has_geo = ('lat' in cols and 'lon' in cols)
has_rota = 'rota' in cols or ('origem' in cols and 'destino' in cols)

# -------------------------
# FILTERS
st.sidebar.header("Filtros rápidos")
if 'ano' in df.columns:
    anos = sorted(df['ano'].dropna().unique().tolist())
else:
    anos = sorted(pd.DatetimeIndex(df['data']).year.unique().tolist()) if 'data' in df.columns else []

selected_anos = st.sidebar.multiselect('Ano', options=anos, default=anos if anos else None)
selected_comp = st.sidebar.multiselect('Companhia', options=sorted(df['companhia'].dropna().unique().tolist()) if 'companhia' in df.columns else [], default=None)
selected_mes = st.sidebar.multiselect('Mês', options=sorted(df['mes'].dropna().unique().tolist()) if 'mes' in df.columns else list(range(1,13)), default=None)

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

st.title('Bora Alí — Capitais (Perfeito)')

# -------------------------
# INSIGHTS: KPI top row
st.subheader('KPIs')
k1, k2, k3, k4 = st.columns(4)
k1.metric('Registros (filtrados)', f"{len(df_f):,}")
if 'tarifa' in df_f.columns and not df_f['tarifa'].isnull().all():
    k2.metric('Tarifa média (R$)', f"{df_f['tarifa'].mean():.2f}")
else:
    k2.metric('Tarifa média (R$)', '—')
if 'temp_media' in df_f.columns and not df_f['temp_media'].isnull().all():
    k3.metric('Temp média (°C)', f"{df_f['temp_media'].mean():.1f}")
else:
    k3.metric('Temp média (°C)', '—')
k4.metric('Rotas únicas', df_f['rota'].nunique() if 'rota' in df_f.columns else df_f.groupby(['origem','destino']).ngroups if ('origem' in df_f.columns and 'destino' in df_f.columns) else '—')

# We'll create >=7 interactive visuals below and label cada um

# 1) MAP: pontos das capitais (hover somente tarifa/temp_media)
st.markdown('## 1 — Mapa de capitais (hover limpo: tarifa / temp_media)')
if has_geo:
    points = df_f.dropna(subset=['lat','lon']).groupby('destino').agg(
        tarifa=('tarifa','mean'), temp_media=('temp_media','mean'), lat=('lat','first'), lon=('lon','first'), registros=('tarifa','count')
    ).reset_index()

    fig_map = px.scatter_mapbox(points, lat='lat', lon='lon', size='registros', size_max=18, zoom=4, height=520, mapbox_style='open-street-map')
    fig_map.update_traces(customdata=np.stack([points['destino'], points['tarifa'], points['temp_media']], axis=-1))
    fig_map.update_traces(hovertemplate=hover_template_destino())
    fig_map.update_layout(margin=dict(l=0,r=0,t=0,b=0))
    st.plotly_chart(fig_map, use_container_width=True)
else:
    st.info('Sem coordenadas (lat/lon). Faça o upload com colunas lat e lon para visualizar o mapa.')

# 2) MAP + Rotas: top N rotas com linhas suaves (hover mostra tarifa média + freq) 
st.markdown('## 2 — Mapa de rotas (top rotas)')
if has_geo and has_rota:
    top_n = st.slider('Top N rotas', 5, 50, 12, key='topn')
    route_counts = df_f['rota'].value_counts().reset_index()
    route_counts.columns = ['rota','freq']
    top_routes = route_counts.head(top_n)['rota'].tolist()
    df_routes = df_f[df_f['rota'].isin(top_routes)]

    # ensure points mapping exists
    points_map = None
    if 'destino' in df_f.columns and 'lat' in df_f.columns:
        points_map = df_f.dropna(subset=['lat','lon']).groupby('destino').agg(lat=('lat','first'), lon=('lon','first')).reset_index()

    fig_routes = px.scatter_mapbox(points_map, lat='lat', lon='lon', zoom=4, height=520, mapbox_style='open-street-map') if points_map is not None else go.Figure()
    # add points as invisible (only to set map center if no points)
    if points_map is not None and not points_map.empty:
        fig_routes.update_traces(marker=dict(size=8), selector=dict(mode='markers'))

    for rota in top_routes:
        try:
            orig, dest = [p.strip() for p in rota.split('-')]
        except Exception:
            continue
        o = points_map[points_map['destino'].str.lower()==orig.lower()] if points_map is not None else pd.DataFrame()
        d = points_map[points_map['destino'].str.lower()==dest.lower()] if points_map is not None else pd.DataFrame()
        if o.empty or d.empty:
            continue
        olat, olon = float(o['lat'].iloc[0]), float(o['lon'].iloc[0])
        dlat, dlon = float(d['lat'].iloc[0]), float(d['lon'].iloc[0])
        lats, lons = smooth_line(olat, olon, dlat, dlon, points=12)
        freq = int(df_routes[df_routes['rota']==rota].shape[0])
        tarifa_media = df_f[df_f['rota']==rota]['tarifa'].mean()
        fig_routes.add_trace(go.Scattermapbox(lat=lats, lon=lons, mode='lines', line=dict(width=2+np.log1p(freq)*2), hoverinfo='text', text=f"{rota} — Tarifa média: R$ {tarifa_media:.2f} — Freq: {freq}", showlegend=False))
    fig_routes.update_layout(margin=dict(l=0,r=0,t=0,b=0))
    st.plotly_chart(fig_routes, use_container_width=True)
else:
    st.info('Para visualizar rotas você precisa de lat/lon e coluna rota (ou origem/destino).')

# 3) Top rotas: barras interativas (tarifa média e freq)
st.markdown('## 3 — Top rotas (barras: frequência e tarifa média)')
if 'rota' in df_f.columns:
    top_table = df_f.groupby('rota').agg(freq=('rota','count'), tarifa_media=('tarifa','mean')).reset_index().sort_values('freq', ascending=False).head(20)
    fig_bar = px.bar(top_table, x='freq', y='rota', orientation='h', text='tarifa_media', height=600)
    fig_bar.update_layout(yaxis={'categoryorder':'total ascending'}, margin=dict(l=200))
    fig_bar.update_traces(hovertemplate='Rota: %{y}<br>Freq: %{x}<br>Tarifa média: R$ %{customdata[0]:.2f}<extra></extra>', customdata=np.stack([top_table['tarifa_media']], axis=-1))
    st.plotly_chart(fig_bar, use_container_width=True)
else:
    st.info('Coluna rota ausente — verifique.')

# 4) Série temporal: tarifa média ao longo do tempo (mensal)
st.markdown('## 4 — Série temporal: tarifa média (mensal)')
if 'data' in df_f.columns:
    ts = df_f.set_index('data').resample('M').agg(tarifa_media=('tarifa','mean')).reset_index()
    fig_ts = px.line(ts, x='data', y='tarifa_media', markers=True, height=400)
    fig_ts.update_layout(xaxis_title='Data', yaxis_title='Tarifa média (R$)', margin=dict(l=20,r=20,t=20,b=20))
    st.plotly_chart(fig_ts, use_container_width=True)
else:
    st.info('Sem coluna data para série temporal.')

# 5) Boxplot por companhia (distribuição de tarifas)
st.markdown('## 5 — Boxplot: tarifa por companhia')
if 'companhia' in df_f.columns:
    # take top companies
    top_comp = df_f['companhia'].value_counts().head(10).index.tolist()
    df_comp = df_f[df_f['companhia'].isin(top_comp)]
    fig_box = px.box(df_comp, x='companhia', y='tarifa', points='all', height=450)
    fig_box.update_traces(hovertemplate='Companhia: %{x}<br>Tarifa: R$ %{y:.2f}<extra></extra>')
    st.plotly_chart(fig_box, use_container_width=True)
else:
    st.info('Coluna companhia ausente.')

# 6) Heatmap: média tarifa por mês x origem (pivot)
st.markdown('## 6 — Heatmap: tarifa média por mês x origem (ou destino)')
if 'mes' in df_f.columns and 'origem' in df_f.columns:
    pivot = df_f.groupby(['mes','origem']).agg(tarifa_media=('tarifa','mean')).reset_index()
    heat = pivot.pivot(index='origem', columns='mes', values='tarifa_media').fillna(0)
    fig_heat = px.imshow(heat, labels=dict(x='Mês', y='Origem', color='Tarifa média (R$)'), aspect='auto', height=600)
    st.plotly_chart(fig_heat, use_container_width=True)
else:
    st.info('Dados insuficientes para heatmap (mes/origem).')

# 7) Distribuição + histogram + KDE (tarifa)
st.markdown('## 7 — Distribuição de tarifas (histograma interativo)')
if 'tarifa' in df_f.columns:
    fig_hist = px.histogram(df_f, x='tarifa', nbins=40, marginal='box', height=420)
    fig_hist.update_layout(xaxis_title='Tarifa (R$)', yaxis_title='Contagem')
    st.plotly_chart(fig_hist, use_container_width=True)
else:
    st.info('Coluna tarifa ausente.')

# 8) Small multiples (opcional) — tarifa média por mês por companhia
st.markdown('## 8 — Small multiples: tarifa média por mês por companhia (top 6)')
if 'companhia' in df_f.columns and 'mes' in df_f.columns:
    comps = df_f['companhia'].value_counts().head(6).index.tolist()
    df_small = df_f[df_f['companhia'].isin(comps)].groupby(['companhia','mes']).agg(tarifa_media=('tarifa','mean')).reset_index()
    fig_small = px.line(df_small, x='mes', y='tarifa_media', color='companhia', facet_col='companhia', facet_col_wrap=3, markers=True, height=700)
    st.plotly_chart(fig_small, use_container_width=True)

# -------------------------
# EXPORT / DOWNLOAD
st.sidebar.header('Exportar')
if st.sidebar.button('Download: relatório (CSV filtrado)'):
    csv = df_f.to_csv(index=False).encode('utf-8')
    st.sidebar.download_button('Clique para baixar CSV filtrado', data=csv, file_name='capitais_filtrado.csv', mime='text/csv')

st.sidebar.markdown('---')
st.sidebar.write('Para suporte e ajustes adicionais - posso adaptar o código ao seu CSV real e criar um mapping IATA->coord se precisar.')

# final note
st.caption('Aplicativo gerado automaticamente — personalize caminhos e mapeamentos IATA conforme necessário.')


