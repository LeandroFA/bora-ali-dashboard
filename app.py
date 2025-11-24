# app.py — Bora Alí (versão final: mapas, 7+ gráficos, insights, rotas melhoradas)
import os
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import pydeck as pdk
from prophet import Prophet
from prophet.plot import plot_plotly
from datetime import datetime

# ---------- Config ----------
CSV_FILE = "INMET_ANAC_ROTAS_APENAS_CAPITAIS.csv"  # deve estar NA RAIZ do repo
OUTPUT_PATH = "outputs"
os.makedirs(OUTPUT_PATH, exist_ok=True)

PRIMARY = "#006DCE"
ACCENT = "#FF6B4A"
BG = "#FBFDFF"
TEXT = "#0F172A"

st.set_page_config(page_title="Bora Alí — Capitais", layout="wide", initial_sidebar_state="expanded")

st.markdown(f"""
    <style>
      body {{background-color:{BG};}}
      h1,h2,h3 {{color:{PRIMARY}; font-weight:800}}
      .stButton>button {{background-color:{PRIMARY}; color:white; border-radius:10px;}}
    </style>
""", unsafe_allow_html=True)

st.title("✈️ Bora Alí — Capitais")
st.caption("Dashboard interativo: tarifas aéreas × clima — mapas, insights e previsão 2026")

# ---------- Load data ----------
@st.cache_data
def load_df(path):
    try:
        df = pd.read_csv(path, low_memory=False)
    except FileNotFoundError:
        st.error(f"Arquivo '{path}' não encontrado na raiz do repositório. Faça upload do CSV e redeploy.")
        st.stop()
    # ensure columns exist and types
    for col in ["ROTA","COMPANHIA","DESTINO","ORIGEM","TARIFA","TEMP_MEDIA","ANO","MES"]:
        if col not in df.columns:
            df[col] = pd.NA
    df["ROTA"] = df["ROTA"].astype(str).str.replace(" - ", " → ").str.strip()
    df["COMPANHIA"] = df["COMPANHIA"].astype(str).str.upper().str.strip()
    df["TARIFA"] = pd.to_numeric(df["TARIFA"], errors="coerce")
    df["TEMP_MEDIA"] = pd.to_numeric(df["TEMP_MEDIA"], errors="coerce")
    df["ANO"] = pd.to_numeric(df["ANO"], errors="coerce").fillna(0).astype(int)
    df["MES"] = pd.to_numeric(df["MES"], errors="coerce").fillna(0).astype(int)
    df["DATA"] = pd.to_datetime(df["ANO"].astype(str) + "-" + df["MES"].astype(str).str.zfill(2) + "-01", errors="coerce")
    return df

df = load_df(CSV_FILE)

# ---------- helper: capitals coords ----------
CAPITAIS = {
    'Rio Branco': (-9.97499, -67.8243),'Maceió': (-9.649847, -35.70895),'Macapá': (0.034934, -51.0694),
    'Manaus': (-3.119028, -60.021731),'Salvador': (-12.97139, -38.50139),'Fortaleza': (-3.71722, -38.543366),
    'Brasília': (-15.793889, -47.882778),'Vitória': (-20.3155, -40.3128),'Goiânia': (-16.686891, -49.264788),
    'São Luís': (-2.52972, -44.30278),'Cuiabá': (-15.601415, -56.097892),'Campo Grande': (-20.4433, -54.6465),
    'Belo Horizonte': (-19.916681, -43.934493),'Belém': (-1.455833, -48.504444),'João Pessoa': (-7.119495, -34.845011),
    'Curitiba': (-25.429596, -49.271272),'Recife': (-8.047562, -34.8770),'Teresina': (-5.08921, -42.8016),
    'Rio de Janeiro': (-22.906847, -43.172896),'Natal': (-5.795, -35.209),'Porto Alegre': (-30.034647, -51.217658),
    'Porto Velho': (-8.7608, -63.9039),'Boa Vista': (2.8196, -60.6733),'Florianópolis': (-27.595377, -48.548046),
    'Aracaju': (-10.9472, -37.0731),'São Paulo': (-23.55052, -46.633308),'Palmas': (-10.184, -48.333)
}

# ---------- Sidebar filters ----------
st.sidebar.header("🎛️ Filtros rápidos (use para ajustar os gráficos)")
anos = sorted(df['ANO'].dropna().unique())
sel_anos = st.sidebar.multiselect("Ano", anos, default=anos)
sel_meses = st.sidebar.multiselect("Mês", list(range(1,13)), default=list(range(1,13)))
companias = sorted(df['COMPANHIA'].dropna().unique())
sel_comp = st.sidebar.multiselect("Companhia", companias, default=companias)

# estação filter
def estacao(m):
    if m in [12,1,2]: return "Verão"
    if m in [3,4,5]: return "Outono"
    if m in [6,7,8]: return "Inverno"
    return "Primavera"

df['ESTACAO'] = df['MES'].apply(estacao)
sel_est = st.sidebar.multiselect("Estação", ["Verão","Outono","Inverno","Primavera"], default=["Verão","Outono","Inverno","Primavera"])

# capitals filter
cap_options = sorted(list(CAPITAIS.keys()))
sel_caps = st.sidebar.multiselect("Capitais (Orig/Dest)", cap_options, default=['São Paulo','Rio de Janeiro','Recife','Brasília','Manaus'])

# apply filters
dff = df[
    (df['ANO'].isin(sel_anos)) &
    (df['MES'].isin(sel_meses)) &
    (df['COMPANHIA'].isin(sel_comp)) &
    (df['ESTACAO'].isin(sel_est))
].copy()

# parse ROTA robust
def parse_rota(rt):
    if pd.isna(rt): return (None,None)
    if '→' in rt:
        p = [x.strip() for x in rt.split('→') if x.strip()]
        if len(p)>=2: return p[0], p[-1]
    if '-' in rt:
        p = [x.strip() for x in rt.split('-') if x.strip()]
        if len(p)>=2: return p[0], p[-1]
    return (None,None)

dff[['ORIG','DEST']] = dff['ROTA'].apply(lambda x: pd.Series(parse_rota(x)))
dff = dff[dff['ORIG'].isin(sel_caps) & dff['DEST'].isin(sel_caps)].copy()

if dff.shape[0]==0:
    st.warning("Nenhum registro após aplicar filtros. Ajuste filtros.")
    st.stop()

# ---------- Prepare datasets for visuals ----------
# agg capitals for point map
agg_cap = dff.groupby('DEST').agg(tarifa_media=('TARIFA','mean'), temp_media=('TEMP_MEDIA','mean'), regs=('TARIFA','count')).reset_index()
agg_cap['lat'] = agg_cap['DEST'].map(lambda x: CAPITAIS.get(x,(np.nan,np.nan))[0])
agg_cap['lon'] = agg_cap['DEST'].map(lambda x: CAPITAIS.get(x,(np.nan,np.nan))[1])

# prepare routes for route map (unique)
routes = dff.groupby(['ROTA']).agg(tarifa_media=('TARIFA','mean'), regs=('TARIFA','count')).reset_index()
routes[['ORIG','DEST']] = routes['ROTA'].apply(lambda x: pd.Series(parse_rota(x)))
routes['olat'] = routes['ORIG'].map(lambda x: CAPITAIS.get(x,(np.nan,np.nan))[0])
routes['olon'] = routes['ORIG'].map(lambda x: CAPITAIS.get(x,(np.nan,np.nan))[1])
routes['dlat'] = routes['DEST'].map(lambda x: CAPITAIS.get(x,(np.nan,np.nan))[0])
routes['dlon'] = routes['DEST'].map(lambda x: CAPITAIS.get(x,(np.nan,np.nan))[1])
routes = routes.dropna(subset=['olat','olon','dlat','dlon']).copy()

# ---------- FIGURE 1: map points (Tarifa size, Temp color) ----------
st.subheader("1) Mapa — Capitais (tamanho = tarifa média · cor = temp média)")
fig1 = px.scatter_mapbox(
    agg_cap.dropna(subset=['lat','lon']),
    lat='lat', lon='lon',
    size='tarifa_media', color='temp_media',
    hover_name='DEST',
    hover_data={'tarifa_media':':.0f', 'temp_media':':.1f', 'lat':False, 'lon':False, 'regs':True},
    color_continuous_scale='thermal', size_max=45, zoom=3.2, height=520
)
fig1.update_layout(mapbox_style='carto-positron', margin={"r":0,"t":0,"l":0,"b":0}, paper_bgcolor="rgba(0,0,0,0)")
st.plotly_chart(fig1, use_container_width=True)

# ---------- FIGURE 2: improved route map (plotly lines) ----------
st.subheader("2) Mapa — Rotas (espessura ≈ tarifa média) — interaja para ver tarifa e temp média")
# Build lines as Scattermapbox traces grouped by tariff bins for thickness control
if not routes.empty:
    # create bins for width
    bins = np.quantile(routes['tarifa_media'], [0,0.25,0.5,0.75,1.0])
    def width_from_tarifa(x):
        if x <= bins[1]: return 1
        if x <= bins[2]: return 2.5
        if x <= bins[3]: return 4
        return 6
    routes['width'] = routes['tarifa_media'].apply(width_from_tarifa)
    # create figure
    fig2 = go.Figure()
    for i,row in routes.iterrows():
        fig2.add_trace(go.Scattermapbox(
            lat=[row['olat'], row['dlat']],
            lon=[row['olon'], row['dlon']],
            mode='lines',
            line=dict(width=row['width'], color=PRIMARY, opacity=0.7),
            hoverinfo='text',
            text=f"<b>Rota:</b> {row['ROTA']}<br><b>Tarifa média:</b> R$ {row['tarifa_media']:.0f}<br><b>Registros:</b> {int(row['regs'])}"
        ))
    # add points for capitals (without showing lat/lon in hover)
    fig2.add_trace(go.Scattermapbox(
        lat=agg_cap['lat'], lon=agg_cap['lon'], mode='markers+text',
        marker=go.scattermapbox.Marker(size=12, color=ACCENT),
        hoverinfo='text',
        text=agg_cap['DEST'],
        textposition='top right',
        textfont=dict(size=11),
        customdata=np.stack([agg_cap['tarifa_media'].round(0), agg_cap['temp_media'].round(1), agg_cap['regs']], axis=1),
        hovertemplate="<b>%{text}</b><br>Tarifa média: R$ %{customdata[0]:.0f}<br>Temp média: %{customdata[1]:.1f} °C<br>Registros: %{customdata[2]}<extra></extra>"
    ))
    fig2.update_layout(mapbox_style='carto-positron', mapbox_center={"lat":-14.2,"lon":-51.9}, mapbox_zoom=3.2, margin={"r":0,"t":0,"l":0,"b":0}, height=520)
    st.plotly_chart(fig2, use_container_width=True)
else:
    st.info("Sem rotas disponíveis para os filtros selecionados.")

# ---------- FIGURE 3: time series (tarifa média nacional / por seleção) ----------
st.subheader("3) Série temporal — Tarifa média ao longo do tempo (agregada por mês)")
ts = dff.groupby('DATA').agg(tarifa=('TARIFA','mean'), temp=('TEMP_MEDIA','mean')).reset_index().sort_values('DATA')
fig3 = px.line(ts, x='DATA', y='tarifa', markers=True, title="Tarifa média mensal")
fig3.update_layout(yaxis_title="Tarifa média (R$)")
st.plotly_chart(fig3, use_container_width=True)

# ---------- FIGURE 4: tarifas por estação (bar com labels) ----------
st.subheader("4) Tarifa média por estação (R$)")
est = dff.groupby('ESTACAO').agg(tarifa=('TARIFA','mean')).reindex(['Verão','Outono','Inverno','Primavera']).reset_index()
est['tarifa'] = est['tarifa'].round(0)
fig4 = px.bar(est, x='ESTACAO', y='tarifa', text='tarifa', color='ESTACAO',
              color_discrete_sequence=[PRIMARY, ACCENT, GOLD, PURPLE], title="Tarifa média por estação")
fig4.update_traces(texttemplate="R$ %{text:.0f}", textposition='outside')
fig4.update_layout(yaxis_title="Tarifa média (R$)")
st.plotly_chart(fig4, use_container_width=True)

# ---------- FIGURE 5: tarifa por região (bar com labels) ----------
st.subheader("5) Tarifa média por região do Brasil")
REGIOES = {
    "Norte": ["Belém","Macapá","Manaus","Boa Vista","Rio Branco","Porto Velho","Palmas"],
    "Nordeste": ["São Luís","Teresina","Fortaleza","Natal","João Pessoa","Recife","Maceió","Aracaju","Salvador"],
    "Centro-Oeste": ["Brasília","Goiânia","Campo Grande","Cuiabá"],
    "Sudeste": ["São Paulo","Rio de Janeiro","Belo Horizonte","Vitória"],
    "Sul": ["Curitiba","Florianópolis","Porto Alegre"]
}
def get_reg(c):
    for k,v in REGIOES.items():
        if c in v: return k
    return "Outro"

dff['REGIAO'] = dff['DEST'].apply(get_reg)
reg = dff.groupby('REGIAO').agg(tarifa=('TARIFA','mean')).reset_index()
reg['tarifa'] = reg['tarifa'].round(0)
fig5 = px.bar(reg, x='REGIAO', y='tarifa', text='tarifa', color='REGIAO',
              color_discrete_sequence=[PRIMARY, ACCENT, GREEN, PURPLE, RED], title="Tarifa média por região")
fig5.update_traces(texttemplate="R$ %{text:.0f}", textposition='outside')
fig5.update_layout(yaxis_title="Tarifa média (R$)")
st.plotly_chart(fig5, use_container_width=True)

# ---------- FIGURE 6: companhias — boxplot + média (comparativo) ----------
st.subheader("6) Companhias — distribuição de tarifas (boxplot) e média")
comp_df = dff[dff['COMPANHIA'].notna()].copy()
if comp_df.shape[0] > 10:
    fig6 = px.box(comp_df, x='COMPANHIA', y='TARIFA', points='outliers', title="Distribuição de tarifas por companhia")
    fig6.update_layout(xaxis_tickangle=-45, yaxis_title="Tarifa (R$)")
    st.plotly_chart(fig6, use_container_width=True)
    # mean bar
    mean_comp = comp_df.groupby('COMPANHIA').agg(tarifa=('TARIFA','mean')).reset_index().sort_values('tarifa', ascending=False)
    fig6b = px.bar(mean_comp.head(10), x='COMPANHIA', y='tarifa', text='tarifa', title="Top 10 médias por companhia")
    fig6b.update_traces(texttemplate="R$ %{text:.0f}", textposition='outside')
    st.plotly_chart(fig6b, use_container_width=True)
else:
    st.info("Dados insuficientes para comparar companhias com robustez.")

# ---------- FIGURE 7: heatmap mês x capital (tarifa média) ----------
st.subheader("7) Heatmap — Tarifa média (mês × capital)")
heat = dff.groupby([dff['DATA'].dt.month.rename('MES'),'DEST']).agg(tarifa=('TARIFA','mean')).reset_index()
heat_pivot = heat.pivot(index='DEST', columns='MES', values='tarifa').fillna(0)
# ensure months order 1..12
heat_pivot = heat_pivot.reindex(columns=sorted(heat_pivot.columns))
fig7 = px.imshow(heat_pivot, aspect='auto', labels=dict(x="Mês", y="Capital", color="Tarifa (R$)"),
                 x=[str(m) for m in heat_pivot.columns], y=heat_pivot.index)
st.plotly_chart(fig7, use_container_width=True)

# ---------- INSIGHTS dinâmicos (texto automático) ----------
st.markdown("---")
st.header("Insights automáticos — Bora Dicas")
# top cidade cara
top_city = agg_cap.loc[agg_cap['tarifa_media'].idxmax()]
low_city = agg_cap.loc[agg_cap['tarifa_media'].idxmin()]
# season with highest avg tariff
season_avg = dff.groupby('ESTACAO').agg(tarifa=('TARIFA','mean')).reset_index()
best_season = season_avg.loc[season_avg['tarifa'].idxmin()]['ESTACAO'] if season_avg.shape[0]>0 else None
c1, c2, c3 = st.columns(3)
c1.metric("Capital mais cara (média)", f"{top_city['DEST']}", f"R$ {top_city['tarifa_media']:.0f}")
c2.metric("Capital mais barata (média)", f"{low_city['DEST']}", f"R$ {low_city['tarifa_media']:.0f}")
c3.metric("Estação com tarifa média menor (sugestão viagem)", f"{best_season if best_season else '-'}")

# generate short textual insights
insights = []
insights.append(f"➡ Entre as capitais filtradas, **{top_city['DEST']}** tem a tarifa média mais alta (R$ {top_city['tarifa_media']:.0f}).")
insights.append(f"➡ A capital com tarifa média mais baixa é **{low_city['DEST']}** (R$ {low_city['tarifa_media']:.0f}).")
if best_season:
    insights.append(f"➡ Geralmente, viajar na **{best_season}** tende a ser mais barato com base nos filtros atuais.")
# add month with highest avg tariff (national)
month_avg = dff.groupby('MES').agg(tarifa=('TARIFA','mean')).round(0).reset_index()
if month_avg.shape[0]>0:
    best_month = month_avg.loc[month_avg['tarifa'].idxmin()]['MES']
    insights.append(f"➡ Mês com menor tarifa média (hoje): {int(best_month)} — avalie comprar com antecedência.")
for s in insights:
    st.markdown(s)

# ---------- Export: save main figures as PNG + CSV summary ----------
if st.button("Exportar 6 gráficos (PNG) + CSV resumo"):
    try:
        # save CSV summary
        summary_path = os.path.join(OUTPUT_PATH, "boraali_summary.csv")
        agg_cap.to_csv(summary_path, index=False)
        # save figures as static images (Plotly can save if Kaleido is installed on Streamlit Cloud; we'll attempt)
        import plotly.io as pio
        pio.write_image(fig1, os.path.join(OUTPUT_PATH, "map_capitais.png"), width=1600, height=900, scale=1)
        pio.write_image(fig2, os.path.join(OUTPUT_PATH, "rotas.png"), width=1600, height=900, scale=1)
        pio.write_image(fig3, os.path.join(OUTPUT_PATH, "serie_tarifa.png"), width=1600, height=900, scale=1)
        pio.write_image(fig4, os.path.join(OUTPUT_PATH, "tarifa_estacoes.png"), width=1600, height=900, scale=1)
        pio.write_image(fig5, os.path.join(OUTPUT_PATH, "tarifa_regioes.png"), width=1600, height=900, scale=1)
        pio.write_image(fig7, os.path.join(OUTPUT_PATH, "heatmap.png"), width=1600, height=900, scale=1)
        st.success(f"Exportados: {summary_path} + 6 imagens PNG em {OUTPUT_PATH}")
    except Exception as e:
        st.warning(f"Export parcial (kaleido pode não estar disponível). CSV salvo: {summary_path}. Erro imagem: {e}")

st.markdown("---")
st.caption("Arquivo de referência (design): /mnt/data/Bora Alí — Dashboard (Capitais) · Streamlit.pdf")
st.markdown("Made with 💙 — Bora Alí")



