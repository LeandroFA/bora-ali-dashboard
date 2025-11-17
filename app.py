
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import unicodedata

st.set_page_config(page_title="Bora Alí — Dashboard (Capitais)", layout="wide", initial_sidebar_state="expanded")

# ---------------- utilitários ----------------
def normaliza(texto):
    """Normaliza string (remove acentos, lower, strip)."""
    if pd.isna(texto):
        return ""
    s = str(texto).strip().lower()
    s = unicodedata.normalize("NFKD", s)
    return "".join([c for c in s if not unicodedata.combining(c)])

def parse_rota(rota_str):
    """Tenta extrair origem e destino a partir da string da rota."""
    if pd.isna(rota_str):
        return None, None
    s = str(rota_str)
    delims = [" - ", "–", "--", ">", "→", "/", "|", " / ", "-"]
    for d in delims:
        if d in s:
            parts = [p.strip() for p in s.split(d) if p.strip()]
            if len(parts) >= 2:
                return parts[0], parts[-1]
    parts = s.split()
    if len(parts) == 2:
        return parts[0], parts[1]
    return None, None

# ---------------- Dicionário de coordenadas das 27 capitais brasileiras ----------------
# Chaves em formato normalizado (sem acento, minúsculo)
capitais_coords = {
    "sao paulo": (-23.55052, -46.633308),
    "rio de janeiro": (-22.906847, -43.172896),
    "brasilia": (-15.826691, -47.921822),
    "salvador": (-12.977749, -38.501629),
    "belo horizonte": (-19.920683, -43.937148),
    "curitiba": (-25.428954, -49.267137),
    "porto alegre": (-30.027708, -51.228734),
    "recife": (-8.047562, -34.877),
    "fortaleza": (-3.71722, -38.5434),
    "manaus": (-3.10194, -60.025),
    "belem": (-1.455754, -48.490179),
    "goiania": (-16.686891, -49.264788),
    "maceio": (-9.66599, -35.735),
    "natal": (-5.79448, -35.211),
    "joao pessoa": (-7.119495, -34.845011),
    "aracaju": (-10.947247, -37.073082),
    "teresina": (-5.091944, -42.80339),
    "palmas": (-10.1849, -48.3336),
    "cuiaba": (-15.601417, -56.09789),
    "campo grande": (-20.44278, -54.6469),
    "vitoria": (-20.3155, -40.3128),
    "florianopolis": (-27.5969, -48.5495),
    "sao luis": (-2.52972, -44.30278),
    "macapa": (0.034934, -51.069389),
    "boa vista": (2.819444, -60.673333),
    "rio branco": (-9.97499, -67.8243),
    "palmas": (-10.1849, -48.3336),
    "manaus": (-3.10194, -60.025)
}

def obter_coords(cidade):
    k = normaliza(cidade)
    return capitais_coords.get(k, (None, None))

# ---------------- Carregar dados ----------------
@st.cache_data
def carregar_dados():
    # lê CSVs (assume que estão no mesmo repositório)
    df = pd.read_csv("INMET_ANAC_ROTAS_APENAS_CAPITAIS.csv", low_memory=False)
    ipca = pd.read_csv("IPCAUNIFICADO.csv", low_memory=False)
    # padroniza nomes de coluna caso venham em formatos diferentes (minúsculas/maiúsculas)
    col_map = {}
    for c in df.columns:
        col_map[c] = c.strip()
    df.rename(columns=col_map, inplace=True)
    # garantir colunas esperadas existam (caso sensível)
    for c in ["COMPANHIA","ANO","MES","DESTINO","ROTA","TARIFA","TEMP_MEDIA"]:
        if c not in df.columns:
            df[c] = np.nan
    # converter tipos
    df["ANO"] = pd.to_numeric(df["ANO"], errors="coerce").astype("Int64")
    df["MES"] = pd.to_numeric(df["MES"], errors="coerce").astype("Int64")
    # adicionar colunas de lat/lon a partir de DESTINO
    lats, lons, lats_o, lons_o = [], [], [], []
    for _, row in df.iterrows():
        dest = row.get("DESTINO")
        lat, lon = obter_coords(dest)
        lats.append(lat)
        lons.append(lon)
        # tentar extrair origem da coluna ROTA
        origem, destino_parsed = parse_rota(row.get("ROTA"))
        if origem:
            lat_o, lon_o = obter_coords(origem)
        else:
            lat_o, lon_o = (None, None)
        lats_o.append(lat_o)
        lons_o.append(lon_o)
    df["LAT"] = lats
    df["LON"] = lons
    df["LAT_ORIGEM"] = lats_o
    df["LON_ORIGEM"] = lons_o
    return df, ipca

df, ipca = carregar_dados()

# ---------------- Sidebar (filtros) ----------------
st.sidebar.title("Filtros")
st.sidebar.markdown("Filtre os dados para explorar capitais, períodos e companhias.")

anos_disponiveis = sorted([int(x) for x in df["ANO"].dropna().unique()]) if df["ANO"].notna().any() else []
meses_disponiveis = sorted([int(x) for x in df["MES"].dropna().unique()]) if df["MES"].notna().any() else []
capitais_disponiveis = sorted(df["DESTINO"].dropna().unique().tolist())
companhias_disponiveis = sorted(df["COMPANHIA"].dropna().unique().tolist())

sel_anos = st.sidebar.multiselect("Ano", anos_disponiveis, default=anos_disponiveis)
sel_meses = st.sidebar.multiselect("Mês", meses_disponiveis, default=meses_disponiveis)
sel_capitais = st.sidebar.multiselect("Capital (DESTINO)", capitais_disponiveis, default=capitais_disponiveis[:6])
sel_companhias = st.sidebar.multiselect("Companhia", companhias_disponiveis, default=companhias_disponiveis)

# aplicar filtros
filtro = df.copy()
if sel_anos:
    filtro = filtro[filtro["ANO"].isin(sel_anos)]
if sel_meses:
    filtro = filtro[filtro["MES"].isin(sel_meses)]
if sel_capitais:
    filtro = filtro[filtro["DESTINO"].isin(sel_capitais)]
if sel_companhias:
    filtro = filtro[filtro["COMPANHIA"].isin(sel_companhias)]

# ---------------- Layout principal ----------------
st.title("🚀 Bora Alí — Painel de Controle (Capitais)")
st.markdown("Painel interativo com mapas, rotas e análises por companhia. Ideal para apresentações SR2.")

# KPIs
k1, k2, k3, k4 = st.columns(4)
k1.metric("Registros (filtrados)", f"{filtro.shape[0]:,}")
k2.metric("Tarifa média (R$)", f"{round(filtro['TARIFA'].mean(),2) if filtro['TARIFA'].notna().any() else '—'}")
k3.metric("Temperatura média (°C)", f"{round(filtro['TEMP_MEDIA'].mean(),2) if filtro['TEMP_MEDIA'].notna().any() else '—'}")
k4.metric("Rotas únicas", f"{int(filtro['ROTA'].nunique()) if 'ROTA' in filtro.columns else '—'}")

st.markdown("---")

# ---------------- Painel: Séries temporais ----------------
st.header("Séries temporais — Tarifa média")
if filtro["TARIFA"].notna().any() and filtro["ANO"].notna().any() and filtro["MES"].notna().any():
    ts = filtro.groupby(["ANO","MES","DESTINO"])["TARIFA"].mean().reset_index()
    ts["DATA"] = pd.to_datetime(ts["ANO"].astype(str) + "-" + ts["MES"].astype(str) + "-01", errors="coerce")
    fig_ts = px.line(ts, x="DATA", y="TARIFA", color="DESTINO", markers=True,
                     title="Tarifa média mensal por capital")
    fig_ts.update_layout(legend_title_text="Capital")
    st.plotly_chart(fig_ts, use_container_width=True)
else:
    st.info("Dados insuficientes para série temporal (verifique ANO/MES/TARIFA).")

# ---------------- Painel: Tarifa x Temperatura ----------------
st.header("Tarifa × Temperatura média")
if filtro["TARIFA"].notna().any() and filtro["TEMP_MEDIA"].notna().any():
    fig_scatter = px.scatter(filtro, x="TEMP_MEDIA", y="TARIFA", color="COMPANHIA",
                             hover_data=["DESTINO","ROTA"], trendline="ols",
                             title="Relação entre temperatura média e tarifa (por companhia)")
    st.plotly_chart(fig_scatter, use_container_width=True)
else:
    st.info("Dados insuficientes para o gráfico Tarifa × Temperatura.")

# ---------------- Painel: Mapa de Capitais (pontos) ----------------
st.header("Mapa — Capitais (tarifa média por ponto)")
map_df = filtro.dropna(subset=["LAT","LON"]).groupby("DESTINO").agg(
    LAT=("LAT","first"), LON=("LON","first"), TARIFA_MEDIA=("TARIFA","mean"), QTDE=("TARIFA","count")
).reset_index()

if map_df.empty:
    st.warning("Nenhuma coordenada encontrada para as capitais selecionadas. Verifique os nomes em 'DESTINO'.")
else:
    fig_map = px.scatter_mapbox(map_df, lat="LAT", lon="LON",
                                size="TARIFA_MEDIA", color="TARIFA_MEDIA",
                                hover_name="DESTINO", hover_data={"TARIFA_MEDIA":":.2f","QTDE":True},
                                zoom=3, height=520,
                                title="Tarifa média por capital")
    fig_map.update_layout(mapbox_style="carto-positron", margin={"r":0,"t":40,"l":0,"b":0})
    st.plotly_chart(fig_map, use_container_width=True)

# ---------------- Painel: Mapa de Rotas (linhas) ----------------
st.header("Mapa — Rotas entre capitais")
st.markdown("O app tenta extrair origem/destino a partir da coluna `ROTA` (ex: 'São Paulo - Rio de Janeiro'). Se as rotas estiverem em siglas IATA (ex: 'GRU-GIG'), pode ser necessário fornecer mapeamento IATA → cidade.")

mostrar_rotas = st.checkbox("Mostrar rotas (linhas)", value=True)
top_n = st.slider("Top N rotas por frequência", 5, 30, 10)

# preparar rotas
rotas = filtro.copy()
rotas["ROTA_STR"] = rotas["ROTA"].astype(str)
rotas_agr = rotas.groupby("ROTA_STR").agg(FREQ=("ROTA_STR","count"), TARIFA_MEDIA=("TARIFA","mean")).reset_index()
rotas_agr = rotas_agr.sort_values("FREQ", ascending=False).head(top_n)

linhas = []
for _, r in rotas_agr.iterrows():
    rota_text = r["ROTA_STR"]
    origem, destino = parse_rota(rota_text)
    if origem and destino:
        lat_o, lon_o = obter_coords(origem)
        lat_d, lon_d = obter_coords(destino)
        if None not in (lat_o, lon_o, lat_d, lon_d):
            linhas.append({
                "origem": origem, "destino": destino,
                "lat_o": lat_o, "lon_o": lon_o, "lat_d": lat_d, "lon_d": lon_d,
                "freq": int(r["FREQ"]), "tarifa_media": float(r["TARIFA_MEDIA"]) if not pd.isna(r["TARIFA_MEDIA"]) else None
            })

if mostrar_rotas:
    if linhas:
        fig = go.Figure()
        # pontos (capitais)
        if not map_df.empty:
            fig.add_trace(go.Scattermapbox(
                lat=map_df["LAT"], lon=map_df["LON"], mode="markers",
                marker=go.scattermapbox.Marker(size=8),
                text=map_df["DESTINO"], hoverinfo="text"
            ))
        max_freq = max([ln["freq"] for ln in linhas]) if linhas else 1
        for ln in linhas:
            largura = 1 + 4 * (ln["freq"] / max_freq)
            hovertxt = f"{ln['origem']} → {ln['destino']} | freq: {ln['freq']} | tarifa média: {ln['tarifa_media']:.2f}" if ln['tarifa_media'] else f"{ln['origem']} → {ln['destino']} | freq: {ln['freq']}"
            fig.add_trace(go.Scattermapbox(
                lat=[ln["lat_o"], ln["lat_d"]], lon=[ln["lon_o"], ln["lon_d"]],
                mode="lines", line=dict(width=largura, color="royalblue"), hoverinfo="text", text=hovertxt
            ))
        fig.update_layout(mapbox_style="carto-positron", mapbox_center={"lat":-14.2,"lon":-51.9}, mapbox_zoom=3,
                          margin={"r":0,"t":40,"l":0,"b":0}, height=600)
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Não foi possível extrair origens/destinos válidos das rotas. Padronize a coluna `ROTA` ou forneça mapeamento IATA → cidade.")

# ---------------- Painel: Análise por Companhia ----------------
st.header("Análise por Companhia aérea")
st.markdown("Compare companhias: tarifa média, número de registros, e série temporal por companhia.")

if filtro["COMPANHIA"].notna().any():
    comp_tab = filtro.groupby("COMPANHIA").agg(Registros=("COMPANHIA","count"), Tarifa_Media=("TARIFA","mean")).reset_index()
    comp_tab = comp_tab.sort_values("Registros", ascending=False)
    st.subheader("Resumo por companhia")
    st.dataframe(comp_tab.style.format({"Tarifa_Media":"{:.2f}"}))

    # gráfico: tarifa média por companhia
    st.subheader("Tarifa média por companhia")
    fig_comp = px.bar(comp_tab, x="COMPANHIA", y="Tarifa_Media", text=comp_tab["Tarifa_Media"].round(2),
                      title="Tarifa média por companhia (filtradas seleções)", labels={"Tarifa_Media":"Tarifa média (R$)"})
    fig_comp.update_layout(xaxis_tickangle=-45)
    st.plotly_chart(fig_comp, use_container_width=True)

    # série temporal por companhia (se houver ANO/MES)
    st.subheader("Série temporal — Tarifa média por companhia")
    ts_comp = filtro.groupby(["ANO","MES","COMPANHIA"])["TARIFA"].mean().reset_index()
    if not ts_comp.empty:
        ts_comp["DATA"] = pd.to_datetime(ts_comp["ANO"].astype(str) + "-" + ts_comp["MES"].astype(str) + "-01", errors="coerce")
        fig_ts_comp = px.line(ts_comp, x="DATA", y="TARIFA", color="COMPANHIA", title="Tarifa média ao longo do tempo por companhia", markers=True)
        st.plotly_chart(fig_ts_comp, use_container_width=True)
else:
    st.info("Não foram encontrados registros de companhia no conjunto filtrado.")

# ---------------- Top rotas ----------------
st.header("Top rotas — frequência e tarifa média")
if "ROTA" in filtro.columns:
    top_rotas = filtro.groupby("ROTA").agg(Frequencia=("ROTA","count"), Tarifa_Media=("TARIFA","mean")).reset_index()
    top_rotas = top_rotas.sort_values("Frequencia", ascending=False).head(20)
    st.dataframe(top_rotas.style.format({"Tarifa_Media":"{:.2f}"}))
else:
    st.info("Coluna `ROTA` ausente no dataset.")

# ---------------- Correlação ----------------
st.header("Matriz de Correlação")
corr_cols = [c for c in ["TARIFA","TEMP_MEDIA","MES","ANO"] if c in filtro.columns]
if len(corr_cols) >= 2:
    corr = filtro[corr_cols].corr()
    fig_corr = px.imshow(corr, text_auto=True, title="Correlação entre variáveis selecionadas")
    st.plotly_chart(fig_corr, use_container_width=True)
else:
    st.info("Não há colunas suficientes para calcular correlação.")

# ---------------- Rodapé / instruções ----------------
st.markdown("---")
st.caption("Observação: o mapa usa um dicionário interno de coordenadas para as capitais. Se algum ponto não aparecer, verifique o nome exato na coluna DESTINO e me passe para eu adicionar ao dicionário. Se suas rotas usam códigos IATA (ex: 'GRU-GIG'), podemos incluir um mapeamento IATA → capital para desenhar as rotas corretamente.")

