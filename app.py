# app.py — Bora Alí (versão final robusta e normalizadora)
import os
import unicodedata
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import pydeck as pdk
from prophet import Prophet
from prophet.plot import plot_plotly

# -----------------------
# Config
# -----------------------
CSV_FILE = "INMET_ANAC_ROTAS_APENAS_CAPITAIS.csv"  # deve estar na raiz
OUTPUT_DIR = "outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

PRIMARY = "#006DCE"
ACCENT = "#FF6B4A"
BG = "#F7F9FB"
TEXT = "#0F172A"

st.set_page_config("Bora Alí — Capitais", layout="wide")
st.markdown(f"""<style>
body {{background:{BG};}}
h1,h2,h3{{color:{PRIMARY};font-weight:800}}
.stButton>button{{background:{PRIMARY};color:white;border-radius:8px;font-weight:700}}
</style>""", unsafe_allow_html=True)

st.title("✈️ Bora Alí — Capitais (Versão Final)")

# -----------------------
# Canonical capitals list (Portuguese names)
# -----------------------
CANONICAL_CAPITAIS = [
    "Rio Branco","Maceió","Macapá","Manaus","Salvador","Fortaleza","Brasília","Vitória","Goiânia",
    "São Luís","Cuiabá","Campo Grande","Belo Horizonte","Belém","João Pessoa","Curitiba","Recife",
    "Teresina","Rio de Janeiro","Natal","Porto Alegre","Porto Velho","Boa Vista","Florianópolis",
    "Aracaju","São Paulo","Palmas"
]

# Build normalized -> canonical mapping
def normalize_str(s):
    if pd.isna(s):
        return s
    s = str(s)
    # remove accents
    s = ''.join(ch for ch in unicodedata.normalize('NFKD', s) if not unicodedata.combining(ch))
    s = s.replace("_", " ").replace("-", " ")
    s = " ".join(s.split())
    s = s.strip().lower()
    return s

NORMAL_MAP = {normalize_str(c): c for c in CANONICAL_CAPITAIS}

# helper to map fuzzy city to canonical if possible
def map_to_canonical(city):
    if pd.isna(city):
        return city
    n = normalize_str(city)
    # direct match
    if n in NORMAL_MAP:
        return NORMAL_MAP[n]
    # try heuristics: remove accents, title-case, replace multiple spaces, try first token match
    # token match
    tokens = n.split()
    for t in tokens:
        # try match token against normalized canonical tokens
        for kc, vc in NORMAL_MAP.items():
            if t in kc.split():
                return vc
    # fallback: title-case cleaned string (best effort)
    return city.strip().title()

# -----------------------
# Load data (robust)
# -----------------------
@st.cache_data(ttl=600)
def load_data(path):
    try:
        df = pd.read_csv(path, low_memory=False)
    except FileNotFoundError:
        st.error(f"Arquivo '{path}' não encontrado na raiz do repositório. Faça upload e redeploy.")
        st.stop()
    # ensure minimal columns exist
    expected = ["ROTA","COMPANHIA","ORIGEM","DESTINO","TARIFA","TEMP_MEDIA","ANO","MES"]
    for col in expected:
        if col not in df.columns:
            df[col] = pd.NA
    # normalize route formatting
    df["ROTA"] = df["ROTA"].astype(str).str.replace(" - ", " → ").str.strip()
    # if ORIGEM/DESTINO empty, attempt to parse ROTA
    def parse_route_to_cols(row):
        rota = row.get("ROTA","")
        if pd.isna(row.get("ORIGEM")) or pd.isna(row.get("DESTINO")):
            if "→" in rota:
                parts = [p.strip() for p in rota.split("→") if p.strip()]
                if len(parts)>=2:
                    return parts[0], parts[-1]
            if "-" in rota:
                parts = [p.strip() for p in rota.split("-") if p.strip()]
                if len(parts)>=2:
                    return parts[0], parts[-1]
        return row.get("ORIGEM"), row.get("DESTINO")
    parsed = df.apply(lambda r: pd.Series(parse_route_to_cols(r), index=["_orig_parsed","_dest_parsed"]), axis=1)
    # prefer explicit ORIGEM/DESTINO, else parsed
    df["ORIG"] = df.apply(lambda r: r["ORIGEM"] if pd.notna(r["ORIGEM"]) and str(r["ORIGEM"]).strip()!="" else r["_orig_parsed"], axis=1)
    df["DEST"] = df.apply(lambda r: r["DESTINO"] if pd.notna(r["DESTINO"]) and str(r["DESTINO"]).strip()!="" else r["_dest_parsed"], axis=1)
    # apply normalization mapping to ORIG and DEST
    df["ORIG_raw"] = df["ORIG"].astype(str)
    df["DEST_raw"] = df["DEST"].astype(str)
    df["ORIG"] = df["ORIG_raw"].apply(map_to_canonical)
    df["DEST"] = df["DEST_raw"].apply(map_to_canonical)
    # numeric conversions
    df["TARIFA"] = pd.to_numeric(df["TARIFA"], errors="coerce")
    df["TEMP_MEDIA"] = pd.to_numeric(df["TEMP_MEDIA"], errors="coerce")
    df["ANO"] = pd.to_numeric(df["ANO"], errors="coerce").fillna(0).astype(int)
    df["MES"] = pd.to_numeric(df["MES"], errors="coerce").fillna(0).astype(int)
    # date
    df["DATA"] = pd.to_datetime(df["ANO"].astype(str) + "-" + df["MES"].astype(str).str.zfill(2) + "-01", errors="coerce")
    return df

df = load_data(CSV_FILE)

# -----------------------
# Sidebar filters (powerful & clean)
# -----------------------
st.sidebar.header("Filtros — Bora Alí")
anos = sorted(df["ANO"].dropna().unique())
sel_anos = st.sidebar.multiselect("Ano", anos, default=anos)
meses = st.sidebar.multiselect("Mês", list(range(1,13)), default=list(range(1,13)))
companias = sorted(df["COMPANHIA"].dropna().unique())
sel_comp = st.sidebar.multiselect("Companhia", companias, default=companias if companias else [])
# estaciones
def estacao_of(m):
    if m in [12,1,2]: return "Verão"
    if m in [3,4,5]: return "Outono"
    if m in [6,7,8]: return "Inverno"
    return "Primavera"
df["ESTACAO"] = df["MES"].apply(estacao_of)
estacoes = ["Verão","Outono","Inverno","Primavera"]
sel_est = st.sidebar.multiselect("Estação", estacoes, default=estacoes)
# capitals choice (all canonical by default)
sel_caps = st.sidebar.multiselect("Capitais (orig/dest)", CANONICAL_CAPITAIS, default=CANONICAL_CAPITAIS)

# apply filters
dff = df[
    (df["ANO"].isin(sel_anos)) &
    (df["MES"].isin(meses)) &
    (df["COMPANHIA"].isin(sel_comp)) &
    (df["ESTACAO"].isin(sel_est))
].copy()

# ensure ORIG/DEST are canonical (re-map any leftover)
dff["ORIG"] = dff["ORIG"].apply(lambda x: map_to_canonical(x) if pd.notna(x) else x)
dff["DEST"] = dff["DEST"].apply(lambda x: map_to_canonical(x) if pd.notna(x) else x)

# keep only records where both ORIG and DEST are in selected capitals (and in canonical list)
dff = dff[dff["ORIG"].isin(sel_caps) & dff["DEST"].isin(sel_caps)].copy()

if dff.shape[0] == 0:
    st.warning("Nenhum registro após filtros. Ajuste filtros (Ano, Mês, Companhia, Capitais, Estação).")
    st.stop()

# -----------------------
# Capitals coords (canonical)
# -----------------------
CAP_COORDS = {
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
# attach coords to dff (for maps)
dff["LAT"] = dff["DEST"].map(lambda x: CAP_COORDS.get(x,(np.nan,np.nan))[0])
dff["LON"] = dff["DEST"].map(lambda x: CAP_COORDS.get(x,(np.nan,np.nan))[1])

# -----------------------
# Top KPIs
# -----------------------
k1, k2, k3, k4 = st.columns(4)
k1.metric("Registros (filtrados)", f"{len(dff):,}")
k2.metric("Tarifa média (R$)", f"{dff['TARIFA'].mean():.0f}")
k3.metric("Temp média (°C)", f"{dff['TEMP_MEDIA'].mean():.1f}")
k4.metric("Rotas únicas", f"{dff['ROTA'].nunique():,}")

st.markdown("---")

# -----------------------
# FIG 1 — Map points (tarifa size, temp color). Hide coords in hover.
# -----------------------
st.subheader("1) Mapa — Capitais (tamanho = tarifa média · cor = temp média)")
agg_cap = dff.groupby("DEST").agg(tarifa_media=("TARIFA","mean"), temp_media=("TEMP_MEDIA","mean"), registros=("TARIFA","count")).reset_index()
agg_cap["lat"] = agg_cap["DEST"].map(lambda x: CAP_COORDS.get(x,(np.nan,np.nan))[0])
agg_cap["lon"] = agg_cap["DEST"].map(lambda x: CAP_COORDS.get(x,(np.nan,np.nan))[1])
fig1 = px.scatter_mapbox(
    agg_cap.dropna(subset=["lat","lon"]),
    lat="lat", lon="lon",
    size="tarifa_media", color="temp_media",
    hover_name="DEST",
    hover_data={"tarifa_media":":.0f", "temp_media":":.1f", "registros":True, "lat":False, "lon":False},
    color_continuous_scale="thermal", size_max=45, zoom=3.2, height=520
)
fig1.update_layout(mapbox_style="carto-positron", margin={"r":0,"t":0,"l":0,"b":0})
st.plotly_chart(fig1, use_container_width=True)

# -----------------------
# FIG 2 — Routes map (Plotly lines for better hover control). Hover shows only route, tarifa, registros.
# -----------------------
st.subheader("2) Mapa — Rotas entre Capitais (espessura ≈ tarifa média)")

routes = (
    dff.groupby("ROTA")
    .agg(tarifa_media=("TARIFA","mean"), regs=("TARIFA","count"))
    .reset_index()
)
routes[["ORIG","DEST"]] = routes["ROTA"].apply(lambda r: pd.Series(parse_rota if False else [ (r.split("→")[0].strip() if "→" in r else r.split("-")[0].strip()),
                                                                                              (r.split("→")[-1].strip() if "→" in r else r.split("-")[-1].strip()) ]))
# safe attach coords
routes["olat"] = routes["ORIG"].map(lambda x: CAP_COORDS.get(map_to_canonical(x),(np.nan,np.nan))[0])
routes["olon"] = routes["ORIG"].map(lambda x: CAP_COORDS.get(map_to_canonical(x),(np.nan,np.nan))[1])
routes["dlat"] = routes["DEST"].map(lambda x: CAP_COORDS.get(map_to_canonical(x),(np.nan,np.nan))[0])
routes["dlon"] = routes["DEST"].map(lambda x: CAP_COORDS.get(map_to_canonical(x),(np.nan,np.nan))[1])
routes = routes.dropna(subset=["olat","olon","dlat","dlon"]).copy()

if not routes.empty:
    bins = np.quantile(routes["tarifa_media"], [0,0.25,0.5,0.75,1.0])
    def width_from_tarifa(x):
        if x <= bins[1]: return 1
        if x <= bins[2]: return 2.5
        if x <= bins[3]: return 4
        return 6
    routes["width"] = routes["tarifa_media"].apply(width_from_tarifa)

    fig2 = go.Figure()
    # add route lines
    for _, r in routes.iterrows():
        fig2.add_trace(go.Scattermapbox(
            lat=[r["olat"], r["dlat"]],
            lon=[r["olon"], r["dlon"]],
            mode="lines",
            line=dict(width=r["width"], color=PRIMARY, opacity=0.75),
            hoverinfo="text",
            text=f"<b>Rota:</b> {r['ROTA']}<br><b>Tarifa média:</b> R$ {r['tarifa_media']:.0f}<br><b>Registros:</b> {int(r['regs'])}"
        ))
    # add marker points (capitals) without showing coords in hover
    fig2.add_trace(go.Scattermapbox(
        lat=agg_cap["lat"], lon=agg_cap["lon"], mode="markers+text",
        marker=go.scattermapbox.Marker(size=10, color=ACCENT),
        text=agg_cap["DEST"], textposition="top right", textfont=dict(size=11),
        hovertemplate="<b>%{text}</b><br>Tarifa média: R$ %{customdata[0]:.0f}<br>Temp média: %{customdata[1]:.1f} °C<extra></extra>",
        customdata=np.stack([agg_cap["tarifa_media"].round(0), agg_cap["temp_media"].round(1)], axis=1)
    ))
    fig2.update_layout(mapbox_style="carto-positron", mapbox_center={"lat":-14.2,"lon":-51.9}, mapbox_zoom=3.2, margin={"r":0,"t":0,"l":0,"b":0}, height=520)
    st.plotly_chart(fig2, use_container_width=True)
else:
    st.info("Sem rotas para os filtros atuais.")

# -----------------------
# FIG 3 — Time series (monthly aggregated)
# -----------------------
st.subheader("3) Série temporal — Tarifa média mensal")
ts = dff.groupby("DATA").agg(tarifa=("TARIFA","mean")).reset_index().sort_values("DATA")
fig3 = px.line(ts, x="DATA", y="tarifa", markers=True, title="Tarifa média (mensal)")
fig3.update_layout(yaxis_title="Tarifa média (R$)", margin={"r":0,"t":30,"l":0,"b":0})
st.plotly_chart(fig3, use_container_width=True)

# -----------------------
# FIG 4 — Bar: Tarifa por estação (labels R$ no topo)
# -----------------------
st.subheader("4) Tarifa média por estação (R$)")
est = dff.groupby("ESTACAO").agg(tarifa=("TARIFA","mean")).reindex(["Verão","Outono","Inverno","Primavera"]).reset_index()
est["tarifa"] = est["tarifa"].round(0)
fig4 = px.bar(est, x="ESTACAO", y="tarifa", text="tarifa", color="ESTACAO",
              color_discrete_sequence=[PRIMARY, ACCENT, "#FDBA74", "#6366F1"])
fig4.update_traces(texttemplate="R$ %{text:.0f}", textposition="outside")
fig4.update_layout(yaxis_title="Tarifa média (R$)")
st.plotly_chart(fig4, use_container_width=True)

# -----------------------
# FIG 5 — Bar: Tarifa por região
# -----------------------
st.subheader("5) Tarifa média por região")
REGIOES = {
    "Norte": ["Belém","Macapá","Manaus","Boa Vista","Rio Branco","Porto Velho","Palmas"],
    "Nordeste": ["São Luís","Teresina","Fortaleza","Natal","João Pessoa","Recife","Maceió","Aracaju","Salvador"],
    "Centro-Oeste": ["Brasília","Goiânia","Campo Grande","Cuiabá"],
    "Sudeste": ["São Paulo","Rio de Janeiro","Belo Horizonte","Vitória"],
    "Sul": ["Curitiba","Florianópolis","Porto Alegre"]
}
def get_region(city):
    for reg, cities in REGIOES.items():
        if city in cities:
            return reg
    return "Outro"
dff["REGIAO"] = dff["DEST"].apply(get_region)
reg = dff.groupby("REGIAO").agg(tarifa=("TARIFA","mean")).reset_index()
reg["tarifa"] = reg["tarifa"].round(0)
fig5 = px.bar(reg, x="REGIAO", y="tarifa", text="tarifa", color="REGIAO",
              color_discrete_sequence=[PRIMARY, ACCENT, "#16A34A", "#9333EA", "#E11D48"])
fig5.update_traces(texttemplate="R$ %{text:.0f}", textposition="outside")
fig5.update_layout(yaxis_title="Tarifa média (R$)")
st.plotly_chart(fig5, use_container_width=True)

# -----------------------
# FIG 6 — Companhias: box + top mean
# -----------------------
st.subheader("6) Companhias — distribuição de tarifas e médias")
cmp = dff[dff["COMPANHIA"].notna()]
if cmp.shape[0] > 20:
    fig6a = px.box(cmp, x="COMPANHIA", y="TARIFA", points="outliers", title="Distribuição de tarifas por companhia")
    fig6a.update_layout(xaxis_tickangle=-45, yaxis_title="Tarifa (R$)")
    st.plotly_chart(fig6a, use_container_width=True)
    mean_cmp = cmp.groupby("COMPANHIA").agg(tarifa=("TARIFA","mean")).reset_index().sort_values("tarifa", ascending=False)
    fig6b = px.bar(mean_cmp.head(10), x="COMPANHIA", y="tarifa", text="tarifa")
    fig6b.update_traces(texttemplate="R$ %{text:.0f}", textposition="outside")
    st.plotly_chart(fig6b, use_container_width=True)
else:
    st.info("Dados de companhia insuficientes para análise aprofundada.")

# -----------------------
# FIG 7 — Heatmap month x capital
# -----------------------
st.subheader("7) Heatmap — Tarifa média (mês × capital)")
heat = dff.groupby([dff["DATA"].dt.month.rename("MES"), "DEST"]).agg(tarifa=("TARIFA","mean")).reset_index()
pivot = heat.pivot(index="DEST", columns="MES", values="tarifa").fillna(0)
pivot = pivot.reindex(sorted(pivot.columns), axis=1)
fig7 = px.imshow(pivot, labels=dict(x="Mês", y="Capital", color="Tarifa (R$)"),
                 x=[str(c) for c in pivot.columns], y=pivot.index)
st.plotly_chart(fig7, use_container_width=True)

# -----------------------
# Insights automáticos (dinâmicos)
# -----------------------
st.markdown("---")
st.header("Insights automáticos — Bora Dicas")
top_city = agg_cap.loc[agg_cap["tarifa_media"].idxmax()]
cheap_city = agg_cap.loc[agg_cap["tarifa_media"].idxmin()]
season_avg = dff.groupby("ESTACAO").agg(tarifa=("TARIFA","mean")).reset_index()
cheapest_season = season_avg.loc[season_avg["tarifa"].idxmin()]["ESTACAO"] if season_avg.shape[0]>0 else None

st.markdown(f"• Capital mais cara (média): **{top_city['DEST']}** — R$ {top_city['tarifa_media']:.0f}")
st.markdown(f"• Capital mais barata (média): **{cheap_city['DEST']}** — R$ {cheap_city['tarifa_media']:.0f}")
if cheapest_season:
    st.markdown(f"• Estação com menor tarifa média: **{cheapest_season}** — considere viajar nessa época.")

# -----------------------
# Forecast (per rota)
# -----------------------
st.markdown("---")
st.header("Previsão — Tarifas para 2026 (Prophet)")
rota_choice = st.selectbox("Escolha uma rota para previsão:", sorted(dff["ROTA"].unique()))
df_model = dff[dff["ROTA"]==rota_choice].groupby("DATA").agg(tarifa=("TARIFA","mean"), temp=("TEMP_MEDIA","mean")).reset_index()

if df_model.shape[0] >= 12:
    dfp = df_model.rename(columns={"DATA":"ds","tarifa":"y","temp":"temp_reg"}).dropna(subset=["ds","y"])
    m = Prophet(yearly_seasonality=True)
    # optionally add regressor if temp present
    if "temp_reg" in dfp.columns and dfp["temp_reg"].notna().sum() > 0:
        m.add_regressor("temp_reg")
    m.fit(dfp)
    future = m.make_future_dataframe(periods=12, freq="MS")
    # fill future temp with monthly averages
    if "temp_reg" in dfp.columns:
        monthly_temp = dfp.groupby(dfp["ds"].dt.month)["temp_reg"].mean().to_dict()
        future["month"] = future["ds"].dt.month
        future["temp_reg"] = future["month"].map(monthly_temp).fillna(dfp["temp_reg"].mean())
    forecast = m.predict(future)
    st.plotly_chart(plot_plotly(m, forecast), use_container_width=True)
    f2026 = forecast[forecast["ds"].dt.year==2026][["ds","yhat","yhat_lower","yhat_upper"]].rename(columns={"ds":"DATA","yhat":"TARIFA_PRED"})
    f2026["TARIFA_PRED"] = f2026["TARIFA_PRED"].round(0)
    st.subheader("Previsão 2026 — mês a mês")
    st.table(f2026.set_index("DATA"))
    # save
    safe_name = rota_choice.replace(" ","_").replace("/","_")
    fpath = os.path.join(OUTPUT_DIR, f"forecast_2026_{safe_name}.csv")
    f2026.to_csv(fpath, index=False)
    st.success(f"Previsão salva: {fpath}")
else:
    st.warning("Dados insuficientes (menos de 12 meses) para previsão desta rota.")

# -----------------------
# Export button: CSV + try images
# -----------------------
st.markdown("---")
if st.button("Exportar resumo CSV + imagens (se possível)"):
    try:
        summary = agg_cap.copy()
        summary["tarifa_media"] = summary["tarifa_media"].round(0)
        csv_out = os.path.join(OUTPUT_DIR, "boraali_summary.csv")
        summary.to_csv(csv_out, index=False)
        # try save images (requires kaleido)
        import plotly.io as pio
        pio.write_image(fig1, os.path.join(OUTPUT_DIR, "map_capitais.png"), width=1600, height=900)
        pio.write_image(fig2, os.path.join(OUTPUT_DIR, "map_rotas.png"), width=1600, height=900)
        pio.write_image(fig3, os.path.join(OUTPUT_DIR, "serie_tarifa.png"), width=1600, height=900)
        pio.write_image(fig4, os.path.join(OUTPUT_DIR, "estacoes.png"), width=1600, height=900)
        pio.write_image(fig5, os.path.join(OUTPUT_DIR, "regioes.png"), width=1600, height=900)
        pio.write_image(fig7, os.path.join(OUTPUT_DIR, "heatmap.png"), width=1600, height=900)
        st.success(f"Export feito. Arquivos em {OUTPUT_DIR}")
    except Exception as e:
        st.warning(f"CSV salvo em {csv_out}. PNGs podem não ter sido gerados (kaleido). Erro: {e}")

st.caption("Design: Bora Alí — jovem, moderno e profissional. Feito para SR2.")
