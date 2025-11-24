# app.py — Bora Alí (Laranja Sunset) — Versão FINAL com rankings + 7 gráficos + previsão 2026
import os
import unicodedata
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from prophet import Prophet
from prophet.plot import plot_plotly

# ---------------------------
# Config / Paleta — Laranja Sunset
# ---------------------------
st.set_page_config(page_title="Bora Alí — Capitais (Laranja Sunset)", layout="wide", page_icon="🧳")

CSV_FILE = "INMET_ANAC_ROTAS_APENAS_CAPITAIS.csv"  # must be in repo root
OUTPUT_DIR = "outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

ORANGE = "#F76715"
PURPLE = "#7E3FF2"
SOFT = "#FFBF69"
BG = "#FCFBFA"
TEXT = "#0F172A"

PDF_REF = "/mnt/data/Bora Alí — Dashboard (Capitais) · Streamlit.pdf"

st.markdown(f"""
<style>
body {{background:{BG};}}
h1,h2,h3 {{color:{PURPLE}; font-weight:800;}}
.stButton>button {{background:{ORANGE}; color:white; border-radius:10px; font-weight:700;}}
</style>
""", unsafe_allow_html=True)

st.title("✈️ Bora Alí — Capitais (Laranja Sunset)")
st.caption("Design: Laranja Sunset — urbano, moderno • 7 visualizações interativas • previsão 2026")
st.sidebar.markdown(f"Fonte (PDF design): `{PDF_REF}`")

# ---------------------------
# Canonical capitals & helpers
# ---------------------------
CANONICAL = [
    "Rio Branco","Maceió","Macapá","Manaus","Salvador","Fortaleza","Brasília","Vitória","Goiânia",
    "São Luís","Cuiabá","Campo Grande","Belo Horizonte","Belém","João Pessoa","Curitiba","Recife",
    "Teresina","Rio de Janeiro","Natal","Porto Alegre","Porto Velho","Boa Vista","Florianópolis",
    "Aracaju","São Paulo","Palmas"
]

def normalize_str(s):
    if pd.isna(s):
        return s
    s = str(s)
    s = ''.join(ch for ch in unicodedata.normalize('NFKD', s) if not unicodedata.combining(ch))
    s = s.replace("_", " ").replace("-", " ")
    s = " ".join(s.split()).strip().lower()
    return s

NORM_TO_CANON = { normalize_str(c): c for c in CANONICAL }

def map_to_canonical(city):
    if pd.isna(city):
        return city
    n = normalize_str(city)
    if n in NORM_TO_CANON:
        return NORM_TO_CANON[n]
    # token fuzzy match
    for key,val in NORM_TO_CANON.items():
        tokens_key = key.split()
        tokens_n = n.split()
        if any(tok in tokens_key for tok in tokens_n):
            return val
    return city.strip().title()

def safe_parse_route(r):
    if pd.isna(r):
        return (None, None)
    s = str(r)
    if "→" in s:
        parts = [p.strip() for p in s.split("→") if p.strip()]
        if len(parts) >= 2:
            return parts[0], parts[-1]
    if "-" in s:
        parts = [p.strip() for p in s.split("-") if p.strip()]
        if len(parts) >= 2:
            return parts[0], parts[-1]
    return (None, None)

# ---------------------------
# Load data (robust)
# ---------------------------
@st.cache_data(ttl=600)
def load_data(path):
    try:
        df = pd.read_csv(path, low_memory=False)
    except FileNotFoundError:
        st.error(f"Arquivo '{path}' não encontrado na raiz do repositório. Faça upload e redeploy.")
        st.stop()
    # normalize columns
    df.columns = [c.strip().upper() for c in df.columns]
    # ensure columns exist
    for c in ["ROTA","COMPANHIA","ORIGEM","DESTINO","TARIFA","TEMP_MEDIA","ANO","MES","TEMP_MIN","TEMP_MAX"]:
        if c not in df.columns:
            df[c] = pd.NA
    # if TEMP_MEDIA missing, attempt mean of min/max
    if df["TEMP_MEDIA"].isna().all() and "TEMP_MIN" in df.columns and "TEMP_MAX" in df.columns:
        df["TEMP_MEDIA"] = (pd.to_numeric(df["TEMP_MIN"], errors="coerce") + pd.to_numeric(df["TEMP_MAX"], errors="coerce")) / 2
    # parse origin/dest
    df["ORIG"] = df["ORIGEM"].where(df["ORIGEM"].notna(), None)
    df["DEST"] = df["DESTINO"].where(df["DESTINO"].notna(), None)
    parsed = df.apply(lambda r: pd.Series(safe_parse_route(r["ROTA"]), index=["_p_orig","_p_dest"]), axis=1)
    df["ORIG"] = df["ORIG"].fillna(parsed["_p_orig"])
    df["DEST"] = df["DEST"].fillna(parsed["_p_dest"])
    # map to canonical
    df["ORIG"] = df["ORIG"].astype(str).apply(map_to_canonical)
    df["DEST"] = df["DEST"].astype(str).apply(map_to_canonical)
    # numeric
    df["TARIFA"] = pd.to_numeric(df["TARIFA"], errors="coerce")
    df["TEMP_MEDIA"] = pd.to_numeric(df["TEMP_MEDIA"], errors="coerce")
    df["ANO"] = pd.to_numeric(df["ANO"], errors="coerce").fillna(0).astype(int)
    df["MES"] = pd.to_numeric(df["MES"], errors="coerce").fillna(0).astype(int)
    # data
    df["DATA"] = pd.to_datetime(df["ANO"].astype(str) + "-" + df["MES"].astype(str).str.zfill(2) + "-01", errors="coerce")
    # standard rota
    df["ROTA"] = df["ORIG"].fillna("") + " → " + df["DEST"].fillna("")
    return df

df = load_data(CSV_FILE)

# ---------------------------
# Sidebar filters (including estação)
# ---------------------------
st.sidebar.header("Filtros — Bora Alí (Laranja Sunset)")
anos = sorted(df["ANO"].dropna().unique())
sel_anos = st.sidebar.multiselect("Ano", anos, default=anos)
sel_meses = st.sidebar.multiselect("Mês", sorted(df["MES"].dropna().unique()), default=sorted(df["MES"].dropna().unique()))
comp = sorted(df["COMPANHIA"].dropna().unique())
sel_comp = st.sidebar.multiselect("Companhia", comp, default=comp if comp else [])

def season_of(m):
    if m in [12,1,2]: return "Verão"
    if m in [3,4,5]: return "Outono"
    if m in [6,7,8]: return "Inverno"
    return "Primavera"

df["ESTACAO"] = df["MES"].apply(lambda x: season_of(int(x)) if not pd.isna(x) else np.nan)
sel_est = st.sidebar.multiselect("Estação", ["Verão","Outono","Inverno","Primavera"], default=["Verão","Outono","Inverno","Primavera"])

present_caps = sorted(set(df["ORIG"].dropna().unique()) | set(df["DEST"].dropna().unique()))
sel_caps = st.sidebar.multiselect("Capitais (orig/dest)", present_caps, default=present_caps)

# apply filters
dff = df[
    (df["ANO"].isin(sel_anos)) &
    (df["MES"].isin(sel_meses)) &
    (df["COMPANHIA"].isin(sel_comp)) &
    (df["ESTACAO"].isin(sel_est))
].copy()

# keep only rows where both origin & dest selected
dff = dff[dff["ORIG"].isin(sel_caps) & dff["DEST"].isin(sel_caps)].copy()

if dff.empty:
    st.warning("Nenhum registro após filtros. Ajuste filtros.")
    st.stop()

# ---------------------------
# Coordinates for canonical capitals
# ---------------------------
COORDS = {
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
dff["LAT"] = dff["DEST"].map(lambda x: COORDS.get(x, (np.nan, np.nan))[0])
dff["LON"] = dff["DEST"].map(lambda x: COORDS.get(x, (np.nan, np.nan))[1])

# ---------------------------
# KPIs + Top Rankings
# ---------------------------
c1, c2, c3, c4 = st.columns(4)
c1.metric("Registros (filtrados)", f"{len(dff):,}")
c2.metric("Tarifa média (R$)", f"{dff['TARIFA'].mean():.0f}")
c3.metric("Temp média (°C)", f"{dff['TEMP_MEDIA'].mean():.1f}")
c4.metric("Rotas únicas", f"{dff['ROTA'].nunique():,}")

# Top Destinos (ranking)
st.markdown("---")
st.subheader("Ranking — Top Destinos (Tarifa média)")
top_dest = dff.groupby("DEST").agg(tarifa_media=("TARIFA","mean"), regs=("TARIFA","count")).reset_index().sort_values("tarifa_media", ascending=False)
st.table(top_dest.head(10).assign(tarifa_media=lambda x: x["tarifa_media"].round(0).astype(int)).rename(columns={"DEST":"Capital","tarifa_media":"Tarifa média (R$)","regs":"Registros"}).set_index("Capital"))

# Top Rotas (ranking by tarifa média)
st.subheader("Ranking — Top Rotas (Tarifa média)")
top_rot = dff.groupby("ROTA").agg(tarifa_media=("TARIFA","mean"), regs=("TARIFA","count")).reset_index().sort_values("tarifa_media", ascending=False)
st.table(top_rot.head(10).assign(tarifa_media=lambda x: x["tarifa_media"].round(0).astype(int)).rename(columns={"ROTA":"Rota","tarifa_media":"Tarifa média (R$)","regs":"Registros"}).set_index("Rota"))

# ---------------------------
# FIG 1 — Map points (tarifa size, temp color) — hover hides coords
# ---------------------------
st.markdown("---")
st.subheader("1) Mapa — Capitais (tamanho = tarifa média · cor = temp média)")
agg_cap = dff.groupby("DEST").agg(tarifa_media=("TARIFA","mean"), temp_media=("TEMP_MEDIA","mean"), regs=("TARIFA","count")).reset_index()
agg_cap["lat"] = agg_cap["DEST"].map(lambda x: COORDS.get(x,(np.nan,np.nan))[0])
agg_cap["lon"] = agg_cap["DEST"].map(lambda x: COORDS.get(x,(np.nan,np.nan))[1])

fig1 = px.scatter_mapbox(
    agg_cap.dropna(subset=["lat","lon"]),
    lat="lat", lon="lon",
    size="tarifa_media", color="temp_media",
    hover_name="DEST",
    hover_data={"tarifa_media":":.0f","temp_media":":.1f","regs":True,"lat":False,"lon":False},
    size_max=45, zoom=3.2,
    color_continuous_scale=[SOFT, ORANGE, PURPLE], height=480
)
fig1.update_layout(mapbox_style="carto-positron", margin={"r":0,"t":0,"l":0,"b":0})
st.plotly_chart(fig1, use_container_width=True)

# ---------------------------
# FIG 2 — Routes map (safe widths; no NaN)
# ---------------------------
st.markdown("---")
st.subheader("2) Mapa — Rotas premium (espessura ≈ tarifa média)")

routes = dff.groupby("ROTA").agg(tarifa_media=("TARIFA","mean"), regs=("TARIFA","count")).reset_index()
routes[["ORIG","DEST"]] = routes["ROTA"].apply(lambda r: pd.Series(safe_parse_route(r)))

# attach coords via canonical mapping fallback
routes["olat"] = routes["ORIG"].map(lambda x: COORDS.get(map_to_canonical(x) if x else x, (np.nan,np.nan))[0])
routes["olon"] = routes["ORIG"].map(lambda x: COORDS.get(map_to_canonical(x) if x else x, (np.nan,np.nan))[1])
routes["dlat"] = routes["DEST"].map(lambda x: COORDS.get(map_to_canonical(x) if x else x, (np.nan,np.nan))[0])
routes["dlon"] = routes["DEST"].map(lambda x: COORDS.get(map_to_canonical(x) if x else x, (np.nan,np.nan))[1])

# drop invalid rows
routes = routes.dropna(subset=["olat","olon","dlat","dlon","tarifa_media"])
routes = routes[routes["tarifa_media"] > 0].copy()

if not routes.empty:
    q = routes["tarifa_media"].quantile([0,0.25,0.5,0.75,1.0]).tolist()
    def safe_width(x):
        try:
            x = float(x)
        except:
            return 1.0
        if x <= q[1]: return 1.2
        if x <= q[2]: return 2.6
        if x <= q[3]: return 4.2
        return 6.8
    routes["width"] = routes["tarifa_media"].apply(safe_width)

    fig2 = go.Figure()
    for _, r in routes.iterrows():
        fig2.add_trace(go.Scattermapbox(
            lat=[r["olat"], r["dlat"]],
            lon=[r["olon"], r["dlon"]],
            mode="lines",
            line=dict(width=float(r["width"]), color=ORANGE, opacity=0.85),
            hoverinfo="text",
            text=f"<b>Rota</b>: {r['ROTA']}<br><b>Tarifa média</b>: R$ {r['tarifa_media']:.0f}<br><b>Registros</b>: {int(r['regs'])}"
        ))
    # capitals markers
    fig2.add_trace(go.Scattermapbox(
        lat=agg_cap["lat"], lon=agg_cap["lon"], mode="markers+text",
        marker=dict(size=9, color=PURPLE),
        text=agg_cap["DEST"], textposition="top right", textfont=dict(size=11),
        hovertemplate="<b>%{text}</b><extra></extra>"
    ))
    fig2.update_layout(mapbox_style="carto-positron",
                       mapbox_center={"lat":-14.2,"lon":-51.9},
                       mapbox_zoom=3.2,
                       margin={"r":0,"t":0,"l":0,"b":0},
                       height=520)
    st.plotly_chart(fig2, use_container_width=True)
else:
    st.info("Sem rotas disponíveis para os filtros atuais.")

# ---------------------------
# FIG 3 — Time series (monthly aggregated)
# ---------------------------
st.markdown("---")
st.subheader("3) Série temporal — Tarifa média mensal")
ts = dff.groupby("DATA").agg(tarifa=("TARIFA","mean")).reset_index().sort_values("DATA")
fig3 = px.line(ts, x="DATA", y="tarifa", markers=True, title="Tarifa média mensal")
fig3.update_layout(yaxis_title="Tarifa média (R$)")
st.plotly_chart(fig3, use_container_width=True, height=380)

# ---------------------------
# FIG 4 — Tarifa por estação (bar + labels)
# ---------------------------
st.markdown("---")
st.subheader("4) Tarifa média por estação")
est = dff.groupby("ESTACAO").agg(tarifa=("TARIFA","mean")).reindex(["Verão","Outono","Inverno","Primavera"]).reset_index()
est["tarifa"] = est["tarifa"].round(0)
fig4 = px.bar(est, x="ESTACAO", y="tarifa", color="ESTACAO", text="tarifa",
              color_discrete_sequence=[ORANGE, SOFT, PURPLE, "#7BC4C4"])
fig4.update_traces(texttemplate="R$ %{text:.0f}", textposition="outside")
fig4.update_layout(yaxis_title="Tarifa média (R$)")
st.plotly_chart(fig4, use_container_width=True, height=360)

# ---------------------------
# FIG 5 — Tarifa por região
# ---------------------------
st.markdown("---")
st.subheader("5) Tarifa média por região")
REGIONS = {
    "Norte": ["Belém","Macapá","Manaus","Boa Vista","Rio Branco","Porto Velho","Palmas"],
    "Nordeste": ["São Luís","Teresina","Fortaleza","Natal","João Pessoa","Recife","Maceió","Aracaju","Salvador"],
    "Centro-Oeste": ["Brasília","Goiânia","Campo Grande","Cuiabá"],
    "Sudeste": ["São Paulo","Rio de Janeiro","Belo Horizonte","Vitória"],
    "Sul": ["Curitiba","Florianópolis","Porto Alegre"]
}
def region_of(city):
    for k,v in REGIONS.items():
        if city in v: return k
    return "Outro"
dff["REGIAO"] = dff["DEST"].apply(region_of)
reg = dff.groupby("REGIAO").agg(tarifa=("TARIFA","mean")).reset_index()
reg["tarifa"] = reg["tarifa"].round(0)
fig5 = px.bar(reg, x="REGIAO", y="tarifa", text="tarifa", color="REGIAO",
              color_discrete_sequence=[ORANGE, PURPLE, "#16A34A", "#9333EA", "#E11D48"])
fig5.update_traces(texttemplate="R$ %{text:.0f}", textposition="outside")
fig5.update_layout(yaxis_title="Tarifa média (R$)")
st.plotly_chart(fig5, use_container_width=True, height=360)

# ---------------------------
# FIG 6 — Companhias: distribution + top means
# ---------------------------
st.markdown("---")
st.subheader("6) Companhias — distribuição de tarifas e top médias")
cmp = dff[dff["COMPANHIA"].notna()].copy()
if cmp.shape[0] > 20:
    fig6a = px.box(cmp, x="COMPANHIA", y="TARIFA", points="outliers", title="Distribuição por companhia")
    fig6a.update_layout(xaxis_tickangle=-45, yaxis_title="Tarifa (R$)")
    st.plotly_chart(fig6a, use_container_width=True, height=360)
    mean_cmp = cmp.groupby("COMPANHIA").agg(tarifa=("TARIFA","mean")).reset_index().sort_values("tarifa", ascending=False)
    fig6b = px.bar(mean_cmp.head(8), x="COMPANHIA", y="tarifa", text="tarifa", title="Top 8 médias por companhia")
    fig6b.update_traces(texttemplate="R$ %{text:.0f}", textposition="outside")
    st.plotly_chart(fig6b, use_container_width=True, height=360)
else:
    st.info("Dados de companhia insuficientes para comparação robusta.")

# ---------------------------
# FIG 7 — Heatmap month x capital
# ---------------------------
st.markdown("---")
st.subheader("7) Heatmap — Tarifa média (mês × capital)")
heat = dff.groupby([dff["DATA"].dt.month.rename("MES"), "DEST"]).agg(tarifa=("TARIFA","mean")).reset_index()
pivot = heat.pivot(index="DEST", columns="MES", values="tarifa").fillna(0)
pivot = pivot.reindex(sorted(pivot.columns), axis=1)
fig7 = px.imshow(pivot, labels=dict(x="Mês", y="Capital", color="Tarifa (R$)"),
                 x=[str(m) for m in pivot.columns], y=pivot.index, aspect="auto")
st.plotly_chart(fig7, use_container_width=True, height=520)

# ---------------------------
# Insights automáticos
# ---------------------------
st.markdown("---")
st.header("Insights automáticos — Bora Dicas")
agg_cap2 = dff.groupby("DEST").agg(tarifa_media=("TARIFA","mean")).reset_index()
most_exp = agg_cap2.loc[agg_cap2["tarifa_media"].idxmax()]
least_exp = agg_cap2.loc[agg_cap2["tarifa_media"].idxmin()]
season_avg = dff.groupby("ESTACAO").agg(tarifa=("TARIFA","mean")).reset_index()
cheapest_season = season_avg.loc[season_avg["tarifa"].idxmin()]["ESTACAO"] if not season_avg.empty else None

st.markdown(f"• Capital com maior tarifa média: **{most_exp['DEST']}** — R$ {most_exp['tarifa_media']:.0f}")
st.markdown(f"• Capital com menor tarifa média: **{least_exp['DEST']}** — R$ {least_exp['tarifa_media']:.0f}")
if cheapest_season:
    st.markdown(f"• Estação mais econômica (média): **{cheapest_season}** — considerar viajar nessa época.")

# ---------------------------
# Forecast (Prophet) per rota with temp regressor
# ---------------------------
st.markdown("---")
st.header("Previsão — Tarifas 2026 (por rota)")
rota_choice = st.selectbox("Escolha uma rota para previsão:", sorted(dff["ROTA"].unique()))
df_model = dff[dff["ROTA"]==rota_choice].groupby("DATA").agg(tarifa=("TARIFA","mean"), temp=("TEMP_MEDIA","mean")).reset_index()

if df_model.shape[0] >= 12:
    dfp = df_model.rename(columns={"DATA":"ds","tarifa":"y","temp":"temp_reg"}).dropna(subset=["ds","y"])
    m = Prophet(yearly_seasonality=True)
    if dfp["temp_reg"].notna().sum() > 0:
        m.add_regressor("temp_reg")
    m.fit(dfp)
    future = m.make_future_dataframe(periods=12, freq="MS")
    if "temp_reg" in dfp.columns:
        monthly_temp = dfp.groupby(dfp["ds"].dt.month)["temp_reg"].mean().to_dict()
        future["month"] = future["ds"].dt.month
        future["temp_reg"] = future["month"].map(monthly_temp).fillna(dfp["temp_reg"].mean())
    forecast = m.predict(future)
    st.plotly_chart(plot_plotly(m, forecast), use_container_width=True, height=500)
    f2026 = forecast[forecast["ds"].dt.year==2026][["ds","yhat","yhat_lower","yhat_upper"]].rename(columns={"ds":"DATA","yhat":"TARIFA_PRED"})
    f2026["TARIFA_PRED"] = f2026["TARIFA_PRED"].round(0)
    st.subheader("Previsão 2026 — mês a mês")
    st.table(f2026.set_index("DATA"))
    # save
    outpred = os.path.join(OUTPUT_DIR, f"forecast_2026_{rota_choice.replace(' ','_')}.csv")
    f2026.to_csv(outpred, index=False)
    st.success(f"Previsão salva: {outpred}")
else:
    st.warning("Dados insuficientes (menos de 12 meses) para previsão desta rota.")

# ---------------------------
# Export button (CSV + PNGs if kaleido available)
# ---------------------------
st.markdown("---")
if st.button("Exportar 7 gráficos (PNG) + resumo CSV"):
    try:
        import plotly.io as pio
        summary = agg_cap.copy()
        summary["tarifa_media"] = summary["tarifa_media"].round(0)
        csvpath = os.path.join(OUTPUT_DIR, "boraali_summary.csv")
        summary.to_csv(csvpath, index=False)
        # save images — requires kaleido
        pio.write_image(fig1, os.path.join(OUTPUT_DIR,"01_map_capitais.png"), width=1600, height=900)
        pio.write_image(fig2, os.path.join(OUTPUT_DIR,"02_map_rotas.png"), width=1600, height=900)
        pio.write_image(fig3, os.path.join(OUTPUT_DIR,"03_serie.png"), width=1600, height=900)
        pio.write_image(fig4, os.path.join(OUTPUT_DIR,"04_estacoes.png"), width=1600, height=900)
        pio.write_image(fig5, os.path.join(OUTPUT_DIR,"05_regioes.png"), width=1600, height=900)
        # choose fig6b if exists else fig6a for companies
        if 'fig6b' in globals():
            pio.write_image(fig6b, os.path.join(OUTPUT_DIR,"06_companhias.png"), width=1600, height=900)
        else:
            pio.write_image(fig6a, os.path.join(OUTPUT_DIR,"06_companhias.png"), width=1600, height=900)
        pio.write_image(fig7, os.path.join(OUTPUT_DIR,"07_heatmap.png"), width=1600, height=900)
        st.success(f"Export concluído em: {OUTPUT_DIR}")
    except Exception as e:
        st.warning(f"Export parcial: CSV salvo. PNGs podem exigir 'kaleido'. Erro: {e}")

st.caption("Paleta: Laranja Sunset — urbano e moderno. Feito para SR2 — Bora Alí.")
