# app.py — Bora Alí versão “segura e funcional”
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

st.set_page_config(page_title="Bora Alí — SR2 (funcional)", layout="wide")

@st.cache_data
def load_data(path="INMET_ANAC_EXTREMAMENTE_REDUZIDO.csv"):
    df = pd.read_csv(path)
    return df

# Carrega os dados
try:
    df = load_data()
except Exception as e:
    st.error(f"Erro ao carregar o CSV: {e}")
    st.stop()

# Verifica se colunas mínimas estão presentes
required = {"COMPANHIA", "ANO", "MES", "ORIGEM", "DESTINO", "TARIFA", "TEMP_MEDIA"}
missing = required - set(df.columns)
if missing:
    st.error(f"Faltando colunas obrigatórias no CSV: {missing}")
    st.stop()

# Tipagens seguras
df = df.copy()
df["MES"] = pd.to_numeric(df["MES"], errors="coerce").astype('Int64')
df["ANO"] = pd.to_numeric(df["ANO"], errors="coerce").astype('Int64')
df["TARIFA"] = pd.to_numeric(df["TARIFA"], errors="coerce")
df["TEMP_MEDIA"] = pd.to_numeric(df["TEMP_MEDIA"], errors="coerce")

# Filtros simples
st.sidebar.header("Filtros")
years = sorted(df["ANO"].dropna().unique())
sel_years = st.sidebar.multiselect("Ano(s)", options=years, default=years)

meses = {
    1:"Janeiro", 2:"Fevereiro", 3:"Março", 4:"Abril",
    5:"Maio", 6:"Junho", 7:"Julho", 8:"Agosto",
    9:"Setembro", 10:"Outubro", 11:"Novembro", 12:"Dezembro"
}
sel_meses = st.sidebar.multiselect("Mês(es)", options=list(meses.values()), default=list(meses.values()))
# converte nomes para números
sel_meses_num = [k for k,v in meses.items() if v in sel_meses]

fil_orig = st.sidebar.text_input("Filtrar ORIGEM (parte do nome)", value="")
fil_dest = st.sidebar.text_input("Filtrar DESTINO (parte do nome)", value="")

df_f = df[
    (df["ANO"].isin(sel_years)) &
    (df["MES"].isin(sel_meses_num))
]

if fil_orig:
    df_f = df_f[df_f["ORIGEM"].str.contains(fil_orig, case=False, na=False)]
if fil_dest:
    df_f = df_f[df_f["DESTINO"].str.contains(fil_dest, case=False, na=False)]

st.title("📊 Bora Alí — Visão Inicial")

st.markdown("### KPI principais")
col1, col2, col3 = st.columns(3)
col1.metric("Registros", f"{len(df_f):,}")
col2.metric("Tarifa média (R$)", f"{df_f['TARIFA'].mean():.2f}")
col3.metric("Temperatura média (°C)", f"{df_f['TEMP_MEDIA'].mean():.1f}")

st.markdown("---")
st.subheader("Distribuição de Tarifas por Origem → Destino")

# Criar rota para agrupar
df_f["ROTA"] = df_f["ORIGEM"] + " → " + df_f["DESTINO"]

# Tabela top 10 rotas por número de voos/registros
route_stats = df_f.groupby("ROTA").agg(
    cnt=("TARIFA","count"),
    tarifa_med=("TARIFA","mean")
).reset_index().sort_values("cnt", ascending=False)

st.dataframe(route_stats.head(10), use_container_width=True)

st.markdown("---")
st.subheader("Histórico de Tarifas — média mensal (todas rotas filtradas)")

# Agrupar por ano-mês
df_time = df_f.dropna(subset=["ANO","MES"])
df_time["YM"] = df_time["ANO"].astype(str) + "-" + df_time["MES"].astype(str).str.zfill(2)
ts = df_time.groupby("YM")["TARIFA"].mean().reset_index()

fig = px.line(ts, x="YM", y="TARIFA", title="Tarifa média por mês", markers=True)
fig.update_layout(xaxis_title="Ano-Mês", yaxis_title="Tarifa (R$)", xaxis_tickangle=-45)
st.plotly_chart(fig, use_container_width=True)

st.markdown("---")
st.write("✅ Se este app abriu até aqui sem mostrar tela vermelha, significa que o código está executando sem erro.")
