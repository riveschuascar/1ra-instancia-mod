import streamlit as st
import plotly.express as px
from engine_final import DynamicSystem, PlayerParameters

st.set_page_config(page_title="Sistema Integrado de Fútbol", layout="wide")
st.title("⚽ Simulador Dinámico de Futbolistas (Día 3)")

# --- Sidebar: Calibración por Posición (Estudiante C) ---
st.sidebar.header("Calibración del Jugador")
pos = st.sidebar.selectbox("Posición", ["Delantero", "Mediocampista", "Defensa"])
perfil = st.sidebar.selectbox("Escenario (Estudiante A)", ["Joven Promesa", "Late Bloomer", "Desarrollo Constante"])

# Configuración de pesos según rúbrica
pesos = {"Delantero": [0.45, 0.40, 0.15], "Mediocampista": [0.30, 0.50, 0.20], "Defensa": [0.40, 0.25, 0.35]}
w = pesos[pos]

# Ajuste de parámetros según perfil
if perfil == "Joven Promesa":
    p = PlayerParameters(0.20, 0.15, 0.10, 0.009, 0.007, 0.005, 0.05, 0.03, w[0], w[1], w[2], 25.0)
elif perfil == "Late Bloomer":
    p = PlayerParameters(0.12, 0.15, 0.10, 0.006, 0.005, 0.004, 0.04, 0.02, w[0], w[1], w[2], 30.0)
else:
    p = PlayerParameters(0.15, 0.15, 0.15, 0.008, 0.006, 0.005, 0.04, 0.02, w[0], w[1], w[2], 27.0)

# --- Sidebar: Simulación (Estudiante A) ---
st.sidebar.header("Análisis de Sensibilidad")
intensidad = st.sidebar.slider("Intensidad Entrenamiento", 0.0, 1.0, 0.8)
lesion_check = st.sidebar.checkbox("Simular Lesión Grave (Edad 22)")

# Ejecutar Modelo
sim = DynamicSystem(p)
df = sim.run_simulation(15, {'F':0.4, 'T':0.4, 'M':0.3, 'A':18.0}, intensidad, 22.0 if lesion_check else None)

# --- Visualización Final (Estudiante D) ---
col1, col2 = st.columns(2)
with col1:
    st.subheader("Trayectoria del Overall Rating (R)")
    st.plotly_chart(px.line(df, x='Edad', y='Rating', color_discrete_sequence=['gold']))

with col2:
    st.subheader("Evolución de Atributos F-T-M")
    st.plotly_chart(px.line(df, x='Edad', y=['Fisico', 'Tecnico', 'Mental']))

st.write("### Conclusión del Caso de Estudio")
pico = df.loc[df['Rating'].idxmax()]
st.info(f"El jugador alcanza su pico de **{pico['Rating']:.1f}** a la edad de **{pico['Edad']:.1f}** años.")