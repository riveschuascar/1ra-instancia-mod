import streamlit as st
import plotly.express as px
import sqlite3
import pandas as pd
import random
from engine_final import DynamicSystem, PlayerParameters

conn = sqlite3.connect("integrationdataset.sqlite")
df = pd.read_sql_query("""
                       SELECT id, estimated_age, physical_score, technical_score, mental_score
                       FROM player_attributes_dataset WHERE estimated_age < 25;""", conn)

conn.close()

jugadores = df.sample(n=5, random_state=42)

st.title("⚽ Pipeline Integrado: Sistema Dinámico de Fútbol")

# --- JUGADOR -----
st.sidebar.header("Selección de Jugador")

# Seleccionar jugador por ID
player_id = st.sidebar.selectbox(
    "Jugador",
    jugadores["id"].tolist()
)

# --- PARÁMETROS ---
st.sidebar.header("Configuración de Entrenamiento")
partidos = st.sidebar.slider("Partidos por semana (Fatiga)", 0, 3, 1)
intensidad = st.sidebar.slider("Intensidad de Práctica", 0.0, 1.0, 0.5)

st.sidebar.header("Evento de Lesión")
hay_lesion = st.sidebar.checkbox("¿Ocurre una lesión?")
dia_lesion = st.sidebar.number_input("Día de inicio (Ej: 365 = 1 año)", 100, 5000, 365)
duracion = st.sidebar.slider("Días de baja", 30, 300, 90)

st.sidebar.header("Duracion en Años")
tiempo = st.sidebar.number_input("Tiempo en años (Ej: 5)", 1, 5000, 10)

# Calibración (Estudiante C)
p = PlayerParameters(
    alpha_F=0.0018,
    alpha_T=0.0015,
    alpha_M=0.0010,
    beta_F=0.008,
    beta_T=0.006,
    beta_M=0.005,
    gamma_FT=0.0005,
    delta=0.0003,
    w_F=0.4,
    w_T=0.4,
    w_M=0.2,
    A_opt=27.0,
    sigma=2.0
)

# Extraer fila del jugador seleccionado
datos_jugador = jugadores[jugadores["id"] == player_id].iloc[0]

# Atributos del jugador
edad = float(datos_jugador["estimated_age"])
F = float(datos_jugador["physical_score"])
T = float(datos_jugador["technical_score"])
M = float(datos_jugador["mental_score"])

# --- SIMULACIÓN ---
lesion_args = {'dia': dia_lesion, 'duracion': duracion, 'severidad': 0.8} if hay_lesion else None

sim = DynamicSystem(p)
df = sim.run_simulation(tiempo, {'F':F, 'T':T, 'M':M, 'A':edad}, intensidad, partidos, lesion_args)

# --- VISUALIZACIÓN ---

st.subheader("Visualización del Pipeline en Tiempo Real")
fig = px.line(df, x='Edad', y='Rating', title="Evolución del Overall Rating (R)")
# Resaltar zona de lesión si existe
if hay_lesion:
    fig.add_vrect(x0=df.iloc[dia_lesion]['Edad'], x1=df.iloc[dia_lesion+duracion]['Edad'], 
                 fillcolor="red", opacity=0.2, annotation_text="LESIÓN")
st.plotly_chart(fig)

st.write("### Tabla de Datos Procesados")
st.dataframe(df.describe())