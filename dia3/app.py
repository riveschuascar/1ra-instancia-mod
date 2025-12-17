import streamlit as st
import plotly.express as px
from engine_final import DynamicSystem, PlayerParameters

st.title("⚽ Pipeline Integrado: Sistema Dinámico de Fútbol")

# --- PARÁMETROS ---
st.sidebar.header("Configuración de Entrenamiento")
partidos = st.sidebar.slider("Partidos por semana (Fatiga)", 0, 3, 1)
intensidad = st.sidebar.slider("Intensidad de Práctica", 0.0, 1.0, 0.7)

st.sidebar.header("Evento de Lesión")
hay_lesion = st.sidebar.checkbox("¿Ocurre una lesión?")
dia_lesion = st.sidebar.number_input("Día de inicio (Ej: 365 = 1 año)", 100, 5000, 365)
duracion = st.sidebar.slider("Días de baja", 30, 300, 90)

# Calibración (Estudiante C)
p = PlayerParameters(0.18, 0.15, 0.10, 0.008, 0.006, 0.005, 0.05, 0.03, 0.4, 0.4, 0.2, 27.0)

# --- SIMULACIÓN ---
lesion_args = {'dia': dia_lesion, 'duracion': duracion, 'severidad': 0.8} if hay_lesion else None

sim = DynamicSystem(p)
df = sim.run_simulation(15, {'F':0.5, 'T':0.5, 'M':0.4, 'A':18.0}, intensidad, partidos, lesion_args)

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