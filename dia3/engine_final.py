import numpy as np
import pandas as pd
from dataclasses import dataclass
# IMPORTACIÓN DEL PIPELINE (Estudiante D)
from player_development_model import (
    PlayerDevelopmentModel,
    PlayerParameters
)

class DynamicSystem:
    def __init__(self, p: PlayerParameters):
        self.model = PlayerDevelopmentModel(p)

    def run_simulation(
        self,
        years: int,
        init_vals: dict,
        intensity: float,
        num_partidos: int,
        lesion_params: dict | None = None
    ) -> pd.DataFrame:

        # 1. Configurar parámetros externos
        self.model.params.matches_per_week = num_partidos

        if lesion_params is not None:
            self.model.params.injury_start_day = lesion_params['dia']
            self.model.params.injury_duration = lesion_params['duracion']
            self.model.params.injury_severity = lesion_params['severidad']

        # 2. Estado inicial
        initial_state = {
            'F': init_vals['F'],
            'T': init_vals['T'],
            'M': init_vals['M'],
            'A': init_vals['A']
        }

        # 3. Training schedule equivalente al slider de intensidad
        def constant_training(age):
            return (intensity, intensity, intensity)

        # 4. Simulación
        results = self.model.simulate(
            initial_state=initial_state,
            training_schedule=constant_training,
            duration_days=int(years * 365),
            dt=1.0
        )

        # 5. Renombrar columnas para compatibilidad con Streamlit
        results = results.rename(columns={
            'Físico (F)': 'Fisico',
            'Técnico (T)': 'Tecnico',
            'Mental (M)': 'Mental',
            'Rating (R)': 'Rating'
        })

        return results[['Fisico', 'Tecnico', 'Mental', 'Edad', 'Rating']]
