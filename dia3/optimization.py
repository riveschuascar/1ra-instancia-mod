import itertools
from typing import Tuple, List

import numpy as np
import pandas as pd

from player_development_model import (
    PlayerDevelopmentModel,
    PlayerParameters,
    constant_training
)

# ==================================================
# CONFIGURACIÓN GLOBAL DE LA OPTIMIZACIÓN
# ==================================================
TRAINING_VALUES = [0.5, 0.6, 0.7, 0.8, 0.9]
SIMULATION_YEARS = 15
DT = 1.0

INJURY_PENALTY_WEIGHT = 15.0
FATIGUE_PENALTY_WEIGHT = 2.0
MAX_FT_DIFF = 0.3  # balance físico vs técnico


# ==================================================
# FUNCIÓN OBJETIVO
# ==================================================
def compute_score(
    results: pd.DataFrame,
    params: PlayerParameters
) -> Tuple[float, float]:
    """
    Calcula el score de una trayectoria.

    Score = rating máximo
            - penalización por severidad de lesión
            - penalización por carga competitiva
    """
    max_rating = results["Rating (R)"].max()

    injury_penalty = params.injury_severity * INJURY_PENALTY_WEIGHT
    fatigue_penalty = params.matches_per_week * FATIGUE_PENALTY_WEIGHT

    score = max_rating - injury_penalty - fatigue_penalty
    return score, max_rating


# ==================================================
# SIMULACIÓN DE UN RÉGIMEN
# ==================================================
def simulate_training_regime(
    training: Tuple[float, float, float],
    params: PlayerParameters,
    initial_state: dict
) -> pd.DataFrame:
    """
    Simula un régimen de entrenamiento fijo.
    """
    E_F, E_T, E_M = training
    model = PlayerDevelopmentModel(params)

    return model.simulate(
        initial_state=initial_state,
        training_schedule=constant_training(E_F, E_T, E_M),
        duration_days=SIMULATION_YEARS * 365,
        dt=DT
    )


# ==================================================
# OPTIMIZACIÓN DE TRAYECTORIAS
# ==================================================
def optimize_training() -> Tuple[float, Tuple[float, float, float], pd.DataFrame]:
    """
    Busca el régimen de entrenamiento óptimo bajo
    criterios de beneficio vs riesgo.
    """
    best_score = -np.inf
    best_training = None
    best_results = None

    initial_state = {
        "F": 0.60,
        "T": 0.55,
        "M": 0.50,
        "A": 18.0
    }

    base_params = PlayerParameters(
        injury_start_day=300,
        injury_duration=120,
        injury_severity=0.4,
        matches_per_week=3
    )

    for E_F, E_T, E_M in itertools.product(TRAINING_VALUES, repeat=3):

        # Restricción de balance físico vs técnico
        if abs(E_F - E_T) > MAX_FT_DIFF:
            continue

        results = simulate_training_regime(
            training=(E_F, E_T, E_M),
            params=base_params,
            initial_state=initial_state
        )

        score, _ = compute_score(results, base_params)

        if score > best_score:
            best_score = score
            best_training = (E_F, E_T, E_M)
            best_results = results

    return best_score, best_training, best_results


# ==================================================
# MAIN
# ==================================================
def main():
    score, training, results = optimize_training()

    print("=" * 60)
    print("RÉGIMEN ÓPTIMO ENCONTRADO")
    print("=" * 60)
    print(f"Score optimizado: {score:.2f}")
    print("Entrenamiento óptimo:")
    print(f"  Físico  (E_F): {training[0]}")
    print(f"  Técnico (E_T): {training[1]}")
    print(f"  Mental  (E_M): {training[2]}")
    print(f"Rating máximo alcanzado: {results['Rating (R)'].max():.2f}")

    results.to_csv("optimal_trajectory.csv", index=False)
    print("\nResultados guardados en optimal_trajectory.csv")


if __name__ == "__main__":
    main()
