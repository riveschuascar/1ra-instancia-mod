import itertools
import numpy as np
import pandas as pd

from player_development_model import (
    PlayerDevelopmentModel,
    PlayerParameters,
    constant_training
)

# ============================
# FUNCIÓN OBJETIVO
# ============================
def objective_function(results, params):
    """
    Función objetivo:
    - maximiza rating
    - penaliza fatiga y lesiones
    """
    max_rating = results["Rating (R)"].max()

    # Penalización por riesgo
    injury_penalty = params.injury_severity * 15
    fatigue_penalty = params.matches_per_week * 2

    score = max_rating - injury_penalty - fatigue_penalty
    return score, max_rating


# ============================
# OPTIMIZACIÓN DE TRAYECTORIAS
# ============================
def optimize_training():
    
    # Espacio de búsqueda (simple y defendible)
    E_values = [0.5, 0.6, 0.7, 0.8, 0.9]

    best_score = -np.inf
    best_config = None
    best_results = None

    # Estado inicial estándar
    initial_state = {
        "F": 0.60,
        "T": 0.55,
        "M": 0.50,
        "A": 18.0
    }

    # Parámetros base
    base_params = PlayerParameters(
        injury_start_day=300,
        injury_duration=120,
        injury_severity=0.4,
        matches_per_week=3
    )

    for E_F, E_T, E_M in itertools.product(E_values, repeat=3):

        # Balance físico vs técnico
        if abs(E_F - E_T) > 0.3:
            continue  # evita extremos poco realistas

        training_fn = constant_training(E_F, E_T, E_M)
        model = PlayerDevelopmentModel(base_params)

        results = model.simulate(
            initial_state=initial_state,
            training_schedule=training_fn,
            duration_days=15 * 365,
            dt=1.0
        )

        score, max_rating = objective_function(results, base_params)

        if score > best_score:
            best_score = score
            best_config = (E_F, E_T, E_M)
            best_results = results

    return best_score, best_config, best_results


# ============================
# MAIN
# ============================
if __name__ == "__main__":
    score, config, results = optimize_training()

    print("=" * 60)
    print("RÉGIMEN ÓPTIMO ENCONTRADO")
    print("=" * 60)
    print(f"Score optimizado: {score:.2f}")
    print(f"Entrenamiento óptimo:")
    print(f"  Físico (E_F): {config[0]}")
    print(f"  Técnico (E_T): {config[1]}")
    print(f"  Mental (E_M): {config[2]}")
    print(f"Rating máximo alcanzado: {results['Rating (R)'].max():.2f}")

    # Guardar resultados
    results.to_csv("optimal_trajectory.csv", index=False)
    print("\nResultados guardados en optimal_trajectory.csv")
