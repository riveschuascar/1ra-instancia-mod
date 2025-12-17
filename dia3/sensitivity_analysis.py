import matplotlib.pyplot as plt

from player_development_model import (
    PlayerDevelopmentModel,
    PlayerParameters,
    constant_training
)

# ============================
# FUNCIÓN BASE DE SIMULACIÓN
# ============================
def simulate_case(name, params, training_fn, years=15):
    model = PlayerDevelopmentModel(params)

    initial_state = {
        "F": 0.60,
        "T": 0.55,
        "M": 0.50,
        "A": 18.0
    }

    results = model.simulate(
        initial_state=initial_state,
        training_schedule=training_fn,
        duration_days=years * 365,
        dt=1.0
    )

    results["Caso"] = name
    return results


# ============================
# ANÁLISIS DE SENSIBILIDAD
# ============================
def sensitivity_analysis():

    cases = []

    # ----------------------------
    # CASO BASE
    # ----------------------------
    base_params = PlayerParameters(
        injury_start_day=300,
        injury_duration=120,
        injury_severity=0.3,
        matches_per_week=2
    )

    base_training = constant_training(0.6, 0.6, 0.6)

    cases.append(
        simulate_case("Base", base_params, base_training)
    )

    # ----------------------------
    # VARIAR INTENSIDAD DE ENTRENAMIENTO
    # ----------------------------
    cases.append(
        simulate_case(
            "Entrenamiento bajo",
            base_params,
            constant_training(0.4, 0.4, 0.4)
        )
    )

    cases.append(
        simulate_case(
            "Entrenamiento alto",
            base_params,
            constant_training(0.9, 0.9, 0.9)
        )
    )

    # ----------------------------
    # LESIÓN GRAVE
    # ----------------------------
    severe_injury_params = PlayerParameters(
        injury_start_day=300,
        injury_duration=240,
        injury_severity=0.8,
        matches_per_week=2
    )

    cases.append(
        simulate_case(
            "Lesión grave",
            severe_injury_params,
            base_training
        )
    )

    # ----------------------------
    # REGÍMENES DIFERENTES
    # ----------------------------
    cases.append(
        simulate_case(
            "Más físico",
            base_params,
            constant_training(0.8, 0.5, 0.4)
        )
    )

    cases.append(
        simulate_case(
            "Más técnico",
            base_params,
            constant_training(0.5, 0.8, 0.4)
        )
    )

    return cases


# ============================
# VISUALIZACIÓN
# ============================
def plot_sensitivity(cases):
    plt.figure(figsize=(13, 7))

    for df in cases:
        label = df["Caso"].iloc[0]
        plt.plot(df["Edad"], df["Rating (R)"], label=label, linewidth=2)

    plt.xlabel("Edad (años)")
    plt.ylabel("Rating")
    plt.title("Análisis de Sensibilidad del Desarrollo del Jugador")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()


# ============================
# MAIN
# ============================
if __name__ == "__main__":
    cases = sensitivity_analysis()
    plot_sensitivity(cases)
