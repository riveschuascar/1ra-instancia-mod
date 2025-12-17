import matplotlib.pyplot as plt

from player_development_model import (
    PlayerDevelopmentModel,
    PlayerParameters,
    adaptive_training,
    constant_training
)

# ============================
# FUNCIÓN AUXILIAR PARA CORRER ESCENARIOS
# ============================
def run_scenario(name, params, initial_state, training_fn, years=15):
    model = PlayerDevelopmentModel(params)
    results = model.simulate(
        initial_state=initial_state,
        training_schedule=training_fn,
        duration_days=years * 365,
        dt=1.0
    )
    results["Escenario"] = name
    return results


# ============================
# DEFINICIÓN DE ESCENARIOS
# ============================
def simulate_all_scenarios():
    
    scenarios = []
    # 1 JOVEN PROMESA
    params_joven = PlayerParameters(
        alpha_F=0.20,
        alpha_T=0.18,
        alpha_M=0.15,
        injury_severity=0.2,
        matches_per_week=1
    )

    initial_joven = {"F": 0.60, "T": 0.55, "M": 0.50, "A": 18.0}

    scenarios.append(
        run_scenario(
            "Joven promesa",
            params_joven,
            initial_joven,
            adaptive_training
        )
    )

    # 2 LENTO PERO SEGURO
    params_lento = PlayerParameters(
        alpha_F=0.10,
        alpha_T=0.10,
        alpha_M=0.10,
        injury_severity=0.1,
        matches_per_week=2
    )

    initial_lento = {"F": 0.55, "T": 0.50, "M": 0.45, "A": 19.0}

    scenarios.append(
        run_scenario(
            "Lento pero seguro",
            params_lento,
            initial_lento,
            constant_training(0.6, 0.6, 0.6)
        )
    )

    # 3 EARLY BLOOMER
    params_early = PlayerParameters(
        alpha_F=0.25,
        alpha_T=0.20,
        alpha_M=0.10,
        injury_severity=0.4,
        matches_per_week=4
    )

    initial_early = {"F": 0.70, "T": 0.65, "M": 0.45, "A": 17.0}

    scenarios.append(
        run_scenario(
            "Early bloomer",
            params_early,
            initial_early,
            adaptive_training
        )
    )

    # 4 LATE BLOOMER
    params_late = PlayerParameters(
        alpha_F=0.08,
        alpha_T=0.10,
        alpha_M=0.15,
        injury_severity=0.15,
        matches_per_week=1
    )

    initial_late = {"F": 0.50, "T": 0.45, "M": 0.55, "A": 20.0}

    scenarios.append(
        run_scenario(
            "Late bloomer",
            params_late,
            initial_late,
            adaptive_training
        )
    )

    return scenarios


# ============================
# VISUALIZACIÓN COMPARATIVA
# ============================
def plot_comparison(scenarios):
    plt.figure(figsize=(12, 7))

    for df in scenarios:
        name = df["Escenario"].iloc[0]
        plt.plot(df["Edad"], df["Rating (R)"], label=name, linewidth=2)

    plt.xlabel("Edad (años)")
    plt.ylabel("Rating")
    plt.title("Comparación de Escenarios de Desarrollo")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()


# ============================
# MAIN
# ============================
if __name__ == "__main__":
    scenarios = simulate_all_scenarios()
    plot_comparison(scenarios)
