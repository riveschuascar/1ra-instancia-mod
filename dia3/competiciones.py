import numpy as np

# ============================
# FATIGA POR COMPETICIONES
# ============================
def efecto_fatiga(partidos_por_semana):
    """
    Modela la fatiga acumulada por competiciones.

    partidos_por_semana : int
        Número de partidos que juega el futbolista por semana.

    Retorna:
        factor en [0.7, 1.0] que reduce las capacidades.
    """
    fatiga = partidos_por_semana * 0.05
    return np.clip(1.0 - fatiga, 0.7, 1.0)
