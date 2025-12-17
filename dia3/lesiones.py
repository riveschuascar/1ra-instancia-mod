# ============================
# EFECTO DE LESIONES
# ============================
def efecto_lesion(t_dia, inicio_lesion, duracion_lesion, severidad):
    """
    Reducción temporal de capacidades por lesión.

    t_dia : float
        Día actual de la simulación.
    inicio_lesion : float
        Día en el que ocurre la lesión.
    duracion_lesion : float
        Duración de la lesión en días.
    severidad : float
        Severidad de la lesión [0, 1].

    Retorna:
        factor en [0, 1] que reduce las capacidades.
    """
    if inicio_lesion <= t_dia <= inicio_lesion + duracion_lesion:
        return 1.0 - severidad
    return 1.0
