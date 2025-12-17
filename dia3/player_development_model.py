"""
Modelo de Desarrollo de Jugador
Sistema de ecuaciones diferenciales para simular la evolución de atributos físicos,
técnicos y mentales de un jugador a lo largo del tiempo.

Implementación con método Runge-Kutta de 4to orden (RK4) para alta precisión.
"""

import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass
from typing import Tuple, List, Callable
import pandas as pd


@dataclass
class PlayerParameters:
    """Parámetros del modelo de desarrollo del jugador"""
    # Tasas de aprendizaje (α)
    alpha_F: float = 0.15  # Tasa aprendizaje físico
    alpha_T: float = 0.12  # Tasa aprendizaje técnico
    alpha_M: float = 0.10  # Tasa aprendizaje mental
    
    # Tasas de decaimiento (β)
    beta_F: float = 0.008  # Decaimiento físico
    beta_T: float = 0.006  # Decaimiento técnico
    beta_M: float = 0.005  # Decaimiento mental
    
    # Factores de sinergia
    gamma_FT: float = 0.05  # Sinergia físico -> técnico
    delta: float = 0.03     # Sinergia (F+T) -> mental
    
    # Pesos para el rating global
    w_F: float = 0.35  # Peso físico
    w_T: float = 0.40  # Peso técnico
    w_M: float = 0.25  # Peso mental
    
    # Ruido estocástico
    epsilon: float = 0.5  # Fluctuación diaria
    
    # Parámetros de la curva gaussiana para edad óptima
    A_opt: float = 27.0  # Edad óptima
    sigma: float = 5.0   # Desviación estándar de la curva
    
    # Edades críticas para decaimiento
    age_threshold_F: float = 30.0
    age_threshold_T: float = 32.0
    age_threshold_M: float = 35.0


class PlayerDevelopmentModel:
    """Modelo de desarrollo de jugador con integración RK4"""
    
    def __init__(self, params: PlayerParameters = None):
        """
        Inicializa el modelo con parámetros dados
        
        Args:
            params: Parámetros del modelo. Si es None, usa valores por defecto
        """
        self.params = params if params is not None else PlayerParameters()
    
    def age_factor(self, A: float) -> float:
        """
        Factor gaussiano que modula el aprendizaje según la edad
        
        Args:
            A: Edad actual del jugador
            
        Returns:
            Factor entre 0 y 1 (máximo en A_opt)
        """
        p = self.params
        return np.exp(-((A - p.A_opt)**2) / (2 * p.sigma**2))
    
    def dF_dt(self, F: float, T: float, M: float, A: float, E_F: float) -> float:
        """
        Derivada temporal del atributo Físico
        
        dF/dt = α_F * E_F * exp(-(A-A_opt)²/2σ²) - β_F * max(0, A-30) * F
        
        Args:
            F: Puntuación física actual (0-1)
            T: Puntuación técnica actual (0-1)
            M: Puntuación mental actual (0-1)
            A: Edad actual (años)
            E_F: Intensidad de entrenamiento físico (0-1)
            
        Returns:
            Tasa de cambio de F
        """
        p = self.params
        learning = p.alpha_F * E_F * self.age_factor(A)
        decay = p.beta_F * max(0, A - p.age_threshold_F) * F
        return learning - decay
    
    def dT_dt(self, F: float, T: float, M: float, A: float, E_T: float) -> float:
        """
        Derivada temporal del atributo Técnico
        
        dT/dt = α_T * E_T + γ_FT * F - β_T * max(0, A-32) * T
        
        Args:
            F: Puntuación física actual (0-1)
            T: Puntuación técnica actual (0-1)
            M: Puntuación mental actual (0-1)
            A: Edad actual (años)
            E_T: Intensidad de entrenamiento técnico (0-1)
            
        Returns:
            Tasa de cambio de T
        """
        p = self.params
        learning = p.alpha_T * E_T
        synergy = p.gamma_FT * F
        decay = p.beta_T * max(0, A - p.age_threshold_T) * T
        return learning + synergy - decay
    
    def dM_dt(self, F: float, T: float, M: float, A: float, E_M: float) -> float:
        """
        Derivada temporal del atributo Mental
        
        dM/dt = α_M * E_M + δ(F+T) - β_M * max(0, A-35) * M
        
        Args:
            F: Puntuación física actual (0-1)
            T: Puntuación técnica actual (0-1)
            M: Puntuación mental actual (0-1)
            A: Edad actual (años)
            E_M: Intensidad de entrenamiento mental (0-1)
            
        Returns:
            Tasa de cambio de M
        """
        p = self.params
        learning = p.alpha_M * E_M
        synergy = p.delta * (F + T)
        decay = p.beta_M * max(0, A - p.age_threshold_M) * M
        return learning + synergy - decay
    
    def dR_dt(self, dF: float, dT: float, dM: float) -> float:
        """
        Derivada temporal del Rating global
        
        dR/dt = w_F * dF/dt + w_T * dT/dt + w_M * dM/dt + ε
        
        Args:
            dF: Tasa de cambio de F
            dT: Tasa de cambio de T
            dM: Tasa de cambio de M
            
        Returns:
            Tasa de cambio de R
        """
        p = self.params
        # Ruido estocástico (simplificado como término constante)
        noise = np.random.normal(0, p.epsilon / 365)
        return p.w_F * dF + p.w_T * dT + p.w_M * dM + noise
    
    def dA_dt(self) -> float:
        """
        Derivada temporal de la Edad
        
        dA/dt = 1/365 (cambio diario)
        
        Returns:
            Tasa de cambio de edad (en años por día)
        """
        return 1.0 / 365.0
    
    def system_derivatives(self, state: np.ndarray, training: Tuple[float, float, float]) -> np.ndarray:
        """
        Calcula todas las derivadas del sistema
        
        Args:
            state: Vector de estado [F, T, M, R, A]
            training: Tupla (E_F, E_T, E_M) con intensidades de entrenamiento
            
        Returns:
            Vector de derivadas [dF/dt, dT/dt, dM/dt, dR/dt, dA/dt]
        """
        F, T, M, R, A = state
        E_F, E_T, E_M = training
        
        dF = self.dF_dt(F, T, M, A, E_F)
        dT = self.dT_dt(F, T, M, A, E_T)
        dM = self.dM_dt(F, T, M, A, E_M)
        dR = self.dR_dt(dF, dT, dM)
        dA = self.dA_dt()
        
        return np.array([dF, dT, dM, dR, dA])
    
    def rk4_step(self, state: np.ndarray, dt: float, training: Tuple[float, float, float]) -> np.ndarray:
        """
        Un paso de integración usando Runge-Kutta de 4to orden
        
        Args:
            state: Vector de estado actual [F, T, M, R, A]
            dt: Paso de tiempo
            training: Tupla (E_F, E_T, E_M) con intensidades de entrenamiento
            
        Returns:
            Nuevo vector de estado después de dt
        """
        # Coeficientes RK4
        k1 = self.system_derivatives(state, training)
        k2 = self.system_derivatives(state + dt * k1 / 2, training)
        k3 = self.system_derivatives(state + dt * k2 / 2, training)
        k4 = self.system_derivatives(state + dt * k3, training)
        
        # Actualización RK4
        new_state = state + (dt / 6) * (k1 + 2*k2 + 2*k3 + k4)
        
        # Restricciones: F, T, M deben estar en [0, 1]
        new_state[0] = np.clip(new_state[0], 0, 1)  # F
        new_state[1] = np.clip(new_state[1], 0, 1)  # T
        new_state[2] = np.clip(new_state[2], 0, 1)  # M
        
        # R debe estar en [40, 100]
        new_state[3] = np.clip(new_state[3], 40, 100)  # R
        
        return new_state
    
    def compute_rating(self, F: float, T: float, M: float) -> float:
        """
        Calcula el rating global a partir de los atributos
        
        R = 40 + 60 * (w_F*F + w_T*T + w_M*M)
        
        Args:
            F: Puntuación física (0-1)
            T: Puntuación técnica (0-1)
            M: Puntuación mental (0-1)
            
        Returns:
            Rating global (40-100)
        """
        p = self.params
        weighted_sum = p.w_F * F + p.w_T * T + p.w_M * M
        return 40 + 60 * weighted_sum
    
    def simulate(self, 
                 initial_state: dict,
                 training_schedule: Callable[[float], Tuple[float, float, float]],
                 duration_days: int,
                 dt: float = 1.0) -> pd.DataFrame:
        """
        Simula el desarrollo del jugador durante un período de tiempo
        
        Args:
            initial_state: Estado inicial {'F': ..., 'T': ..., 'M': ..., 'A': ...}
            training_schedule: Función que toma edad y retorna (E_F, E_T, E_M)
            duration_days: Duración de la simulación en días
            dt: Paso de tiempo en días
            
        Returns:
            DataFrame con la evolución temporal de todos los atributos
        """
        # Estado inicial
        F0 = initial_state['F']
        T0 = initial_state['T']
        M0 = initial_state['M']
        A0 = initial_state['A']
        R0 = self.compute_rating(F0, T0, M0)
        
        state = np.array([F0, T0, M0, R0, A0])
        
        # Arrays para almacenar resultados
        n_steps = int(duration_days / dt)
        times = np.zeros(n_steps + 1)
        F_history = np.zeros(n_steps + 1)
        T_history = np.zeros(n_steps + 1)
        M_history = np.zeros(n_steps + 1)
        R_history = np.zeros(n_steps + 1)
        A_history = np.zeros(n_steps + 1)
        
        # Guardar estado inicial
        times[0] = 0
        F_history[0], T_history[0], M_history[0], R_history[0], A_history[0] = state
        
        # Simulación
        for i in range(n_steps):
            training = training_schedule(state[4])  # training basado en edad actual
            state = self.rk4_step(state, dt, training)
            
            times[i+1] = (i+1) * dt
            F_history[i+1], T_history[i+1], M_history[i+1], R_history[i+1], A_history[i+1] = state
        
        # Crear DataFrame con resultados
        results = pd.DataFrame({
            'Día': times,
            'Edad': A_history,
            'Físico (F)': F_history,
            'Técnico (T)': T_history,
            'Mental (M)': M_history,
            'Rating (R)': R_history
        })
        
        return results


def constant_training(E_F: float, E_T: float, E_M: float) -> Callable[[float], Tuple[float, float, float]]:
    """
    Crea una función de entrenamiento constante
    
    Args:
        E_F: Intensidad física constante
        E_T: Intensidad técnica constante
        E_M: Intensidad mental constante
        
    Returns:
        Función que retorna siempre los mismos valores
    """
    return lambda age: (E_F, E_T, E_M)


def adaptive_training(age: float) -> Tuple[float, float, float]:
    """
    Programa de entrenamiento adaptativo según la edad
    
    Args:
        age: Edad actual del jugador
        
    Returns:
        Tupla (E_F, E_T, E_M) con intensidades ajustadas a la edad
    """
    if age < 23:
        # Jugador joven: enfoque balanceado con énfasis físico
        return (0.8, 0.7, 0.5)
    elif age < 28:
        # Peak años: máximo entrenamiento
        return (0.9, 0.85, 0.7)
    elif age < 32:
        # Madurez: mantener físico, mejorar mental
        return (0.7, 0.8, 0.85)
    else:
        # Veterano: reducir físico, maximizar mental
        return (0.5, 0.6, 0.9)


def plot_results(results: pd.DataFrame, save_path: str = None):
    """
    Visualiza los resultados de la simulación
    
    Args:
        results: DataFrame con resultados de la simulación
        save_path: Ruta donde guardar la figura (opcional)
    """
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('Simulación del Desarrollo del Jugador', fontsize=16, fontweight='bold')
    
    # Gráfica 1: Atributos individuales vs Edad
    ax1 = axes[0, 0]
    ax1.plot(results['Edad'], results['Físico (F)'], label='Físico (F)', linewidth=2, color='#e74c3c')
    ax1.plot(results['Edad'], results['Técnico (T)'], label='Técnico (T)', linewidth=2, color='#3498db')
    ax1.plot(results['Edad'], results['Mental (M)'], label='Mental (M)', linewidth=2, color='#2ecc71')
    ax1.set_xlabel('Edad (años)', fontsize=11)
    ax1.set_ylabel('Puntuación (0-1)', fontsize=11)
    ax1.set_title('Evolución de Atributos Individuales', fontsize=12, fontweight='bold')
    ax1.legend(loc='best')
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim([0, 1.05])
    
    # Gráfica 2: Rating global vs Edad
    ax2 = axes[0, 1]
    ax2.plot(results['Edad'], results['Rating (R)'], linewidth=2.5, color='#9b59b6')
    ax2.set_xlabel('Edad (años)', fontsize=11)
    ax2.set_ylabel('Rating (40-100)', fontsize=11)
    ax2.set_title('Evolución del Rating Global', fontsize=12, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim([40, 100])
    ax2.axhline(y=70, color='orange', linestyle='--', alpha=0.5, label='Umbral 70')
    ax2.legend()
    
    # Gráfica 3: Comparación de tasas de cambio
    ax3 = axes[1, 0]
    dF = np.gradient(results['Físico (F)'].values, results['Día'].values)
    dT = np.gradient(results['Técnico (T)'].values, results['Día'].values)
    dM = np.gradient(results['Mental (M)'].values, results['Día'].values)
    
    ax3.plot(results['Edad'], dF * 365, label='dF/dt', linewidth=1.5, alpha=0.7, color='#e74c3c')
    ax3.plot(results['Edad'], dT * 365, label='dT/dt', linewidth=1.5, alpha=0.7, color='#3498db')
    ax3.plot(results['Edad'], dM * 365, label='dM/dt', linewidth=1.5, alpha=0.7, color='#2ecc71')
    ax3.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    ax3.set_xlabel('Edad (años)', fontsize=11)
    ax3.set_ylabel('Tasa de cambio (por año)', fontsize=11)
    ax3.set_title('Tasas de Cambio de Atributos', fontsize=12, fontweight='bold')
    ax3.legend(loc='best')
    ax3.grid(True, alpha=0.3)
    
    # Gráfica 4: Heatmap de contribución al rating
    ax4 = axes[1, 1]
    contribution_F = results['Físico (F)'] * 0.35 * 60 + 40 * 0.35
    contribution_T = results['Técnico (T)'] * 0.40 * 60
    contribution_M = results['Mental (M)'] * 0.25 * 60
    
    ax4.fill_between(results['Edad'], 40, contribution_F, alpha=0.3, color='#e74c3c', label='Contribución Física')
    ax4.fill_between(results['Edad'], contribution_F, contribution_F + contribution_T, 
                     alpha=0.3, color='#3498db', label='Contribución Técnica')
    ax4.fill_between(results['Edad'], contribution_F + contribution_T, 
                     contribution_F + contribution_T + contribution_M, 
                     alpha=0.3, color='#2ecc71', label='Contribución Mental')
    ax4.plot(results['Edad'], results['Rating (R)'], 'k-', linewidth=2, label='Rating Total')
    
    ax4.set_xlabel('Edad (años)', fontsize=11)
    ax4.set_ylabel('Rating', fontsize=11)
    ax4.set_title('Descomposición del Rating por Componente', fontsize=12, fontweight='bold')
    ax4.legend(loc='best')
    ax4.grid(True, alpha=0.3)
    ax4.set_ylim([40, 100])
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Figura guardada en: {save_path}")
    
    return fig


def main():
    """Ejemplo de uso del modelo"""
    
    # Crear modelo con parámetros por defecto
    model = PlayerDevelopmentModel()
    
    # Estado inicial: jugador joven con atributos iniciales moderados
    initial_state = {
        'F': 0.65,  # Físico inicial
        'T': 0.55,  # Técnico inicial
        'M': 0.45,  # Mental inicial
        'A': 20.0   # Edad inicial: 20 años
    }
    
    # Simular 15 años de desarrollo (20 a 35 años)
    duration_years = 5
    results = model.simulate(
        initial_state=initial_state,
        training_schedule=adaptive_training,
        duration_days=duration_years * 365,
        dt=1.0  # Paso de 1 día
    )
    
    # Visualizar resultados
    plot_results(results, save_path='player_development_simulation.png')
    
    # Imprimir estadísticas clave
    print("\n" + "="*60)
    print("RESUMEN DE LA SIMULACIÓN")
    print("="*60)
    print(f"\nEstado Inicial (Edad {initial_state['A']:.1f}):")
    print(f"  Físico: {initial_state['F']:.3f}")
    print(f"  Técnico: {initial_state['T']:.3f}")
    print(f"  Mental: {initial_state['M']:.3f}")
    print(f"  Rating: {model.compute_rating(initial_state['F'], initial_state['T'], initial_state['M']):.1f}")
    
    # Encontrar el pico de rendimiento
    peak_idx = results['Rating (R)'].idxmax()
    peak_age = results.loc[peak_idx, 'Edad']
    peak_rating = results.loc[peak_idx, 'Rating (R)']
    
    print(f"\nPico de Rendimiento:")
    print(f"  Edad: {peak_age:.1f} años")
    print(f"  Rating: {peak_rating:.1f}")
    print(f"  Físico: {results.loc[peak_idx, 'Físico (F)']:.3f}")
    print(f"  Técnico: {results.loc[peak_idx, 'Técnico (T)']:.3f}")
    print(f"  Mental: {results.loc[peak_idx, 'Mental (M)']:.3f}")
    
    # Estado final
    final_idx = len(results) - 1
    final_age = results.loc[final_idx, 'Edad']
    
    print(f"\nEstado Final (Edad {final_age:.1f}):")
    print(f"  Físico: {results.loc[final_idx, 'Físico (F)']:.3f}")
    print(f"  Técnico: {results.loc[final_idx, 'Técnico (T)']:.3f}")
    print(f"  Mental: {results.loc[final_idx, 'Mental (M)']:.3f}")
    print(f"  Rating: {results.loc[final_idx, 'Rating (R)']:.1f}")
    
    print("\n" + "="*60)
    
    # Guardar datos a CSV
    results.to_csv('player_development_data.csv', index=False)
    print("\nDatos guardados en: player_development_data.csv")
    
    return results, model


if __name__ == "__main__":
    results, model = main()
    plt.show()