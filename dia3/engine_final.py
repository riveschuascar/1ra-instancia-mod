import numpy as np
import pandas as pd
from dataclasses import dataclass

@dataclass
class PlayerParameters:
    # Tasas de aprendizaje y decaimiento (Rúbrica Estudiante C)
    alpha_F: float; alpha_T: float; alpha_M: float
    beta_F: float; beta_T: float; beta_M: float
    gamma: float; delta: float
    wF: float; wT: float; wM: float
    A_opt: float; sigma: float = 4.0

class DynamicSystem:
    def __init__(self, p: PlayerParameters):
        self.p = p

    def get_derivatives(self, state, E, injury_factor):
        F, T, M, A = state
        
        # dF/dt: Físico con decaimiento a partir de los 30
        gauss = np.exp(-((A - self.p.A_opt)**2) / (2 * self.p.sigma**2))
        dF = (self.p.alpha_F * E[0] * gauss * injury_factor) - (self.p.beta_F * max(0, A - 30) * F)
        
        # dT/dt: Técnico con sinergia gamma
        dT = (self.p.alpha_T * E[1]) + (self.p.gamma * F) - (self.p.beta_T * max(0, A - 32) * T)
        
        # dM/dt: Mental con sinergia delta
        dM = (self.p.alpha_M * E[2]) + (self.p.delta * (F + T)) - (self.p.beta_M * max(0, A - 35) * M)
        
        dA = 1/365.0
        return np.array([dF, dT, dM, dA])

    def rk4_step(self, state, E, dt, injury_factor):
        # Implementación RK4 Manual (Requisito Estudiante B)
        k1 = self.get_derivatives(state, E, injury_factor)
        k2 = self.get_derivatives(state + k1*dt/2, E, injury_factor)
        k3 = self.get_derivatives(state + k2*dt/2, E, injury_factor)
        k4 = self.get_derivatives(state + k3*dt, E, injury_factor)
        return state + (dt/6.0) * (k1 + 2*k2 + 2*k3 + k4)

    def run_simulation(self, years, init_vals, intensity, injury_age=None):
        state = np.array([init_vals['F'], init_vals['T'], init_vals['M'], init_vals['A']])
        history = []
        
        for day in range(int(years * 365)):
            # Lógica de lesiones (Estudiante A)
            infac = 1.0
            if injury_age and abs(state[3] - injury_age) < 0.1: # Lesión por 4 meses
                infac = 0.05 
            
            state = self.rk4_step(state, [intensity]*3, 1.0, infac)
            R = (state[0]*self.p.wF + state[1]*self.p.wT + state[2]*self.p.wM) * 100
            history.append(list(state) + [R])
            
        return pd.DataFrame(history, columns=['Fisico', 'Tecnico', 'Mental', 'Edad', 'Rating'])