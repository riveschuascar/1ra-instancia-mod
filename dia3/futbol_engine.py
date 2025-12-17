import numpy as np
import pandas as pd
from dataclasses import dataclass

@dataclass
class PlayerParameters:
    # Tasas de aprendizaje y decaimiento
    alpha_F: float; alpha_T: float; alpha_M: float
    beta_F: float; beta_T: float; beta_M: float
    # Sinergias y Pesos
    gamma: float; delta: float
    wF: float; wT: float; wM: float
    A_opt: float; sigma: float = 4.0

class FootballSimulator:
    def __init__(self, params: PlayerParameters):
        self.p = params

    def ecuaciones(self, state, E, injury_factor):
        F, T, M, A = state
        
        # dF/dt: Desarrollo físico con campana de Gauss
        gauss = np.exp(-((A - self.p.A_opt)**2) / (2 * self.p.sigma**2))
        dF = (self.p.alpha_F * E[0] * gauss * injury_factor) - (self.p.beta_F * max(0, A - 30) * F)
        
        # dT/dt: Desarrollo técnico + sinergia física
        dT = (self.p.alpha_T * E[1]) + (self.p.gamma * F) - (self.p.beta_T * max(0, A - 32) * T)
        
        # dM/dt: Desarrollo mental + sinergia (F+T)
        dM = (self.p.alpha_M * E[2]) + (self.p.delta * (F + T)) - (self.p.beta_M * max(0, A - 35) * M)
        
        dA = 1/365.0 # Avance diario de edad
        return np.array([dF, dT, dM, dA])

    def rk4_step(self, state, E, dt, injury_factor):
        k1 = self.ecuaciones(state, E, injury_factor)
        k2 = self.ecuaciones(state + k1*dt/2, E, injury_factor)
        k3 = self.ecuaciones(state + k2*dt/2, E, injury_factor)
        k4 = self.ecuaciones(state + k3*dt, E, injury_factor)
        return state + (dt/6.0) * (k1 + 2*k2 + 2*k3 + k4)

    def simulate(self, years, init_state, e_intensity, injury_event=None):
        state = np.array([init_state['F'], init_state['T'], init_state['M'], init_state['A']])
        history = []
        
        for day in range(int(years * 365)):
            # Lógica de lesión
            infac = 1.0
            if injury_event and injury_event['start'] <= day <= injury_event['end']:
                infac = 0.1 # El entrenamiento físico cae al 10%
            
            # Entrenamiento (E_F, E_T, E_M)
            E = [e_intensity] * 3 
            
            state = self.rk4_step(state, E, 1.0, infac)
            
            # Cálculo de Overall Rating (R)
            R = (state[0]*self.p.wF + state[1]*self.p.wT + state[2]*self.p.wM) * 100
            history.append(list(state) + [R])
            
        return pd.DataFrame(history, columns=['Fisico', 'Tecnico', 'Mental', 'Edad', 'Rating'])