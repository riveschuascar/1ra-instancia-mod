import numpy as np
import pandas as pd
from dataclasses import dataclass
# IMPORTACIÓN DEL PIPELINE (Estudiante D)
from lesiones import efecto_lesion
from competiciones import efecto_fatiga

@dataclass
class PlayerParameters:
    alpha_F: float; alpha_T: float; alpha_M: float
    beta_F: float; beta_T: float; beta_M: float
    gamma: float; delta: float
    wF: float; wT: float; wM: float
    A_opt: float; sigma: float = 4.0

class DynamicSystem:
    def __init__(self, p: PlayerParameters):
        self.p = p

    def get_derivatives(self, state, E, external_factors):
        F, T, M, A = state
        infac = external_factors['lesion']
        fatiga = external_factors['fatiga']
        
        # dF/dt: Aplicamos fatiga y lesión al entrenamiento físico
        gauss = np.exp(-((A - self.p.A_opt)**2) / (2 * self.p.sigma**2))
        dF = (self.p.alpha_F * (E[0] * fatiga) * gauss * infac) - (self.p.beta_F * max(0, A - 30) * F)
        
        # dT/dt
        dT = (self.p.alpha_T * E[1] * fatiga) + (self.p.gamma * F) - (self.p.beta_T * max(0, A - 32) * T)
        
        # dM/dt
        dM = (self.p.alpha_M * E[2]) + (self.p.delta * (F + T)) - (self.p.beta_M * max(0, A - 35) * M)
        
        dA = 1/365.0
        return np.array([dF, dT, dM, dA])

    def rk4_step(self, state, E, dt, external_factors):
        k1 = self.get_derivatives(state, E, external_factors)
        k2 = self.get_derivatives(state + k1*dt/2, E, external_factors)
        k3 = self.get_derivatives(state + k2*dt/2, E, external_factors)
        k4 = self.get_derivatives(state + k3*dt, E, external_factors)
        return state + (dt/6.0) * (k1 + 2*k2 + 2*k3 + k4)

    def run_simulation(self, years, init_vals, intensity, num_partidos, lesion_params=None):
        state = np.array([init_vals['F'], init_vals['T'], init_vals['M'], init_vals['A']])
        history = []
        
        for day in range(int(years * 365)):
            # 1. Pipeline de datos externos
            fac_fatiga = efecto_fatiga(num_partidos)
            
            fac_lesion = 1.0
            if lesion_params:
                fac_lesion = efecto_lesion(day, lesion_params['dia'], lesion_params['duracion'], lesion_params['severidad'])
            
            ext_factors = {'lesion': fac_lesion, 'fatiga': fac_fatiga}
            
            # 2. Paso numérico
            state = self.rk4_step(state, [intensity]*3, 1.0, ext_factors)
            
            # 3. Rating
            R = (state[0]*self.p.wF + state[1]*self.p.wT + state[2]*self.p.wM) * 100
            history.append(list(state) + [R, fac_lesion])
            
        return pd.DataFrame(history, columns=['Fisico', 'Tecnico', 'Mental', 'Edad', 'Rating', 'Factor_Lesion'])