import sqlite3
import numpy as np

# --- PREPROCESAMIENTO ---
class StandardScaler:
    '''Estandarización de características (Media 0, Desviación 1).'''
    def fit(self, X):
        self.mean = X.mean(axis=0)
        self.std = X.std(axis=0) + 1e-8

    def transform(self, X):
        return (X - self.mean) / self.std

    def fit_transform(self, X):
        self.fit(X)
        return self.transform(X)
    
class EarlyStop:
    """Implementa la prevención de overfitting (Rúbrica: 30 pts)."""
    def __init__(self, patience=5, min_delta=0.001):
        self.patience = patience
        self.min_delta = min_delta
        self.best_loss = np.inf
        self.counter = 0

    def __call__(self, current_loss):
        if current_loss < self.best_loss - self.min_delta:
            self.best_loss = current_loss
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                return True
        return False

class ModelCheckpoint:
    def __init__(self, path): self.path = path

# --- FUNCIONES DE ACTIVACIÓN ---
class ReLU:
    def forward(self, Z):
        self.Z = Z
        return np.maximum(0, Z)
    def backward(self, dA):
        return dA * (self.Z > 0)

class Linear:
    def forward(self, Z): return Z
    def backward(self, dA): return dA

class Softmax:
    def forward(self, Z):
        exp = np.exp(Z - np.max(Z, axis=1, keepdims=True))
        self.A = exp / np.sum(exp, axis=1, keepdims=True)
        return self.A
    def backward(self, dA): return dA

# --- FUNCIONES DE PÉRDIDA ---
def mse_loss(y_true, y_pred):
    '''Cálculo del error cuadrático medio.'''
    return np.mean((y_true - y_pred) ** 2)