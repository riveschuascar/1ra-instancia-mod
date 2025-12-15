import sqlite3
import numpy as np

class SQLiteDataset:
    '''Acceso a los datos de SQLite.'''
    def __init__(self, db_path, table_name):
        self.db_path = db_path
        self.table_name = table_name

    def load_columns(self, feature_cols, target_col):
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cols = feature_cols + [target_col]
        query = f"SELECT {','.join(cols)} FROM {self.table_name}"
        cursor.execute(query)

        data = cursor.fetchall()
        conn.close()

        data = np.array(data, dtype=float)
        X = data[:, :-1]
        y = data[:, -1]

        return X, y

# Estandarización de características con media y desviación estándar (0,1)
class StandardScaler:
    '''Estandarización de características.'''
    def fit(self, X):
        self.mean = X.mean(axis=0)
        self.std = X.std(axis=0) + 1e-8

    def transform(self, X):
        return (X - self.mean) / self.std

    def fit_transform(self, X):
        self.fit(X)
        return self.transform(X)

# Funciones de activacion
class ReLU:
    def forward(self, Z):
        self.Z = Z
        return np.maximum(0, Z)

    def backward(self, dA):
        return dA * (self.Z > 0)

class Linear:
    def forward(self, Z):
        return Z

    def backward(self, dA):
        return dA

class Softmax:
    def forward(self, Z):
        exp = np.exp(Z - np.max(Z, axis=1, keepdims=True))
        self.A = exp / np.sum(exp, axis=1, keepdims=True)
        return self.A

    def backward(self, dA):
        return dA

def mse_loss(y_true, y_pred):
    '''Cálculo del error cuadrático medio.'''
    return np.mean((y_true - y_pred) ** 2)
