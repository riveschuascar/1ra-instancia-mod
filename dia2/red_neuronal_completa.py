"""
Red Neuronal Completa - Día 2
Sistema de Análisis de Futbolistas FIFA

Implementa:
- Red 1: Predicción de Potencial Máximo (Regresión)
- Red 2: Clasificación de Perfil de Jugador (Clasificación)
- Backpropagation completo
- Funciones de pérdida (MSE y Cross-Entropy)
"""

import numpy as np
import pickle
import json
from pathlib import Path

# ============================================================================
# CAPAS DE LA RED NEURONAL
# ============================================================================

class DenseLayer:
    """Capa densa con regularización L2 y dropout."""
    def __init__(self, input_dim, output_dim, l2_lambda=0.01, dropout_rate=0.0):
        # Inicialización He para ReLU
        self.W = np.random.randn(input_dim, output_dim) * np.sqrt(2.0 / input_dim)
        self.b = np.zeros((1, output_dim))
        self.l2_lambda = l2_lambda
        self.dropout_rate = dropout_rate
        
        # Para almacenar valores durante forward/backward
        self.A_prev = None
        self.Z = None
        self.dropout_mask = None

    def forward(self, A, training=True):
        """Forward pass con dropout opcional."""
        self.A_prev = A
        self.Z = A @ self.W + self.b
        
        # Aplicar dropout durante entrenamiento
        if training and self.dropout_rate > 0:
            self.dropout_mask = np.random.rand(*self.Z.shape) > self.dropout_rate
            self.Z *= self.dropout_mask
            self.Z /= (1 - self.dropout_rate)  # Scaling inverso
        
        return self.Z

    def backward(self, dZ, lr):
        """Backward pass con regularización L2."""
        m = self.A_prev.shape[0]
        
        # Aplicar máscara de dropout si fue usado
        if self.dropout_mask is not None:
            dZ = dZ * self.dropout_mask / (1 - self.dropout_rate)
        
        # Gradientes
        dW = (self.A_prev.T @ dZ) / m + (self.l2_lambda * self.W)
        db = np.sum(dZ, axis=0, keepdims=True) / m
        dA_prev = dZ @ self.W.T
        
        # Actualización de parámetros
        self.W -= lr * dW
        self.b -= lr * db
        
        return dA_prev

# ============================================================================
# FUNCIONES DE ACTIVACIÓN
# ============================================================================

class ReLU:
    """Rectified Linear Unit."""
    def forward(self, Z, training=True):
        self.Z = Z
        return np.maximum(0, Z)
    
    def backward(self, dA):
        return dA * (self.Z > 0)

class Linear:
    """Activación lineal (identidad) para regresión."""
    def forward(self, Z, training=True):
        self.Z = Z
        return Z
    
    def backward(self, dA):
        return dA

class Softmax:
    """Softmax para clasificación multiclase."""
    def forward(self, Z, training=True):
        # Estabilidad numérica
        exp_Z = np.exp(Z - np.max(Z, axis=1, keepdims=True))
        self.A = exp_Z / np.sum(exp_Z, axis=1, keepdims=True)
        return self.A
    
    def backward(self, dA):
        return dA

# ============================================================================
# FUNCIONES DE PÉRDIDA
# ============================================================================

def mse_loss(y_true, y_pred):
    """Mean Squared Error para regresión."""
    return np.mean((y_true - y_pred) ** 2)

def mse_gradient(y_true, y_pred):
    """Gradiente de MSE."""
    return 2 * (y_pred - y_true) / y_true.shape[0]

def cross_entropy_loss(y_true, y_pred):
    """Cross-Entropy para clasificación multiclase."""
    m = y_true.shape[0]
    # Estabilidad numérica
    y_pred = np.clip(y_pred, 1e-10, 1 - 1e-10)
    return -np.sum(y_true * np.log(y_pred)) / m

def cross_entropy_gradient(y_true, y_pred):
    """Gradiente de Cross-Entropy con Softmax."""
    return y_pred - y_true

# ============================================================================
# RED NEURONAL COMPLETA
# ============================================================================

class NeuralNetwork:
    """
    Red Neuronal Multicapa con backpropagation completo.
    
    Soporta:
    - Múltiples capas ocultas
    - Regularización L2
    - Dropout
    - Early stopping
    - Optimización con mini-batch
    """
    
    def __init__(self, learning_rate=0.001, task='regression'):
        self.layers = []
        self.activations = []
        self.lr = learning_rate
        self.task = task  # 'regression' o 'classification'
        self.history = {
            'train_loss': [],
            'val_loss': [],
            'train_acc': [],
            'val_acc': []
        }
    
    def add(self, layer, activation):
        """Agrega una capa con su función de activación."""
        self.layers.append(layer)
        self.activations.append(activation)
    
    def forward(self, X, training=True):
        """Forward propagation a través de todas las capas."""
        A = X
        for layer, act in zip(self.layers, self.activations):
            Z = layer.forward(A, training=training)
            A = act.forward(Z, training=training)
        return A
    
    def backward(self, y_true, y_pred):
        """Backward propagation a través de todas las capas."""
        # Calcular gradiente de la pérdida
        if self.task == 'regression':
            dA = mse_gradient(y_true, y_pred)
        else:  # classification
            dA = cross_entropy_gradient(y_true, y_pred)
        
        # Backpropagation a través de las capas
        for layer, activation in zip(reversed(self.layers), reversed(self.activations)):
            dZ = activation.backward(dA)
            dA = layer.backward(dZ, self.lr)
    
    def compute_loss(self, y_true, y_pred):
        """Calcula la pérdida según la tarea."""
        if self.task == 'regression':
            return mse_loss(y_true, y_pred)
        else:
            return cross_entropy_loss(y_true, y_pred)
    
    def train_step(self, X_batch, y_batch):
        """Un paso de entrenamiento (forward + backward)."""
        # Forward pass
        y_pred = self.forward(X_batch, training=True)
        
        # Calcular pérdida
        loss = self.compute_loss(y_batch, y_pred)
        
        # Backward pass
        self.backward(y_batch, y_pred)
        
        return loss, y_pred
    
    def fit(self, X_train, y_train, X_val=None, y_val=None, 
            epochs=100, batch_size=32, early_stopping_patience=10, verbose=True):
        """
        Entrena la red neuronal.
        
        Args:
            X_train: Datos de entrenamiento
            y_train: Etiquetas de entrenamiento
            X_val: Datos de validación (opcional)
            y_val: Etiquetas de validación (opcional)
            epochs: Número de épocas
            batch_size: Tamaño del mini-batch
            early_stopping_patience: Paciencia para early stopping
            verbose: Mostrar progreso
        """
        n_samples = X_train.shape[0]
        best_val_loss = float('inf')
        patience_counter = 0
        
        for epoch in range(epochs):
            # Mezclar datos
            indices = np.random.permutation(n_samples)
            X_shuffled = X_train[indices]
            y_shuffled = y_train[indices]
            
            # Mini-batch training
            epoch_losses = []
            for i in range(0, n_samples, batch_size):
                X_batch = X_shuffled[i:i+batch_size]
                y_batch = y_shuffled[i:i+batch_size]
                
                loss, _ = self.train_step(X_batch, y_batch)
                epoch_losses.append(loss)
            
            # Calcular pérdida promedio de entrenamiento
            train_loss = np.mean(epoch_losses)
            self.history['train_loss'].append(train_loss)
            
            # Calcular accuracy si es clasificación
            if self.task == 'classification':
                train_pred = self.predict(X_train)
                train_acc = np.mean(train_pred == np.argmax(y_train, axis=1))
                self.history['train_acc'].append(train_acc)
            
            # Validación
            if X_val is not None and y_val is not None:
                val_pred = self.forward(X_val, training=False)
                val_loss = self.compute_loss(y_val, val_pred)
                self.history['val_loss'].append(val_loss)
                
                if self.task == 'classification':
                    val_acc = np.mean(self.predict(X_val) == np.argmax(y_val, axis=1))
                    self.history['val_acc'].append(val_acc)
                
                # Early stopping
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    patience_counter = 0
                    self.save_weights('best_model_weights.pkl')
                else:
                    patience_counter += 1
                
                if patience_counter >= early_stopping_patience:
                    if verbose:
                        print(f"Early stopping en época {epoch+1}")
                    self.load_weights('best_model_weights.pkl')
                    break
            
            # Mostrar progreso
            if verbose and (epoch + 1) % 10 == 0:
                if X_val is not None:
                    if self.task == 'classification':
                        print(f"Época {epoch+1}/{epochs} - Loss: {train_loss:.4f} - "
                              f"Acc: {train_acc:.4f} - Val Loss: {val_loss:.4f} - "
                              f"Val Acc: {val_acc:.4f}")
                    else:
                        print(f"Época {epoch+1}/{epochs} - Loss: {train_loss:.4f} - "
                              f"Val Loss: {val_loss:.4f}")
                else:
                    print(f"Época {epoch+1}/{epochs} - Loss: {train_loss:.4f}")
    
    def predict(self, X):
        """Realiza predicciones."""
        y_pred = self.forward(X, training=False)
        
        if self.task == 'classification':
            return np.argmax(y_pred, axis=1)
        else:
            return y_pred.flatten()
    
    def predict_proba(self, X):
        """Devuelve probabilidades para clasificación."""
        if self.task != 'classification':
            raise ValueError("predict_proba solo disponible para clasificación")
        return self.forward(X, training=False)
    
    def save_weights(self, filepath):
        """Guarda los pesos de la red."""
        weights = {
            'layers': [(layer.W, layer.b) for layer in self.layers],
            'lr': self.lr,
            'task': self.task,
            'history': self.history
        }
        with open(filepath, 'wb') as f:
            pickle.dump(weights, f)
    
    def load_weights(self, filepath):
        """Carga los pesos de la red."""
        with open(filepath, 'rb') as f:
            weights = pickle.load(f)
        
        for i, (W, b) in enumerate(weights['layers']):
            self.layers[i].W = W
            self.layers[i].b = b
        
        self.lr = weights['lr']
        self.task = weights['task']
        self.history = weights['history']
    
    def save_architecture(self, filepath):
        """Guarda la arquitectura de la red en JSON."""
        architecture = {
            'task': self.task,
            'learning_rate': self.lr,
            'layers': [
                {
                    'type': 'Dense',
                    'input_dim': layer.W.shape[0],
                    'output_dim': layer.W.shape[1],
                    'l2_lambda': layer.l2_lambda,
                    'dropout_rate': layer.dropout_rate,
                    'activation': type(act).__name__
                }
                for layer, act in zip(self.layers, self.activations)
            ]
        }
        
        with open(filepath, 'w') as f:
            json.dump(architecture, f, indent=2)

# ============================================================================
# FUNCIONES AUXILIARES PARA CREAR REDES
# ============================================================================

def create_regression_network(input_dim=20, l2_lambda=0.01, dropout_rate=0.2, lr=0.0005):
    """
    Red 1: Predicción de Potencial Máximo
    Arquitectura: 20-256-128-64-1
    """
    model = NeuralNetwork(learning_rate=lr, task='regression')
    
    # Capa 1: 20 -> 256
    model.add(DenseLayer(input_dim, 256, l2_lambda=l2_lambda, dropout_rate=dropout_rate), ReLU())
    
    # Capa 2: 256 -> 128
    model.add(DenseLayer(256, 128, l2_lambda=l2_lambda, dropout_rate=dropout_rate), ReLU())
    
    # Capa 3: 128 -> 64
    model.add(DenseLayer(128, 64, l2_lambda=l2_lambda, dropout_rate=dropout_rate), ReLU())
    
    # Capa de salida: 64 -> 1
    model.add(DenseLayer(64, 1, l2_lambda=0.0, dropout_rate=0.0), Linear())
    
    return model

def create_classification_network(input_dim=15, n_classes=7, l2_lambda=0.01, 
                                 dropout_rate=0.2, lr=0.001):
    """
    Red 2: Clasificación de Perfil de Jugador
    Arquitectura: 15-256-128-7
    """
    model = NeuralNetwork(learning_rate=lr, task='classification')
    
    # Capa 1: 15 -> 256
    model.add(DenseLayer(input_dim, 256, l2_lambda=l2_lambda, dropout_rate=dropout_rate), ReLU())
    
    # Capa 2: 256 -> 128
    model.add(DenseLayer(256, 128, l2_lambda=l2_lambda, dropout_rate=dropout_rate), ReLU())
    
    # Capa de salida: 128 -> n_classes
    model.add(DenseLayer(128, n_classes, l2_lambda=0.0, dropout_rate=0.0), Softmax())
    
    return model

if __name__ == "__main__":
    print("Red Neuronal Completa - Día 2")
    print("="*60)
    print("\nArquitecturas disponibles:")
    print("1. Red de Regresión: 20-256-128-64-1 (Potencial Máximo)")
    print("2. Red de Clasificación: 15-256-128-7 (Perfil de Jugador)")
    print("\nCaracterísticas implementadas:")
    print("✓ Backpropagation completo")
    print("✓ Regularización L2")
    print("✓ Dropout")
    print("✓ Early stopping")
    print("✓ Mini-batch training")
    print("✓ Funciones de pérdida (MSE y Cross-Entropy)")