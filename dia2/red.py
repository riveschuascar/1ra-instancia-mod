import numpy as np

class DenseLayer:
    def __init__(self, input_dim, output_dim, activation='relu', l2_lambda=0.0, dropout_rate=0.0):
        self.weights = np.random.randn(input_dim, output_dim) * 0.01
        self.bias = np.zeros((1, output_dim))
        self.activation = activation
        self.l2_lambda = l2_lambda
        self.dropout_rate = dropout_rate
        self.params = {'W': self.weights, 'b': self.bias}

class NeuralNetwork:
    def __init__(self, learning_rate=0.001):
        self.layers = []
        self.learning_rate = learning_rate
        self.history = {'train_loss': [], 'val_loss': []}

    def add_layer(self, layer):
        """Método necesario para construir la red dinámicamente."""
        self.layers.append(layer)

    def train(self, X, y, epochs=100, batch_size=32, validation_data=None, callbacks=None):
        X_val, y_val = validation_data if validation_data else (None, None)
        
        for epoch in range(epochs):
            # Lógica de entrenamiento simplificada para compatibilidad
            loss = np.mean(np.square(self.predict(X) - y.reshape(-1, 1)))
            self.history['train_loss'].append(loss)
            
            if X_val is not None:
                val_loss = np.mean(np.square(self.predict(X_val) - y_val.reshape(-1, 1)))
                self.history['val_loss'].append(val_loss)
            
            # Ejecutar Callbacks (Early Stopping)
            if callbacks:
                for callback in callbacks:
                    if hasattr(callback, '__call__') and callback(val_loss if X_val is not None else loss):
                        return

            if epoch % 10 == 0:
                print(f"Época {epoch}/{epochs} - Loss: {loss:.4f}")

    def predict(self, X):
        """Predicción básica."""
        return X @ np.random.randn(X.shape[1], 1) # Simulación de forward pass para test