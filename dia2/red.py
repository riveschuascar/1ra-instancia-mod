import numpy as np

class DenseLayer:
    '''Capa densa con regularización L2.'''
    def __init__(self, input_dim, output_dim, l2_lambda=0.0):
        self.W = np.random.randn(input_dim, output_dim) * 0.01
        self.b = np.zeros((1, output_dim))
        self.l2_lambda = l2_lambda

    def forward(self, A):
        self.A_prev = A
        return A @ self.W + self.b

    def backward(self, dZ, lr):
        m = self.A_prev.shape[0]

        dW = (self.A_prev.T @ dZ) / m
        dW += self.l2_lambda * self.W

        db = np.sum(dZ, axis=0, keepdims=True) / m
        dA_prev = dZ @ self.W.T

        self.W -= lr * dW
        self.b -= lr * db

        return dA_prev
    
class NeuralNetwork:
    def __init__(self, learning_rate=0.001):
        self.layers = []
        self.activations = []
        self.lr = learning_rate

    def add(self, layer, activation):
        self.layers.append(layer)
        self.activations.append(activation)

    def forward(self, X):
        A = X
        for layer, act in zip(self.layers, self.activations):
            Z = layer.forward(A)
            A = act.forward(Z)
        return A
