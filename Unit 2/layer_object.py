print("\n")

import numpy as np
class Dense_Layer:
    def __init__(self, n_inputs, n_neurons):
        self.weights = 0.01 * np.random.randn(n_neurons, n_inputs)
        self.biases = np.zeros([1, n_neurons])

# =========================================================
# Updated Layer with Prediction (forward) Method

class Dense_Layer:
    def __init__(self, n_inputs, n_neurons):
        self.weights = 0.01 * np.random.randn(n_inputs, n_neurons)
        self.biases = np.zeros([1, n_neurons])

    def forward(self, X):
        self.outputs = np.dot(X, self.weights) + self.biases

print("\n")
