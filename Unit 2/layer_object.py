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

    def forward(self, inputs):    # inputs is X
        self.outputs = np.dot(inputs, self.weights) + self.biases

# =========================================================
# Testing Dense Layer (treat this section as a script)

import numpy as np
np.random.seed(0)    # For repeatability

class Dense_Layer:
    def __init__(self, n_inputs, n_neurons):
        self.weights = 0.01 * np.random.randn(n_inputs, n_neurons)
        self.biases = np.zeros([1, n_neurons])

    def forward(self, inputs):    # inputs is X
        self.outputs = np.dot(inputs, self.weights) + self.biases

X = np.array([[2, 0, -7],
             [-2, 9, 0],
             [5, 1, -1],
             [3, 0, -4]])

my_layer = Dense_Layer(3, 2)    # 3 input features, 2 neurons
print(my_layer.weights)
print(my_layer.biases)

# Changing Weights and Biases
my_layer.weights = np.array([[5, -2, -10],    # neuron 1's weights
                             [0, 9, -5]]).T    # neuron 2's weights
my_layer.biases = np.array([-10, 7])    # 2 biases, one for each neuron, vector

# Prediction
my_layer.forward(X)
print(my_layer.outputs)

print("\n")
