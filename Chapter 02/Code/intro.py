print("\n")

import numpy as np
np.random.seed(0)    # For repeatability

class Dense_Layer:
    def __init__(self, n_inputs, n_neurons):
        self.weights = 0.01 * np.random.randn(n_inputs, n_neurons)
        self.biases = np.zeros([1, n_neurons])

    def forward(self, inputs):    # inputs is X
        self.outputs = np.dot(inputs, self.weights) + self.biases

# hours studied
X = np.array([[0], [1], [2], [3], [4], [5], [6], [7], [8]])    # one input feature for each example
# percentage
y = np.array([[20], [30], [40], [50], [60], [70], [80], [90], [100]])    # one output feature for each example

dumb_model1 = Dense_Layer(1, 1)    # 1 input feature, 1 neuron (output feature)
dumb_model1.forward(X)
print(dumb_model1.outputs)

dumb_model2 = Dense_Layer(1, 1)    # 1 input feature, 1 neuron (output feature)
dumb_model2.forward(X)
print(dumb_model2.outputs)

print("\n")
