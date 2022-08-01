print("\n")

import numpy as np    # version 1.22.2
np.random.seed(0)    # For repeatability

class Dense_Layer:
    def __init__(self, n_inputs, n_neurons):
        self.weights = 0.01 * np.random.randn(n_inputs, n_neurons)
        self.biases = np.zeros([1, n_neurons])

    def forward(self, inputs):    # inputs is X
        self.inputs = inputs
        self.outputs = np.dot(inputs, self.weights) + self.biases

    def backward(self, dvalues):
        self.dweights = np.dot(self.inputs.T, dvalues)
        self.dbiases = np.sum(dvalues, axis=0, keepdims=True)
        self.dinputs = np.dot(dvalues, self.weights.T)


class ReLU_Activation:
    def forward(self, inputs):    # inputs are outputs from dense layer
        self.inputs = inputs    # save inputs for backward method
        self.outputs = np.maximum(0, inputs)

    def backward(self, dvalues):    # dvalues has same shape as inputs
        self.dinputs = dvalues * (self.inputs >= 0)


print("\n")
