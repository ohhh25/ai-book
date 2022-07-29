import matplotlib.pyplot as plt    # version 3.4.0
import numpy as np    # version 1.22.2
import pandas as pd    # version 1.4.0
np.random.seed(0)    # For repeatability

class Sigmoid_Activation:
    def forward(self, inputs):    # inputs are outputs from dense layer
        self.outputs = 1 / (1 + np.exp(-inputs))
        
    def backward(self, dvalues):    # dvalues has same shape as inputs
        self.dinputs = dvalues * (self.outputs * (1 - self.outputs))

random_data = np.random.randn(10, 1)
print(random_data)

activation_function = Sigmoid_Activation()
activation_function.forward(random_data)
print("\n", activation_function.outputs)
