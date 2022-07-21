print("\n")

import numpy as np    # version 1.22.2
np.random.seed(0)    # For repeatability

class Softmax_Activation:
    def forward(self, inputs):
        exponentiated = np.exp(inputs)
        self.outputs = exponentiated / np.sum(exponentiated, axis=1, keepdims=True)


y_hat = np.array([[-2, 2, 5]])
softmax = Softmax_Activation()
softmax.forward(y_hat)
percentages = softmax.outputs * 100    # convert to percentage
print(percentages)
print(np.sum(percentages, axis=1))    # should be 100%


print("\n")
