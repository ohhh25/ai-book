print("\n")

import numpy as np    # version 1.22.2
np.random.seed(0)    # For repeatability

class Softmax_Activation:
    def forward(self, inputs):
        exponentiated = np.exp(inputs - np.max(inputs, axis=1, keepdims=True))
        self.outputs = exponentiated / np.sum(exponentiated, axis=1, keepdims=True)

class Categorical_Cross_Entropy_Cost:
    def forward(self, y_pred, y_true):
        y_pred_clipped = np.clip(y_pred, 1e-7, 1)
        correct_class_pred = y_pred_clipped[range(len(y_true)), y_true]
        return np.mean(-np.log(correct_class_pred))

class Softmax_Cross_Entropy:
    def __init__(self):
        self.softmax = Softmax_Activation()
        self.cce = Categorical_Cross_Entropy_Cost()

    def forward(self, y_pred, y_true):
        self.softmax.forward(y_pred)
        return self.cce.forward(self.softmax.outputs, y_true)

    def backward(self, y_true):
        self.dinputs = self.softmax.outputs.copy()
        self.dinputs[range(len(y_true)), y_true] -= 1
        self.dinputs = self.dinputs / len(y_true)


y_hat = np.array([[-2, 2, 5]])
y_true = np.array([2])

combo = Softmax_Cross_Entropy()
print(combo.forward(y_hat, y_true))
combo.backward(y_true)
print(combo.dinputs)

print("\n")
