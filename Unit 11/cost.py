print("\n")

import numpy as np    # version 1.22.2
np.random.seed(0)    # For repeatability

class Softmax_Activation:
    def forward(self, inputs):
        exponentiated = np.exp(inputs - np.max(inputs, axis=1, keepdims=True))
        self.output = exponentiated / np.sum(exponentiated, axis=1, keepdims=True)

class Categorical_Cross_Entropy_Cost:
    def forward(self, y_pred, y_true):
        y_pred_clipped = np.clip(y_pred, 1e-7, 1)
        correct_class_pred = y_pred_clipped[range(len(y_true)), y_true]
        return np.mean(-np.log(correct_class_pred))

y_hat = np.array([[-2, 2, 5]])
y_true = np.array([2])

softmax = Softmax_Activation()
cce = Categorical_Cross_Entropy_Cost()

softmax.forward(y_hat)
print(cce.forward(softmax.output, y_true))


print("\n")
