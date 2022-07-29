print("\n")

X = 2
def one_neuron(X):    # one input feature
    w = 5
    b = -10
    return (w * X) + b

print(one_neuron(X))

# =========================================================
# Multiple Examples

import numpy as np
Xs = np.array([2, -2, 5, 3])    # 4 examples, vector
def one_neuron(Xs):    # one input feature, multiple examples
    w = 5
    b = -10
    return (w * Xs) + b

print(one_neuron(Xs))

print("\n")
