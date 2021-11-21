
import numpy as np
X = np.array([2, 0, -7])    # 3 input features, vector
def one_neuron(X):    # multiple input features
    w = np.array([5, -2, -10])
    b = -10
    return np.sum(w * X) + b    

print(one_neuron(X))

# =========================================================
# Multiple Examples

Xs = np.array([[2, 0, -7],
              [-2, 9, 0],
              [5, 1, -1],
              [3, 0, -4]])
def one_neuron(Xs):    # multiple input features, multiple examples
    w = np.array([5, -2, -10])
    b = -10
    return np.sum(w * Xs, axis=1) + b

print(one_neuron(Xs))

# =========================================================
# Without Summation Function
Xs = np.array([[2, 0, -7],
              [-2, 9, 0],
              [5, 1, -1],
              [3, 0, -4]])
def one_neuron(Xs):    # multiple input features, multiple examples
    w = np.array([5, -2, -10])
    b = -10
    return np.dot(Xs, w) + b

print(one_neuron(Xs))
