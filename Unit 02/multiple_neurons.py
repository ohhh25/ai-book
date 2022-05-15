print("\n")

import numpy as np
X = np.array([2, 0, -7])    # 3 input features, vector
def multiple_neurons(X):    # multiple input features
    w = np.array([[5, -2, -10],    # neuron 1's weights
                  [0, 9, -5]])    # neuron 2's weights
    b = np.array([-10, 7])    # 2 biases, one for each neuron, vector
    return np.dot(w, X.T) + b    

print(multiple_neurons(X))

# =========================================================
# Multiple Examples

Xs = np.array([[2, 0, -7],
              [-2, 9, 0],
              [5, 1, -1],
              [3, 0, -4]])

def multiple_neurons(Xs):    # multiple input features, multiple examples
    w = np.array([[5, -2, -10],    # neuron 1's weights
                  [0, 9, -5]])    # neuron 2's weights
    b = np.array([-10, 7])    # 2 biases, one for each neuron, vector
    return np.dot(Xs, w.T) + b  

print(multiple_neurons(Xs))

print("\n")
