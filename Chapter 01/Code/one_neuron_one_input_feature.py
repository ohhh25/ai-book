print("\n")

X = 2
def one_neuron(X):    # one input feature
    w = 5
    b = -10
    return (w * X) + b

print(one_neuron(X))

# =========================================================
# Multiple Examples

person1_X = 2
person2_X = -2
person3_X = 5
person4_X = 3
def one_neuron(X):    # one input feature
    w = 5
    b = -10
    return (w * X) + b

print(one_neuron(person1_X))
print(one_neuron(person2_X))
print(one_neuron(person3_X))
print(one_neuron(person4_X))

# =========================================================
# Multiple Examples (Vectorized)

import numpy as np
Xs = np.array([2, -2, 5, 3])    # 4 examples, vector
def one_neuron(Xs):    # one input feature, multiple examples
    w = 5
    b = -10
    return (w * Xs) + b

print(one_neuron(Xs))

print("\n")
