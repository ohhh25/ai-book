print("\n")

import numpy as np
np.random.seed(1)    # for repeatability

def normalize(X, min=0):
    X = (X - X.min(axis=0)) / (X.max(axis=0) - X.min(axis=0))    # normalize to [0, 1] range
    if min == -1:    # if the minimum value for the range is -1 (assuming [-1, 1] range)
        X = (2 * X) - 1
    return X

X = np.random.uniform(low=-10, high=20, size=[4, 3])
print(X) 
print(X.min(axis=0), X.max(axis=0))

X = normalize(X)
print("\n", X)
print(X.min(axis=0), X.max(axis=0))

print("\n")
