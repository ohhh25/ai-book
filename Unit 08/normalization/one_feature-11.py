print("\n")

import numpy as np
np.random.seed(0)    # for repeatability

def normalize(X, min=0):
    X = (X - X.min()) / (X.max() - X.min())    # normalize to [0, 1] range
    if min == -1:    # if the minimum value for the range is -1 (assuming [-1, 1] range)
        X = (2 * X) - 1
    return X

X = np.random.uniform(low=-20, high=10, size=[4, 1])
print(X)

X = normalize(X)
print("\n[0, 1] range:")
print(X)

X = normalize(X, min=-1)
print("\n[-1, 1] range:")
print(X)

print("\n")
