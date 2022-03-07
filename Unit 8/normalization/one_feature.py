print("\n")

import numpy as np
np.random.seed(0)    # for repeatability

def normalize(X):
    return (X - X.min()) / (X.max() - X.min())

X = np.random.uniform(low=-20, high=10, size=[4, 1])
print(X)

X = normalize(X)
print("\n")
print(X)

print("\n")
