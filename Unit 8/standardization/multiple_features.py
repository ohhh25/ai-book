print("\n")

import numpy as np
np.random.seed(1)    # for repeatability

def standarize(X):
    mean_of_zero = X - np.mean(X, axis=0)    # bringing data to have mean of 0
    X = mean_of_zero / np.std(X, axis=0)    # brinding data to have standard deviation of 1
    return X

X = np.random.uniform(low=-10, high=20, size=[4, 3])
print(X) 
print(np.mean(X, axis=0), np.std(X, axis=0))

X = standarize(X)
print("\n", X)
print(np.mean(X, axis=0), np.std(X, axis=0))
print(X.min(axis=0), X.max(axis=0))

print("\n")
