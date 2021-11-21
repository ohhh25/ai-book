import numpy as np
np.random.seed(0)

print(np.round_(np.random.uniform(low=-10.0, high=10.0), decimals=3))
print(np.round_(np.random.uniform(low=-10.0, high=10.0, size=4), decimals=3))
print(np.round_(np.random.uniform(low=-10.0, high=10.0, size=[4, 3]), decimals=3))

# =========================================================

scalar = 0.976

vector = [4.304, 2.055, 0.898, -1.527]

matrix = [[2.918, -1.248,  7.835],
          [9.273, -2.331, 5.835],
          [0.578, 1.361, 8.512],
          [-8.579, -8.257, -9.596]]

