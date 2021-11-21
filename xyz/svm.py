import numpy as np
np.random.seed(0)

print(np.round_(np.random.uniform(low=-10.0), decimals=3))
print(np.round_(np.random.randn(4), decimals=3))
print(np.round_(np.random.randn(4, 3), decimals=3))

# =========================================================

scalar = -3.963

vector = [0.742, 15.33, -2.268, 1.334]

matrix = [[-0.843, 1.97, 1.266],
          [-0.506, 2.545, 1.081],
          [0.484, 0.579, -0.18],
          [1.41, -0.374, 0.275]]

