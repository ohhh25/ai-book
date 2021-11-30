import numpy as np
np.random.seed(3)

print(np.random.randint(low=-10, high=10, size=2))   # remaining X
print(np.random.randint(low=-10, high=10, size=2))    # remaining w

# =========================================================
print(np.random.randint(low=-10, high=10, size=[3, 2]))    # remaining Xs
