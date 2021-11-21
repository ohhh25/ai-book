import numpy as np
np.random.seed(0)

print(np.random.randint(low=-10, high=10))    # X
print(np.random.randint(low=-10, high=10))    # w
print(np.random.randint(low=-10, high=10))    # b

# =========================================================
np.random.seed(2)
print(np.random.randint(low=-10, high=10, size=3))    # remaining Xs
