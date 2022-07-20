print("\n")

import numpy as np    # version 1.22.2
import pandas as pd    # version 1.4.0
np.random.seed(0)    # For repeatability

dataset = pd.read_excel("Pistachio_16_Features_Dataset.xlsx", )
to_drop = ['ECCENTRICITY', 'EQDIASQ', 'SOLIDITY', 'CONVEX_AREA', 
           'EXTENT', 'COMPACTNESS', 'SHAPEFACTOR_3', 'SHAPEFACTOR_4']

dataset = dataset.drop(to_drop, axis=1)

# Convert the Classes to 0s and 1s
classes = {"Kirmizi_Pistachio": 0, "Siit_Pistachio": 1}
dataset['Class'] = dataset['Class'].replace(classes)

# Split the dataset into X and y
dataset = dataset.to_numpy(dtype=float)
X, y = dataset[:, 0:-1], dataset[:, -1].astype(int)
y = np.expand_dims(y, axis=-1)
print(X.shape, y.shape)

print("\n")
