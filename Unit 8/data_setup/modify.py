print("\n")

import matplotlib.pyplot as plt    # version 3.4.0
import numpy as np    # version 1.22.2
import pandas as pd    # version 1.4.0

dataset = pd.read_csv("2639600.csv")
dataset = dataset.drop(labels=['STATION', 'NAME', 'DATE'], axis=1)
datset = dataset.to_numpy()

LENGTH_OF_MOVING_AVERAGE = 125
modified = np.empty([(len(datset)-LENGTH_OF_MOVING_AVERAGE)+1, 2])
for i in range(len(modified)):
    modified[i] = np.mean(dataset[i:i+125], axis=0)

fig = plt.figure()    # create a graphing space
plt.plot(modified[:, 0], label="TMAX")    # plotting maximum temperatures on graphing sapce
plt.plot(modified[:, 1], label="TMIN")    # plotting minimum temperatures on graphing space
plt.legend()    # show legend to differentiate colored lines

plt.ylabel("temperature (°C)")
plt.legend()

fig.savefig("modified.png")
plt.close()

print("\n")
