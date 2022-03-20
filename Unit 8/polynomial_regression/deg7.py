print("\n")

import matplotlib.pyplot as plt    # version 3.4.0
import numpy as np    # version 1.22.2
import pandas as pd    # version 1.4.0
np.random.seed(0)    # For repeatability

class Dense_Layer:
    def __init__(self, n_inputs, n_neurons):
        self.weights = 0.01 * np.random.randn(n_inputs, n_neurons)
        self.biases = np.zeros([1, n_neurons])

    def forward(self, inputs):    # inputs is X
        self.inputs = inputs
        self.outputs = np.dot(inputs, self.weights) + self.biases

    def backward(self, dvalues):    # dvalues is .dinputs from cost function
        self.dweights = np.dot(self.inputs.T, dvalues)
        self.dbiases = np.sum(dvalues, axis=0, keepdims=True)

class MSE_Cost:
    def forward(self, y_pred, y_true):
        return np.mean(np.square(y_pred - y_true))

    def backward(self, y_pred, y_true):
        self.dinputs = (2 / y_pred.size) * (y_pred - y_true)

class SGD_Optimizer:
    def __init__(self, learning_rate, mu=0):    # mu is coefficient of friction 
        self.lr = learning_rate
        self.mu = mu
    
    def update_params(self, layer):    # dense layer
        # if layer does not have the attribute "v_weights",
        # meaning that the layer also does not have the attribute "v_biases",
        # so let's initialize those attributes with velocities as 0
        if hasattr(layer, "v_weights") == False:
            layer.v_weights = np.zeros_like(layer.weights)
            layer.v_biases = np.zeros_like(layer.biases)
        
        # new velocity is equal to the product of the old velocity and the coefficent of friction
        # plus the product of the new gradient (acceleration)
        layer.v_weights = (self.mu * layer.v_weights) + (self.lr * -layer.dweights)
        layer.v_biases = (self.mu * layer.v_biases) + (self.lr * -layer.dbiases)

        layer.weights += layer.v_weights
        layer.biases += layer.v_biases


dataset = pd.read_csv("2639600.csv")
dataset = dataset.drop(labels=['STATION', 'NAME', 'DATE'], axis=1)
datset = dataset.to_numpy()

LENGTH_OF_MOVING_AVERAGE = 125
modified = np.empty([(len(datset)-LENGTH_OF_MOVING_AVERAGE)+1, 2])
for i in range(len(modified)):
    modified[i] = np.mean(dataset[i:i+125], axis=0)

X = np.array(range(len(modified)))
y = modified.copy()

# add in x^2, x^3, x^4, x^5, x^6, and x^7 input feature
DEGREE = 7    # degree  of polynomial
Xs = np.empty_like(X, shape=[len(X), DEGREE])
for power in range(1, DEGREE+1):
    Xs[:, power-1] = np.power(X, power)
X = Xs.copy()

def normalize(X, min=0):
    X = (X - X.min(axis=0)) / (X.max(axis=0) - X.min(axis=0))    # normalize to [0, 1] range
    if min == -1:    # if the minimum value for the range is -1 (assuming [-1, 1] range)
        X = (2 * X) - 1
    return X

X = normalize(X, min=-1)    # normalize data to [-1, 1] range

model = Dense_Layer(7, 2)    # seven input features, 2 neurons (output features)
mse = MSE_Cost()    # define cost function
optimizer = SGD_Optimizer(0.1, mu=0.9)    # learning rate of 0.1, momentum of 0.9

model.forward(X)
print("Initial Cost:", mse.forward(model.outputs, y))

for epochs in range(200):
    # forward pass
    model.forward(X)
    cost = mse.forward(model.outputs, y)
    if epochs % 20 == 0:
        print(cost)

    # backward pass
    mse.backward(model.outputs, y)
    model.backward(mse.dinputs)
    optimizer.update_params(model)

# Check New Cost
model.forward(X)
cost = mse.forward(model.outputs, y)
print("Final Cost:", cost)

fig = plt.figure()    # create a graphing space
# plot correct values
plt.plot(y[:, 0], label="TMAX")
plt.plot(y[:, 1], label="TMIN")

# plot predicted values
plt.plot(model.outputs[:, 0], label="ŷ -- TMAX")
plt.plot(model.outputs[:, 1], label="ŷ -- TMIN")

# customization
plt.xlabel("MODIFIED day of the year")
plt.ylabel("temperature (°C)")
plt.legend()

# Save and Close Graph
fig.savefig("deg7.png")
plt.close()

print("\n")
