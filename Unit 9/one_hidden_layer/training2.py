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

    def backward(self, dvalues):
        self.dweights = np.dot(self.inputs.T, dvalues)
        self.dbiases = np.sum(dvalues, axis=0, keepdims=True)
        self.dinputs = np.dot(dvalues, self.weights.T)

class ReLU_Activation:
    def forward(self, inputs):    # inputs are outputs from dense layer
        self.inputs = inputs    # save inputs for backward method
        self.outputs = np.maximum(0, inputs)

    def backward(self, dvalues):    # dvalues has same shape as inputs
        self.dinputs = dvalues * (self.inputs >= 0)

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

X = np.expand_dims(np.array(range(len(modified))), axis=1)
y = modified.copy()

def normalize(X, min=0):
    X = (X - X.min(axis=0)) / (X.max(axis=0) - X.min(axis=0))    # normalize to [0, 1] range
    if min == -1:    # if the minimum value for the range is -1 (assuming [-1, 1] range)
        X = (2 * X) - 1
    return X

X = normalize(X, min=-1)    # normalize data to [-1, 1] range

class Neural_Network:
    def __init__(self, n_inputs, n_hidden, n_outputs):    # n_hidden is the number of hidden neurons
        self.hidden_layer = Dense_Layer(n_inputs, n_hidden)
        self.activation = ReLU_Activation()
        self.output_layer = Dense_Layer(n_hidden, n_outputs)
        self.cost_function = MSE_Cost()

        self.trainable_layers = [self.hidden_layer, self.output_layer]

    def forward(self, inputs, y_true):
        self.hidden_layer.forward(inputs)
        self.activation.forward(self.hidden_layer.outputs)
        self.output_layer.forward(self.activation.outputs)
        self.cost = self.cost_function.forward(self.output_layer.outputs, y_true)

    def backward(self, y_true):
        self.cost_function.backward(self.output_layer.outputs, y_true)
        self.output_layer.backward(self.cost_function.dinputs)
        self.activation.backward(self.output_layer.dinputs)
        self.hidden_layer.backward(self.activation.dinputs)
        

model = Neural_Network(1, 4, 2)    # 1 input feature, 4 hidden units, 2 output features
optimizer = SGD_Optimizer(learning_rate=0.01, mu=0.5)   # learning rate is 0.001, momentum of 0.5

model.forward(X, y)
print("Initial Cost:", model.cost)

cost_history = []    # append to this in training loop
for epochs in range(40):
    # forward pass
    model.forward(X, y)
    cost_history.append(model.cost)

    # check for dead neurons
    if epochs % 4 == 0:
        n = sum((model.hidden_layer.outputs < 0).all(axis=0))    # number of dead neurons
        percentage = (n / model.hidden_layer.biases.size) * 100    # percentage of dead neurons
        print(f"Cost: {model.cost} - Percentage of Dead neurons: {percentage}%")

    # backward pass
    model.backward(y)
    for layer in model.trainable_layers:
        optimizer.update_params(layer)

# Check New Cost
model.forward(X, y)
cost_history.append(model.cost)
print("Final Cost:", model.cost)

# Model Predictions vs. Correct Answer Graph

fig = plt.figure()    # create a graphing space
# plot correct values
plt.plot(y[:, 0], label="TMAX")
plt.plot(y[:, 1], label="TMIN")

# plot predicted values
plt.plot(model.output_layer.outputs[:, 0], label="ŷ -- TMAX")
plt.plot(model.output_layer.outputs[:, 1], label="ŷ -- TMIN")

# customization
plt.xlabel("MODIFIED day of the year")
plt.ylabel("temperature (°C)")
plt.legend()

# Save and Close Graph
fig.savefig("training2.png")
plt.close()

# Cost History Graph
fig = plt.figure()
plt.plot(cost_history)
plt.xlabel("Epochs")    
plt.ylabel("Cost")
fig.savefig("cost2.png")
plt.close()

print("\n")
