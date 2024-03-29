print("\n")

import numpy as np
import matplotlib.pyplot as plt
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

class MAE_Cost:
    def forward(self, y_pred, y_true):    # y_pred is prediction from model, y_true is the answer
        return np.mean(np.abs(y_pred - y_true))

    def backward(self, y_pred, y_true):
        self.dinputs = np.sign(y_pred - y_true) / y_pred.size

class MSE_Cost:
    def forward(self, y_pred, y_true):
        return np.mean(np.square(y_pred - y_true))

    def backward(self, y_pred, y_true):
        self.dinputs = (2 / y_pred.size) * (y_pred - y_true)

class SGD_Optimizer:
    def __init__(self, learning_rate):
        self.lr = learning_rate
    
    def update_params(self, layer):    # dense layer
        layer.weights += self.lr * -layer.dweights
        layer.biases += self.lr * -layer.dbiases

# hours studied
X = np.array([[0], [1], [2], [3], [4], [5], [6], [7], [8]])    # one input feature for each example
# percentage
y = np.array([[20], [30], [40], [50], [60], [70], [80], [90], [100]])    # one output feature for each example

mse = MSE_Cost()    # define cost function
model = Dense_Layer(1, 1)    # 1 input feature, 1 neuron (output feature)
optimizer = SGD_Optimizer(0.01)    # learning rate of 0.01

model.forward(X)
print(model.weights, model.biases)
cost_history = []    # append to this list in the loop
print("Original Cost:", mse.forward(model.outputs, y))

for epochs in range(1000):
    # forward pass
    model.forward(X)
    cost_history.append(mse.forward(model.outputs, y))

    # backward pass
    mse.backward(model.outputs, y)
    model.backward(mse.dinputs)

    optimizer.update_params(model)    # update

# Check New Cost
model.forward(X)
print(model.weights, model.biases)
cost_history.append(mse.forward(model.outputs, y))
print("New Cost:", cost_history[-1])

# Initialize Graph
fig = plt.figure()    # create a graphing space
plt.plot(cost_history)     # plot on graphing space

# Label Graph
plt.xlabel("Epochs")    
plt.ylabel("Cost")

# Save and Close Graph
fig.savefig("history.png")
plt.close()

print("\n")
