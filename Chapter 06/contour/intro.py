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
    def __init__(self, learning_rate, mu=0):    # mu is coefficient of friction 
        self.lr = learning_rate
        self.mu = mu
    
    def update_params(self, layer):    # dense layer
        # if layer does not have the attribute "v_weights",
        # meaning that the layer also does not have the attribute "v_biases",
        # we will give the let's initialize those attributes with velocities as 0
        if hasattr(layer, "v_weights") == False:
            layer.v_weights = np.zeros_like(layer.weights)
            layer.v_biases = np.zeros_like(layer.biases)
        
        # new velocity is equal to the product of the old velocity and the coefficent of friction
        # plus the product of the new gradient (acceleration)
        layer.v_weights = (self.mu * layer.v_weights) + (self.lr * -layer.dweights)
        layer.v_biases = (self.mu * layer.v_biases) + (self.lr * -layer.dbiases)

        layer.weights += layer.v_weights
        layer.biases += layer.v_biases


# hours studied
X = np.array([[0], [1], [2], [3], [4], [5], [6], [7], [8]])    # one input feature for each example
# percentage
y = np.array([[20], [30], [40], [50], [60], [70], [80], [90], [100]])    # one output feature for each example

mse = MSE_Cost()    # define cost function
model = Dense_Layer(1, 1)    # 1 input feature, 1 neuron (output feature)
optimizer = SGD_Optimizer(0.001, mu=0.95)    # learning rate of 0.001 and coefficent of friction of 0.95

model.forward(X)
print(model.weights, model.biases)
cost_history = []    # append to this list in the loop
weight_history, bias_history = [], []    # append in loop
print("Original Cost:", mse.forward(model.outputs, y))

for epochs in range(200):
    # forward pass
    model.forward(X)
    weight_history.append(float(model.weights))
    bias_history.append(float(model.biases))
    cost_history.append(mse.forward(model.outputs, y))

    # backward pass
    mse.backward(model.outputs, y)
    model.backward(mse.dinputs)

    optimizer.update_params(model)    # update

# Check New Cost
model.forward(X)
weight_history.append(float(model.weights))
bias_history.append(float(model.biases))
print(model.weights, model.biases)
cost_history.append(mse.forward(model.outputs, y))
print("New Cost:", cost_history[-1])

def graph(data, xlabel, ylabel, fname):
    # Initialize Graph
    fig = plt.figure()    # create a graphing space
    plt.plot(data)     # plot on graphing space

    # Label Graph
    plt.xlabel(xlabel)    
    plt.ylabel(ylabel)

    # Save and Close Graph
    fig.savefig(fname)
    plt.close()

graph(weight_history, "Epochs", "Weight Value", "weight.png")
graph(bias_history, "Epochs", "Bias Value", "bias.png")
graph(cost_history, "Epochs", "Cost", "cost.png")

print("\n")
