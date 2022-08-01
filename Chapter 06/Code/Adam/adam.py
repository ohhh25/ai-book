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

class Adam_Optimizer:
    def __init__(self, learning_rate, beta1=0.9, beta2=0.999, eps=1e-8):    # rho is decay rate
        self.epochs = 0
        self.lr = learning_rate
        self.beta1 = beta1    # beta1 is coefficient of friction for momentum
        self.beta2 = beta2    # beta2 is decay rate for RMSProp
        self.eps = eps     # epsilon
    
    def update_params(self, layer):    # dense layer
        self.epochs += 1
        # if layer does not have the attribute "v_weights", the layer also does not have
        # the attributes "v_biases", "cache_weights", and "cache_biases"
        # we will give the let's initialize those attributes with cache as 0
        if hasattr(layer, "v_weights") == False:
            layer.v_weights = np.zeros_like(layer.weights)
            layer.v_biases = np.zeros_like(layer.biases)
            layer.cache_weights = np.zeros_like(layer.weights)
            layer.cache_biases = np.zeros_like(layer.biases)
        
        # velocities
        layer.v_weights = (layer.v_weights * self.beta1) + ((1 - self.beta1) * layer.dweights * 2)
        layer.v_biases = (layer.v_biases * self.beta1) + ((1 - self.beta1) * layer.dbiases * 2)

        # velocity corrections
        layer.v_weights_corrected = layer.v_weights / (1 - (self.beta1 ** self.epochs))
        layer.v_biases_corrected = layer.v_biases / (1 - (self.beta1 ** self.epochs))

        # caches
        layer.cache_weights = (layer.cache_weights * self.beta2) + ((1 - self.beta2) * layer.dweights ** 2)
        layer.cache_biases = (layer.cache_biases * self.beta2) + ((1 - self.beta2) * layer.dbiases ** 2)

        # cache corrections
        layer.cache_weights_corrected = layer.cache_weights / (1 - (self.beta2 ** self.epochs))
        layer.cache_biases_corrected = layer.cache_biases / (1 - (self.beta2 ** self.epochs))

        # update
        layer.weights += (self.lr / (np.sqrt(layer.cache_weights_corrected) + self.eps)) * -layer.v_weights_corrected
        layer.biases += (self.lr / (np.sqrt(layer.cache_biases_corrected) + self.eps)) * -layer.v_biases_corrected


class Learning_Rate_Decayer:
    def __init__(self, optimizer, decay_factor):
        self.epochs = 0
        self.optimizer = optimizer
        self.initial_lr = optimizer.lr
        self.decay_factor = decay_factor

    def update_learning_rate(self):
        self.epochs += 1
        self.optimizer.lr = self.initial_lr / (1 + (self.decay_factor * self.epochs))


# hours studied
X = np.array([[0], [1], [2], [3], [4], [5], [6], [7], [8]])    # one input feature for each example
# percentage
y = np.array([[20], [30], [40], [50], [60], [70], [80], [90], [100]])    # one output feature for each example

mse = MSE_Cost()    # define cost function
model = Dense_Layer(1, 1)    # 1 input feature, 1 neuron (output feature)
optimizer = Adam_Optimizer(0.1)    # learning rate of 0.1

model.forward(X)
print(model.weights, model.biases)
cost_history = []    # append to this list in the loop
weight_history, bias_history = [], []    # append in loop
print("Original Cost:", mse.forward(model.outputs, y))

for epochs in range(400):
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
