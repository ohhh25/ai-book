print("\n")

import matplotlib.pyplot as plt    # version 3.4.0
import numpy as np    # version 1.22.2
import pandas as pd    # version 1.4.0
np.random.seed(0)    # For repeatability

class Dense_Layer:
    def __init__(self, n_inputs, n_neurons, regularization_lambda=0):
        self.weights = 0.01 * np.random.randn(n_inputs, n_neurons)
        self.biases = np.zeros([1, n_neurons])
        self.regularization_lambda = regularization_lambda

    def forward(self, inputs):    # inputs is X
        self.inputs = inputs
        self.outputs = np.dot(inputs, self.weights) + self.biases

    def backward(self, dvalues):
        self.dweights = np.dot(self.inputs.T, dvalues)
        self.dbiases = np.sum(dvalues, axis=0, keepdims=True)
        self.dinputs = np.dot(dvalues, self.weights.T)

        if self.regularization_lambda > 0:
            self.dweights += 2 * self.regularization_lambda * self.weights
            self.dbiases += 2 * self.regularization_lambda * self.biases

    def save_params(self, filename):
        np.savez(filename, weights=self.weights, biases=self.biases)

    def load_params(self, filename):
        data = np.load(filename)
        self.weights = data["weights"]
        self.biases = data["biases"]

class ReLU_Activation:
    def forward(self, inputs):    # inputs are outputs from dense layer
        self.inputs = inputs    # save inputs for backward method
        self.outputs = np.maximum(0, inputs)

    def backward(self, dvalues):    # dvalues has same shape as inputs
        self.dinputs = dvalues * (self.inputs >= 0)

class Softmax_Activation:
    def forward(self, inputs):
        exponentiated = np.exp(inputs - np.max(inputs, axis=1, keepdims=True))
        self.outputs = exponentiated / np.sum(exponentiated, axis=1, keepdims=True)

class Categorical_Cross_Entropy_Cost:
    def forward(self, y_pred, y_true):
        y_pred_clipped = np.clip(y_pred, 1e-7, 1)
        correct_class_pred = y_pred_clipped[range(len(y_true)), y_true]
        return np.mean((-np.log(correct_class_pred)))

class Softmax_Cross_Entropy:
    def __init__(self):
        self.softmax = Softmax_Activation()
        self.cce = Categorical_Cross_Entropy_Cost()

    def forward(self, inputs, y_true):
        self.softmax.forward(inputs)
        return self.cce.forward(self.softmax.outputs, y_true)

    def backward(self, y_true):
        self.dinputs = self.softmax.outputs.copy()
        self.dinputs[range(len(y_true)), y_true] -= 1
        self.dinputs = self.dinputs / len(y_true)

dataset = pd.read_csv("iris.csv")
classes = {"Iris-setosa": 0, "Iris-versicolor": 1, "Iris-virginica": 2}
dataset["class"] = dataset["class"].replace(classes)
dataset = dataset.to_numpy(dtype=float)

def standardize(X, mean=None, std=None):
    if mean is None and std is None:    # for the training set
        mean, std = np.mean(X, axis=0), np.std(X, axis=0)
    standardized = (X - mean) / std
    return standardized, (mean, std)

X, y = dataset[:, :-1], dataset[:, -1].astype(int)

class Neural_Network:
    def __init__(self, n_inputs, n_hidden, n_outputs):
        self.hidden_layer1 = Dense_Layer(n_inputs, n_hidden)
        self.activation_layer1 = ReLU_Activation()
        self.hidden_layer2 = Dense_Layer(n_hidden, n_hidden)
        self.activation_layer2 = ReLU_Activation()
        self.output_layer = Dense_Layer(n_hidden, n_outputs)
        self.combo = Softmax_Cross_Entropy()

        self.trainable_layers = [self.hidden_layer1, self.hidden_layer2, self.output_layer]

    def forward(self, inputs, y_true):
        self.hidden_layer1.forward(inputs)
        self.activation_layer1.forward(self.hidden_layer1.outputs)
        self.hidden_layer2.forward(self.activation_layer1.outputs)
        self.activation_layer2.forward(self.hidden_layer2.outputs)
        self.output_layer.forward(self.activation_layer2.outputs)
        self.cost = self.combo.forward(self.output_layer.outputs, y_true)

    def backward(self, y_true):
        self.combo.backward(y_true)
        self.output_layer.backward(self.combo.dinputs)
        self.activation_layer2.backward(self.output_layer.dinputs)
        self.hidden_layer2.backward(self.activation_layer2.dinputs)
        self.activation_layer1.backward(self.hidden_layer2.dinputs)
        self.hidden_layer1.backward(self.activation_layer1.dinputs)

def get_accuracy(y_pred, y_true):
    predicted_classes = np.argmax(y_pred, axis=1)
    return np.mean(predicted_classes == y_true) * 100


# Initialize the model and load the parameters
model = Neural_Network(4, 16, 3)    # 4 input features, 16 hidden units, 3 output features
layers = ["hidden1", "hidden2", "output"]
for layer, name in zip(model.trainable_layers, layers):
    layer.load_params("saved/" + name + ".npz")

data = np.load("saved/standardization.npz")
mean, std = data["mean"], data["std"]
X, _ = standardize(X, mean=mean, std=std)

# Check the model's accuracy
model.forward(X, y)
print(f"Accuracy: {get_accuracy(model.output_layer.outputs, y)}%")


print("\n")
