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

class SGD_Optimizer:
    def __init__(self, learning_rate, mu=0):
        self.lr = learning_rate    # lr is short for learning rate
        self.mu = mu    # friction


    def update_params(self, layer):
        if hasattr(layer, "v_weights") == False:
            layer.v_weights = np.zeros_like(layer.weights)
            layer.v_biases = np.zeros_like(layer.biases)

        # Velocities
        layer.v_weights = (self.mu * layer.v_weights) + (-layer.dweights * self.lr)
        layer.v_biases = (self.mu * layer.v_biases) + (-layer.dbiases * self.lr)
        
        # add weighted sum of gradients to update
        layer.weights += layer.v_weights
        layer.biases += layer.v_biases

dataset = pd.read_csv("iris.csv")
classes = {"Iris-setosa": 0, "Iris-versicolor": 1, "Iris-virginica": 2}
dataset["class"] = dataset["class"].replace(classes)
dataset = dataset.to_numpy(dtype=float)
np.random.shuffle(dataset)

train = int(len(dataset) * 0.8)    # use 80% of the dataset for training
val = int(len(dataset) * 0.1)    # use 10% of the dataset for validation

# Split the dataset into training, validation and test sets
X_train, y_train = dataset[0:train, 0:-1], dataset[0:train, -1].astype(int)
X_val, y_val = dataset[train:train + val, 0:-1], dataset[train:train + val, -1].astype(int)
X_test, y_test = dataset[train + val:, 0:-1], dataset[train + val:, -1].astype(int)

print(len(X_train), len(X_val), len(X_test))    # number of examples in each set

def standardize(X, mean=None, std=None):
    if mean is None and std is None:    # for the training set
        mean, std = np.mean(X, axis=0), np.std(X, axis=0)
    standardized = (X - mean) / std
    return standardized, (mean, std)

X_train, (mean, std) = standardize(X_train)
X_val, _ = standardize(X_val, mean, std)
X_test, _ = standardize(X_test, mean, std)

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


# Initialize the model and the optimizer
model = Neural_Network(4, 16, 3)    # 4 input features, 16 hidden units, 3 output features
optimizer = SGD_Optimizer(0.1)
cost_history, accuracy_history = [], []
val_cost, val_accuracy = [], []
batch_size = 16

for epoch in range(200):
    model.forward(X_train, y_train)
    cost_history.append(model.cost)
    accuracy_history.append(get_accuracy(model.combo.softmax.outputs, y_train))

    # Validate the model
    model.forward(X_val, y_val)
    val_cost.append(model.cost)
    val_accuracy.append(get_accuracy(model.combo.softmax.outputs, y_val))

    if epoch % 20 == 0:
        print(f"Cost: {cost_history[-1]} - Accuracy: {accuracy_history[-1]}%"+
              f" - Validation Cost: {val_cost[-1]} - Validation Accuracy: {val_accuracy[-1]}%")

    for i in range(0, len(X_train), batch_size):
        X_batch = X_train[i:i + batch_size]
        y_batch = y_train[i:i + batch_size]

        model.forward(X_batch, y_batch)
        model.backward(y_batch)

        for layer in model.trainable_layers:
            optimizer.update_params(layer)


# Check Ending Cost and Accuracy
model.forward(X_train, y_train)
cost_history.append(model.cost)
accuracy_history.append(get_accuracy(model.combo.softmax.outputs, y_train))
print(f"Final Cost: {cost_history[-1]} - Final Accuracy: {accuracy_history[-1]}%")

model.forward(X_val, y_val)
val_cost.append(model.cost)
val_accuracy.append(get_accuracy(model.combo.softmax.outputs, y_val))
print(f"Final Val Cost: {val_cost[-1]} - Final Val Accuracy: {val_accuracy[-1]}%")

# Test the model on the test set
model.forward(X_test, y_test)
accuracy = get_accuracy(model.combo.softmax.outputs, y_test)
print(f"Test Set Results ~ Cost: {model.cost} - Accuracy: {accuracy}%")

# Cost History Graph
fig = plt.figure()
plt.plot(cost_history, label="Training Cost")
plt.plot(val_cost, label="Validation Cost")
plt.xlabel("Epochs")    
plt.ylabel("Cost")
plt.legend()
fig.savefig("cost.png")
plt.close()

# Accuracy History Graph
fig = plt.figure()
plt.plot(accuracy_history, label="Training Accuracy")
plt.plot(val_accuracy, label="Validation Accuracy")
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.legend()
fig.savefig("accuracy.png")
plt.close()    


print("\n")
