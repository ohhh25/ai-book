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

class Sigmoid_Activation:
    def forward(self, inputs):    # inputs are outputs from dense layer
        self.outputs = 1 / (1 + np.exp(-inputs))
        
    def backward(self, dvalues):    # dvalues has same shape as inputs
        self.dinputs = dvalues * (self.outputs * (1 - self.outputs))

class BCE_Cost:
    def forward(self, y_pred, y_true):
        y_pred_clipped = np.clip(y_pred, 1e-7, 1 - 1e-7)
        return np.mean((y_true * -np.log(y_pred_clipped)) + \
            ((1 - y_true) * -np.log(1 - y_pred_clipped)))

    def backward(self, y_pred, y_true):
        y_pred_clipped = np.clip(y_pred, 1e-7, 1 - 1e-7)
        self.dinputs = -(y_true / y_pred_clipped) + \
            ((1 - y_true) / (1 - y_pred_clipped))

        self.dinputs /= len(y_pred_clipped)

class RMSProp_Optimizer:
    def __init__(self, learning_rate, rho=0.9, eps=1e-7):    # rho is decay rate
        self.lr = learning_rate
        self.rho = rho
        self.eps = eps
    
    def update_params(self, layer):    # dense layer
        # if layer does not have the attribute "cache_weights",
        # meaning that the layer also does not have the attribute "cache_biases",
        # so let's initialize those attributes with cache as 0
        if hasattr(layer, "cache_weights") == False:
            layer.cache_weights = np.zeros_like(layer.weights)
            layer.cache_biases = np.zeros_like(layer.biases)
        
        layer.cache_weights = (layer.cache_weights * self.rho) + ((1 - self.rho) * layer.dweights ** 2)
        layer.cache_biases = (layer.cache_biases * self.rho) + ((1 - self.rho) * layer.dbiases ** 2)

        layer.weights += (self.lr / (np.sqrt(layer.cache_weights) + self.eps)) * -layer.dweights
        layer.biases += (self.lr / (np.sqrt(layer.cache_biases) + self.eps)) * -layer.dbiases

class Learning_Rate_Decayer:
    def __init__(self, optimizer, decay_factor):
        self.epochs = 0
        self.optimizer = optimizer
        self.initial_lr = optimizer.lr
        self.decay_factor = decay_factor

    def update_learning_rate(self):
        self.epochs += 1
        self.optimizer.lr = self.initial_lr / (1 + (self.decay_factor * self.epochs))


dataset = pd.read_excel("Pistachio_16_Features_Dataset.xlsx")
to_drop = ['ECCENTRICITY', 'EQDIASQ', 'SOLIDITY', 'CONVEX_AREA', 
           'EXTENT', 'COMPACTNESS', 'SHAPEFACTOR_3', 'SHAPEFACTOR_4']

dataset = dataset.drop(to_drop, axis=1)

# Convert the Classes to 0s and 1s
classes = {"Kirmizi_Pistachio": 0, "Siit_Pistachio": 1}
dataset['Class'] = dataset['Class'].replace(classes)
dataset = dataset.to_numpy(dtype=float)
np.random.shuffle(dataset)

train = int(len(dataset) * 0.8)    # use 80% of the dataset for training
val = int(len(dataset) * 0.1)    # use 10% of the dataset for validation

# Split the dataset into training, validation and test sets
X_train, y_train = dataset[0:train, 0:-1], dataset[0:train, -1].astype(int)
X_val, y_val = dataset[train:train + val, 0:-1], dataset[train:train + val, -1].astype(int)
X_test, y_test = dataset[train + val:, 0:-1], dataset[train + val:, -1].astype(int)

y_train = np.expand_dims(y_train, axis=-1)
y_val = np.expand_dims(y_val, axis=-1)
y_test = np.expand_dims(y_test, axis=-1)

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
    def __init__(self, n_inputs, n_hidden, n_outputs):    # n_hidden is the number of hidden neurons
        self.hidden_layer1 = Dense_Layer(n_inputs, n_hidden, regularization_lambda=0.01)
        self.activation1 = ReLU_Activation()
        self.hidden_layer2 = Dense_Layer(n_hidden, n_hidden, regularization_lambda=0.01)
        self.activation2 = ReLU_Activation()
        self.hidden_layer3 = Dense_Layer(n_hidden, n_hidden,  regularization_lambda=0.01)
        self.activation3 = ReLU_Activation()
        self.output_layer = Dense_Layer(n_hidden, n_outputs, regularization_lambda=0.01)
        self.output_activation = Sigmoid_Activation()
        self.cost_function = BCE_Cost()

        self.trainable_layers = [self.hidden_layer1, self.hidden_layer2,
                                 self.hidden_layer3, self.output_layer]

    def forward(self, inputs, y_true):
        self.hidden_layer1.forward(inputs)
        self.activation1.forward(self.hidden_layer1.outputs)
        self.hidden_layer2.forward(self.activation1.outputs)
        self.activation2.forward(self.hidden_layer2.outputs)
        self.hidden_layer3.forward(self.activation2.outputs)
        self.activation3.forward(self.hidden_layer3.outputs)
        self.output_layer.forward(self.activation3.outputs)
        self.output_activation.forward(self.output_layer.outputs)
        self.cost = self.cost_function.forward(self.output_activation.outputs, y_true)

    def backward(self, y_true):
        self.cost_function.backward(self.output_activation.outputs, y_true)
        self.output_activation.backward(self.cost_function.dinputs)
        self.output_layer.backward(self.output_activation.dinputs)
        self.activation3.backward(self.output_layer.dinputs)
        self.hidden_layer3.backward(self.activation3.dinputs)
        self.activation2.backward(self.hidden_layer3.dinputs)
        self.hidden_layer2.backward(self.activation2.dinputs)
        self.activation1.backward(self.hidden_layer2.dinputs)
        self.hidden_layer1.backward(self.activation1.dinputs)


def get_accuracy(y_pred, y_true):
    predicted_classes = y_pred.copy()
    predicted_classes[y_pred < 0.5] = 0
    predicted_classes[y_pred >= 0.5] = 1
    return np.mean(predicted_classes == y_true) * 100

def get_regularization_penalty(layers, regularization_lambda):
    penalty = 0
    for layer in layers:
        penalty += np.sum(np.square(layer.weights))
        penalty += np.sum(np.square(layer.biases))
    return regularization_lambda * penalty

# Initialize the model and the optimizer
model = Neural_Network(8, 16, 1)     # 8 input features, 16 hidden units, 1 output feature
optimizer = RMSProp_Optimizer(0.01)    # learning rate of 0.01
decayer = Learning_Rate_Decayer(optimizer, 0.02)
cost_history, accuracy_history = [], []
val_cost, val_accuracy = [], []

for i in range(400):
    # Get Validation Cost and Accuracy
    model.forward(X_val, y_val)
    val_cost.append(model.cost)
    val_accuracy.append(get_accuracy(model.output_activation.outputs, y_val))

    model.forward(X_train, y_train)
    regularization_penalty = get_regularization_penalty(model.trainable_layers, 0.01)
    cost_history.append(model.cost + regularization_penalty)
    accuracy_history.append(get_accuracy(model.output_activation.outputs, y_train))

    if i % 40 == 0:
        print(f"Cost: {cost_history[-1]} - Accuracy: {accuracy_history[-1]}%"+
              f" - Validation Cost: {val_cost[-1]} - Validation Accuracy: {val_accuracy[-1]}%")

    model.backward(y_train)
    for layer in model.trainable_layers:
        optimizer.update_params(layer)
    
    decayer.update_learning_rate()


# Check Ending Cost and Accuracy
model.forward(X_train, y_train)
regularization_penalty = get_regularization_penalty(model.trainable_layers, 0.01)
cost_history.append(model.cost + regularization_penalty)
accuracy_history.append(get_accuracy(model.output_activation.outputs, y_train))
print(f"Final Cost: {cost_history[-1]} - Final Accuracy: {accuracy_history[-1]}%")

model.forward(X_val, y_val)
val_cost.append(model.cost)
val_accuracy.append(get_accuracy(model.output_activation.outputs, y_val))
print(f"Final Val Cost: {val_cost[-1]} - Final Val Accuracy: {val_accuracy[-1]}%")

# Test the model on the test set
model.forward(X_test, y_test)
accuracy = get_accuracy(model.output_activation.outputs, y_test)
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
