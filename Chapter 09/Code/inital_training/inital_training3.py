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

dataset = pd.read_excel("Pistachio_16_Features_Dataset.xlsx")
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

def standardize(X):
    mean_of_zero = X - np.mean(X, axis=0)    # bringing data to have mean of 0
    X = mean_of_zero / np.std(X, axis=0)    # brinding data to have standard deviation of 1
    return X

X = standardize(X)    # standardize the data

class Neural_Network:
    def __init__(self, n_inputs, n_hidden, n_outputs):    # n_hidden is the number of hidden neurons
        self.hidden_layer1 = Dense_Layer(n_inputs, n_hidden)
        self.activation1 = ReLU_Activation()
        self.hidden_layer2 = Dense_Layer(n_hidden, n_hidden)
        self.activation2 = ReLU_Activation()
        self.hidden_layer3 = Dense_Layer(n_hidden, n_hidden)
        self.activation3 = ReLU_Activation()
        self.output_layer = Dense_Layer(n_hidden, n_outputs)
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

# Initialize the model and the optimizer
model = Neural_Network(8, 16, 1)     # 8 input features, 16 hidden units, 1 output feature
optimizer = SGD_Optimizer(0.1, mu=0.9)    # learning rate is 0.1, mu is 0.9
cost_history, accuracy_history = [], []

for i in range(100):
    model.forward(X, y)
    cost_history.append(model.cost)
    accuracy_history.append(get_accuracy(model.output_activation.outputs, y))

    if i % 10 == 0:
        print(f"Cost: {cost_history[-1]} - Accuracy: {accuracy_history[-1]}%")

    model.backward(y)
    for layer in model.trainable_layers:
        optimizer.update_params(layer)


# Check Ending Cost and Accuracy
model.forward(X, y)
cost_history.append(model.cost)
accuracy_history.append(get_accuracy(model.output_activation.outputs, y))
print(f"Final Cost: {cost_history[-1]} - Final Accuracy: {accuracy_history[-1]}%")

# Cost History Graph
fig = plt.figure()
plt.plot(cost_history)
plt.xlabel("Epochs")    
plt.ylabel("Cost")
fig.savefig("cost3.png")
plt.close()

# Accuracy History Graph
fig = plt.figure()
plt.plot(accuracy_history)
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
fig.savefig("accuracy3.png")
plt.close()


print("\n")
