print("\n")

import numpy as np
np.random.seed(0)    # For repeatability

class Dense_Layer:
    def __init__(self, n_inputs, n_neurons):
        self.weights = 0.01 * np.random.randn(n_inputs, n_neurons)
        self.biases = np.zeros([1, n_neurons])

    def forward(self, inputs):    # inputs is X
        self.outputs = np.dot(inputs, self.weights) + self.biases

class MAE_Cost:
    def forward(self, y_pred, y_true):    # y_pred is prediction from model, y_true is the answer
        return np.mean(np.abs(y_pred - y_true))

class MSE_Cost:
    def forward(self, y_pred, y_true):
        return np.mean(np.square(y_pred - y_true))

# hours studied
X = np.array([[0], [1], [2], [3], [4], [5], [6], [7], [8]])    # one input feature for each example
# percentage
y = np.array([[20], [30], [40], [50], [60], [70], [80], [90], [100]])    # one output feature for each example

mse = MSE_Cost()    # define cost function
model = Dense_Layer(1, 1)    # 1 input feature, 1 neuron (output feature)
model.forward(X)

# as of now, this is our lowest cost
lowest_cost = mse.forward(model.outputs, y)
print("Initial Cost:", lowest_cost)

for trial in range(1000):
    # amount to update
    update_w = 0.05 * np.random.randn(1, 1)
    update_b = 0.05 * np.random.randn(1, 1)

    # update and get cost
    model.weights += update_w
    model.biases += update_b
    model.forward(X)
    cost = mse.forward(model.outputs, y)

    # if new cost is higher, undo the updates, otherwise update lowest_cost
    if cost > lowest_cost:
        model.weights -= update_w
        model.biases -= update_b
    else:
        lowest_cost = cost


# Summary
print("\nSummary of Updated Model\n")
print("Weights:", model.weights)
print("Biases:", model.biases)
print("Lowest Cost:", lowest_cost)
print("Prediction:", model.outputs)

print("\n")
