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
og_cost = mse.forward(model.outputs, y)    # original cost
EPS = 1e-4    # epsilon, (1 * (10 ** -4))

print("Weights:", model.weights)
print("Biases:", model.biases)
print("Cost:", mse.forward(model.outputs, y))
print("EPSILON:", EPS)

# (x2, y2) -- w
model.weights += EPS    # x2
x2 = float(model.weights)    # since there is only 1 weight value
model.forward(X)
cost2 = mse.forward(model.outputs, y)    # y2
model.weights -= EPS    # reset to original value

# (x1, y1) -- w
model.weights -= EPS    # x1
x1 = float(model.weights)    # since there is only 1 weight value
model.forward(X)
cost1 = mse.forward(model.outputs, y)    # y2
model.weights += EPS    # reset to original value

slope_w = (cost2 - cost1) / (x2 - x1)
print("Slope of w:", slope_w)

# increase/decrease -- w
if slope_w < 0:
    print("slope of w is negative, we need to increase the weight to decrease the cost")
    model.weights += EPS
elif slope_w > 0:
    print("slope of w is positive, we need to decrease the weight to decrease the cost")
    model.weights -= EPS
else:
    print("oh no! :(")

# (x2, y2) -- b
model.biases += EPS    # x2
x2 = float(model.biases)    # since there is only 1 bias value
model.forward(X)
cost2 = mse.forward(model.outputs, y)    # y2
model.biases -= EPS    # reset to original value

# (x1, y1) -- b
model.biases -= EPS    # x1
x1 = float(model.biases)    # since there is only 1 bias value
model.forward(X)
cost1 = mse.forward(model.outputs, y)    # y2
model.biases += EPS    # reset to original value

slope_b = (cost2 - cost1) / (x2 - x1)
print("Slope of b:", slope_b)

# increase/decrease -- b
if slope_b < 0:
    print("slope of b is negative, we need to increase the weight to decrease the cost")
    model.biases += EPS
elif slope_b > 0:
    print("slope of b is positive, we need to decrease the weight to decrease the cost")
    model.biases -= EPS
else:
    print("oh no! :(")

# check if cost decreased
model.forward(X)
new_cost = mse.forward(model.outputs, y)    # y2
print(new_cost)
if new_cost < og_cost:
    print("Change:", new_cost-og_cost)
else:
    print("oh no! :(")

print("\n")
