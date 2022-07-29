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
dumb_model1 = Dense_Layer(1, 1)    # 1 input feature, 1 neuron (output feature)
dumb_model1.forward(X)

# set to best because this is the only model we have at the moment
best_model = dumb_model1
lowest_cost = mse.forward(dumb_model1.outputs, y)
print("Initial Cost:", lowest_cost)

for trial in range(1000):
    model = Dense_Layer(1, 1)    # initialize new model
    model.forward(X)    # get new model's predictions
    cost = mse.forward(model.outputs, y)    # get new model's cost

    # save model if the cost is better than our past trials
    if cost < lowest_cost:
        lowest_cost, best_model = cost, model
        print("On trial", str(trial), "we have a lower cost of", cost)

# Summary
print("\nSummary of Best Model\n")
print("Weights:", best_model.weights)
print("Biases:", best_model.biases)
print("Lowest Cost:", lowest_cost)
print("Prediction:", best_model.outputs)

print("\n")
