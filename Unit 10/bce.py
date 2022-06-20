import numpy as np
np.random.seed(0)    # For repeatability

# Does not handle for extreme values (e.g. 0 or 1)

class BCE_Cost:
    def forward(self, y_pred, y_true):
        return np.mean((y_true * -np.log(y_pred)) + ((1 - y_true) * -np.log(1 - y_pred)))
    
    def backward(self, y_pred, y_true):
        self.dinputs = -(y_true / y_pred) + ((1 - y_true) / (1 - y_pred))
        self.dinputs /= len(y_pred)


# Handles for extreme values (e.g. 0 or 1)

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


