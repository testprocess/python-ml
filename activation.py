import numpy as np

class Activation:
    def __init__(self):
        pass

    def step(self, x):
        y = x > 0
        return y.astype(np.int)
    
    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))
    
    def relu(self, x):
        return np.maximum(0, x)