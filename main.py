import matplotlib.pyplot as plt
import numpy as np

from logic import Logic
from activation import Activation


perceptron = Logic()
activation = Activation()
print(perceptron.AND(1, 0))
print(perceptron.XOR(0, 1))

x = np.arange(-5.0, 5.0, 0.1)
y = activation.leakyRelu(x)
print(x, y)
plt.plot(x, y)
plt.ylim(-0.1, 5)
plt.show()