import numpy as np
from activation_functions import sigmoid, sigmoid_derivative

class DenseLayer:
    def __init__(self, input_size, output_size, activation=sigmoid):
        self.weights = np.random.rand(input_size, output_size)
        self.biases = np.random.rand(output_size)
        self.activation = activation

    def forward(self, inputs):
        self.inputs = inputs
        self.z = np.dot(inputs, self.weights) + self.biases
        self.a = self.activation(self.z)
        return self.a

    def backward(self, output_error, learning_rate):
        # Placeholder backward method
        pass