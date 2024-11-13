from neural_network import NeuralNetwork
from layers import DenseLayer
from activation_functions import sigmoid
import numpy as np

def train_model():
    # Load your data here
    X_train = np.random.rand(1000, 2)  # Example data
    y_train = np.random.rand(1000, 1)  # Example labels
    
    # Create and configure the neural network
    model = NeuralNetwork()
    model.add_layer(DenseLayer(2, 4, activation=sigmoid))
    model.add_layer(DenseLayer(4, 1, activation=sigmoid))
    
    # Train the model
    model.train(X_train, y_train, epochs=1000, learning_rate=0.01)
    
    # Save the model (implement save functionality)
    return model

if __name__ == "__main__":
    trained_model = train_model()