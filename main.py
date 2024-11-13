from neural_network import NeuralNetwork
from utils import load_data
from activation_functions import sigmoid
from layers import DenseLayer

def main():
    # Load data
    X_train, y_train, X_test, y_test = load_data()

    # Initialize neural network
    nn = NeuralNetwork()
    nn.add_layer(DenseLayer(input_size=784, output_size=128, activation=sigmoid))
    nn.add_layer(DenseLayer(input_size=128, output_size=10, activation=sigmoid))

    # Train neural network
    nn.train(X_train, y_train, epochs=10, learning_rate=0.01)

    # Test neural network
    accuracy = nn.evaluate(X_test, y_test)
    print(f"Test accuracy: {accuracy:.2f}%")

if __name__ == "__main__":
    main()

    