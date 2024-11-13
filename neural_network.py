import numpy as np

class NeuralNetwork:
    def __init__(self):
        self.layers = []
        self.loss_history = []
        
    def add_layer(self, layer):
        self.layers.append(layer)
        
    def forward(self, X):
        current_input = X
        for layer in self.layers:
            current_input = layer.forward(current_input)
        return current_input
    
    def backward(self, X, y, learning_rate):
        m = X.shape[0]
        output = self.forward(X)
        
        # Calculate initial gradient
        gradient = 2 * (output - y) / m
        
        # Backpropagate through layers
        for layer in reversed(self.layers):
            gradient = layer.backward(gradient, learning_rate)
            
    def train(self, X, y, epochs, learning_rate, batch_size=32):
        for epoch in range(epochs):
            # Mini-batch training
            for i in range(0, len(X), batch_size):
                X_batch = X[i:i + batch_size]
                y_batch = y[i:i + batch_size]
                
                # Forward and backward pass
                self.backward(X_batch, y_batch, learning_rate)
                
            # Calculate loss for the epoch
            predictions = self.forward(X)
            loss = np.mean((predictions - y) ** 2)
            self.loss_history.append(loss)
            
            if epoch % 100 == 0:
                print(f"Epoch {epoch}, Loss: {loss:.4f}")