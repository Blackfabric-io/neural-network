import numpy as np
import matplotlib.pyplot as plt

def evaluate_model(model, X_test, y_test):
    # Make predictions
    predictions = model.forward(X_test)
    
    # Calculate metrics
    mse = np.mean((predictions - y_test) ** 2)
    accuracy = np.mean(np.abs(predictions - y_test) < 0.5)
    
    # Plot results
    plt.figure(figsize=(10, 5))
    
    # Loss history
    plt.subplot(1, 2, 1)
    plt.plot(model.loss_history)
    plt.title('Training Loss History')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    
    # Predictions vs Actual
    plt.subplot(1, 2, 2)
    plt.scatter(y_test, predictions)
    plt.plot([0, 1], [0, 1], 'r--')
    plt.title('Predictions vs Actual')
    plt.xlabel('Actual')
    plt.ylabel('Predicted')
    
    plt.tight_layout()
    plt.show()
    
    print(f"Mean Squared Error: {mse:.4f}")
    print(f"Accuracy: {accuracy:.2%}")
    
    return mse, accuracy