import numpy as np

def load_data():
    # Placeholder function to load data
    # Replace with actual data loading logic
    X_train = np.random.rand(60000, 784)
    y_train = np.random.randint(0, 10, 60000)
    X_test = np.random.rand(10000, 784)
    y_test = np.random.randint(0, 10, 10000)
    return X_train, y_train, X_test, y_test