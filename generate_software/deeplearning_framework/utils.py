# utils.py

import numpy as np

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    s = sigmoid(x)
    return s * (1 - s)

def relu(x):
    return np.maximum(0, x)

def relu_derivative(x):
    return (x > 0).astype(float)

def tanh(x):
    return np.tanh(x)

def tanh_derivative(x):
    return 1 - np.tanh(x)**2

def softmax(x):
    exps = np.exp(x - np.max(x, axis=1, keepdims=True))
    return exps / np.sum(exps, axis=1, keepdims=True)

def mean_squared_error(y_true, y_pred):
    return np.mean((y_true - y_pred) ** 2)

def mean_squared_error_derivative(y_true, y_pred):
    return 2 * (y_pred - y_true) / y_true.size

def cross_entropy_loss(y_true, y_pred):
    m = y_true.shape[0]
    p = softmax(y_pred)
    log_likelihood = -np.log(p[range(m),y_true.argmax(axis=1)])
    loss = np.sum(log_likelihood) / m
    return loss

def cross_entropy_derivative(y_true, y_pred):
    m = y_true.shape[0]
    grad = softmax(y_pred)
    grad[range(m), y_true.argmax(axis=1)] -= 1
    grad = grad / m
    return grad

def initialize_weights(shape, type='xavier'):
    if type == 'xavier':
        stddev = np.sqrt(2 / np.sum(shape))
        return np.random.randn(*shape) * stddev
    elif type == 'he':
        stddev = np.sqrt(2 / shape[0])
        return np.random.randn(*shape) * stddev
    else:
        return np.random.randn(*shape) * 0.01

def initialize_bias(shape):
    return np.zeros(shape)

if __name__ == "__main__":
    # Test utility functions
    x = np.array([[1.0, -1.0, 0.0], [1.0, -2.0, 2.0]])
    y_true = np.array([[0.0, 1.0], [1.0, 0.0]])
    y_pred = np.array([[0.2, 0.8], [0.9, 0.1]])

    print("Sigmoid:", sigmoid(x))
    print("ReLU:", relu(x))
    print("Tanh:", tanh(x))
    print("Softmax:", softmax(x))
    print("MSE Loss:", mean_squared_error(y_true, y_pred))
    print("Cross Entropy Loss:", cross_entropy_loss(y_true, softmax(y_pred)))
    print("Weight Initialization (Xavier):", initialize_weights((2, 3), 'xavier'))
    print("Bias Initialization:", initialize_bias((2,)))
