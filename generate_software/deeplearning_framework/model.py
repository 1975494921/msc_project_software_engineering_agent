# model.py

import numpy as np
from layers import Layer, FullyConnected
from cnn_layers import Convolution, MaxPooling
from rnn_layers import RNNCell, GRUCell
from lstm_layers import LSTMCell
from transformer_layers import TransformerBlock
from optimizers import Optimizer, SGD, Adam
from tensor import Tensor
from utils import mean_squared_error, mean_squared_error_derivative, cross_entropy_loss, cross_entropy_derivative

class Model:
    def __init__(self):
        self.layers = []
        self.loss_function = None
        self.optimizer = None

    def add(self, layer):
        assert isinstance(layer, Layer), "Non-layer object cannot be added to the model"
        self.layers.append(layer)

    def set_loss_function(self, loss_function):
        self.loss_function = loss_function

    def set_optimizer(self, optimizer):
        assert isinstance(optimizer, Optimizer), "Non-optimizer object cannot be set"
        self.optimizer = optimizer

    def forward(self, x):
        for layer in self.layers:
            x = layer.forward(x)
        return x

    def backward(self, grad_output):
        for layer in reversed(self.layers):
            grad_output = layer.backward(grad_output)

    def compute_loss(self, predicted, true):
        return self.loss_function(true, predicted)

    def train_step(self, x, y):
        predicted = self.forward(x)
        loss = self.compute_loss(predicted, y)
        self.backward(Tensor(mean_squared_error_derivative(y.data, predicted.data), requires_grad=False))
        self.optimizer.step()
        return loss

    def predict(self, x):
        return self.forward(x)

# Example usage
if __name__ == "__main__":
    model = Model()
    model.add(FullyConnected(784, 128))
    model.add(FullyConnected(128, 10))
    model.set_loss_function(mean_squared_error)

    # Dummy data
    x = Tensor(np.random.randn(1, 784), requires_grad=True)
    y_true = Tensor(np.random.randn(1, 10), requires_grad=True)

    # Forward pass to ensure all parameters have their gradients computed
    output = model.forward(x)
    loss = model.compute_loss(output, y_true)
    grad_output = Tensor(mean_squared_error_derivative(y_true.data, output.data), requires_grad=False)
    model.backward(grad_output)

    # Collect parameters
    parameters = []
    for layer in model.layers:
        if hasattr(layer, 'weights'):
            parameters.append(layer.weights)
        if hasattr(layer, 'biases'):
            parameters.append(layer.biases)

    # Set optimizer now that gradients are initialized
    model.set_optimizer(SGD(parameters, lr=0.01))

    # Training step
    loss = model.train_step(x, y_true)
    print("Training loss:", loss)

    # Prediction
    predictions = model.predict(x)
    print("Predictions:", predictions)
