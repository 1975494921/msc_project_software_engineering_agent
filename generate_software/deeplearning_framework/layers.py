# layers.py

import numpy as np
from tensor import Tensor
from utils import initialize_weights, initialize_bias

class Layer:
    def forward(self, inputs):
        raise NotImplementedError
    
    def backward(self, grad_output):
        raise NotImplementedError

class FullyConnected(Layer):
    def __init__(self, input_dim, output_dim):
        self.weights = Tensor(initialize_weights((input_dim, output_dim)), requires_grad=True)
        self.biases = Tensor(initialize_bias((output_dim,)), requires_grad=True)

    def forward(self, inputs):
        self.inputs = inputs
        return self.inputs @ self.weights + self.biases

    def backward(self, grad_output):
        # Gradient of the input
        inputs_grad = grad_output @ self.weights.data.T
        
        # Gradient of the weights
        if self.inputs.data.ndim == 1:
            weights_grad = np.outer(self.inputs.data, grad_output.data)
        else:
            weights_grad = self.inputs.data.T @ grad_output.data
        
        # Gradient of the biases
        biases_grad = np.sum(grad_output.data, axis=0)
        
        # Backpropagate gradients if required
        if self.weights.requires_grad:
            self.weights.backward(Tensor(weights_grad))
        if self.biases.requires_grad:
            self.biases.backward(Tensor(biases_grad))
        
        return inputs_grad

# Example usage
if __name__ == "__main__":
    from utils import initialize_weights, initialize_bias

    np.random.seed(0)
    fc = FullyConnected(5, 3)
    x = Tensor(np.random.randn(10, 5), requires_grad=True)
    
    # Forward pass
    output = fc.forward(x)
    print("Forward Output:", output)
    
    # Backward pass
    output_grad = Tensor(np.random.randn(10, 3))
    x_grad = fc.backward(output_grad)
    print("Gradients w.r.t input x:", x_grad)
    print("Gradients w.r.t weights:", fc.weights.grad)
    print("Gradients w.r.t biases:", fc.biases.grad)


class ReLU(Layer):
    def forward(self, inputs):
        # Store inputs for use in the backward pass
        self.inputs = inputs

        # ReLU activation: max(0, x)
        return Tensor(np.maximum(0, inputs.data), requires_grad=inputs.requires_grad)

    def backward(self, grad_output):
        # Gradient of ReLU is 1 for positive inputs, 0 for negative inputs
        relu_grad = (self.inputs.data > 0).astype(float)

        # Element-wise multiplication with the incoming gradient
        grad_input = grad_output.data * relu_grad

        # Return the gradient of the input
        return Tensor(grad_input)