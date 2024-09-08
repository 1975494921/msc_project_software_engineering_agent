# optimizers.py

import numpy as np
from tensor import Tensor

class Optimizer:
    def step(self):
        raise NotImplementedError

class SGD(Optimizer):
    def __init__(self, parameters, lr=0.01, momentum=0.0):
        self.parameters = parameters
        self.lr = lr
        self.momentum = momentum
        self.velocity = [np.zeros_like(p.grad.data) for p in parameters if p.requires_grad]

    def step(self):
        for i, param in enumerate(self.parameters):
            if param.requires_grad:
                self.velocity[i] = self.momentum * self.velocity[i] + self.lr * param.grad.data
                param.data -= self.velocity[i]

class Adam(Optimizer):
    def __init__(self, parameters, lr=0.001, beta1=0.9, beta2=0.999, epsilon=1e-8):
        self.parameters = parameters
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.m = [np.zeros_like(p.grad.data) for p in parameters if p.requires_grad]
        self.v = [np.zeros_like(p.grad.data) for p in parameters if p.requires_grad]
        self.t = 0

    def step(self):
        self.t += 1
        for i, param in enumerate(self.parameters):
            if param.requires_grad:
                self.m[i] = self.beta1 * self.m[i] + (1 - self.beta1) * param.grad.data
                self.v[i] = self.beta2 * self.v[i] + (1 - self.beta2) * (param.grad.data ** 2)
                
                m_hat = self.m[i] / (1 - self.beta1 ** self.t)
                v_hat = self.v[i] / (1 - self.beta2 ** self.t)
                
                param.data -= self.lr * m_hat / (np.sqrt(v_hat) + self.epsilon)

# Example usage
if __name__ == "__main__":
    from layers import FullyConnected

    np.random.seed(0)
    fc_layer = FullyConnected(5, 3)
    x = Tensor(np.random.randn(10, 5), requires_grad=True)
    output = fc_layer.forward(x)
    
    output_grad = Tensor(np.random.randn(10, 3))
    fc_layer.backward(output_grad)
    
    optimizer = Adam([fc_layer.weights, fc_layer.biases])
    optimizer.step()
    
    print("Updated weights:", fc_layer.weights)
    print("Updated biases:", fc_layer.biases)
