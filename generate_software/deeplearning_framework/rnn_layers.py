# rnn_layers.py

import numpy as np
from tensor import Tensor
from layers import Layer

class RNNCell(Layer):
    def __init__(self, input_size, hidden_size):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.Wx = Tensor(np.random.randn(input_size, hidden_size) * 0.01, requires_grad=True)
        self.Wh = Tensor(np.random.randn(hidden_size, hidden_size) * 0.01, requires_grad=True)
        self.b = Tensor(np.zeros(hidden_size), requires_grad=True)
    
    def forward(self, input, hidden):
        self.prev_input = input
        self.prev_hidden = hidden
        return Tensor(np.tanh(input.data @ self.Wx.data + hidden.data @ self.Wh.data + self.b.data), requires_grad=True)

    def backward(self, grad_output):
        # Compute gradients for input and hidden state
        dtanh = (1 - self.prev_hidden.data**2) * grad_output.data  # derivative through tanh
        grad_input = dtanh @ self.Wx.data.T
        grad_hidden = dtanh @ self.Wh.data.T

        # Compute gradients for weights
        grad_Wx = self.prev_input.data.T @ dtanh
        grad_Wh = self.prev_hidden.data.T @ dtanh
        grad_b = np.sum(dtanh, axis=0)

        # Update gradients in tensors
        self.Wx.backward(Tensor(grad_Wx))
        self.Wh.backward(Tensor(grad_Wh))
        self.b.backward(Tensor(grad_b))

        return Tensor(grad_input), Tensor(grad_hidden)

class GRUCell(Layer):
    def __init__(self, input_size, hidden_size):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.reset_gate = RNNCell(input_size, hidden_size)
        self.update_gate = RNNCell(input_size, hidden_size)
        self.candidate_hidden = RNNCell(input_size, hidden_size)

    def forward(self, input, hidden):
        reset = self.reset_gate.forward(input, hidden)
        update = self.update_gate.forward(input, hidden)
        candidate = self.candidate_hidden.forward(input, Tensor(reset.data * hidden.data, requires_grad=True))
        new_hidden = Tensor(update.data * hidden.data + (1 - update.data) * candidate.data, requires_grad=True)
        return new_hidden

    def backward(self, grad_output):
        # This would also need to handle gradients properly
        raise NotImplementedError("Backward pass for GRUCell is not implemented.")

# Example usage
if __name__ == "__main__":
    np.random.seed(0)
    input_size = 5
    hidden_size = 10
    x = Tensor(np.random.randn(1, input_size), requires_grad=True)
    h = Tensor(np.zeros((1, hidden_size)), requires_grad=True)

    rnn_cell = RNNCell(input_size, hidden_size)
    output = rnn_cell.forward(x, h)
    print("RNN Cell Output:", output)

    gru_cell = GRUCell(input_size, hidden_size)
    gru_output = gru_cell.forward(x, h)
    print("GRU Cell Output:", gru_output)
