# lstm_layers.py

import numpy as np
from tensor import Tensor
from layers import Layer
from utils import sigmoid, tanh

class LSTMCell(Layer):
    def __init__(self, input_size, hidden_size):
        self.input_size = input_size
        self.hidden_size = hidden_size
        
        # LSTM has four sets of gates, each with its own weight and bias
        self.Wf = Tensor(np.random.randn(input_size, hidden_size) * 0.01, requires_grad=True)  # Forget gate weights
        self.Wi = Tensor(np.random.randn(input_size, hidden_size) * 0.01, requires_grad=True)  # Input gate weights
        self.Wo = Tensor(np.random.randn(input_size, hidden_size) * 0.01, requires_grad=True)  # Output gate weights
        self.Wc = Tensor(np.random.randn(input_size, hidden_size) * 0.01, requires_grad=True)  # Cell state weights
        
        self.Uf = Tensor(np.random.randn(hidden_size, hidden_size) * 0.01, requires_grad=True) # Forget gate recurrent weights
        self.Ui = Tensor(np.random.randn(hidden_size, hidden_size) * 0.01, requires_grad=True) # Input gate recurrent weights
        self.Uo = Tensor(np.random.randn(hidden_size, hidden_size) * 0.01, requires_grad=True) # Output gate recurrent weights
        self.Uc = Tensor(np.random.randn(hidden_size, hidden_size) * 0.01, requires_grad=True) # Cell state recurrent weights
        
        self.bf = Tensor(np.zeros(hidden_size), requires_grad=True)  # Forget gate bias
        self.bi = Tensor(np.zeros(hidden_size), requires_grad=True)  # Input gate bias
        self.bo = Tensor(np.zeros(hidden_size), requires_grad=True)  # Output gate bias
        self.bc = Tensor(np.zeros(hidden_size), requires_grad=True)  # Cell state bias

    def forward(self, input, hidden, cell_state):
        # Forget gate
        ft = Tensor(sigmoid(input.data @ self.Wf.data + hidden.data @ self.Uf.data + self.bf.data))
        
        # Input gate
        it = Tensor(sigmoid(input.data @ self.Wi.data + hidden.data @ self.Ui.data + self.bi.data))
        
        # Cell candidate
        ct_hat = Tensor(tanh(input.data @ self.Wc.data + hidden.data @ self.Uc.data + self.bc.data))
        
        # New cell state
        ct = ft * cell_state + it * ct_hat
        
        # Output gate
        ot = Tensor(sigmoid(input.data @ self.Wo.data + hidden.data @ self.Uo.data + self.bo.data))
        
        # New hidden state
        ht = ot * Tensor(tanh(ct.data))
        
        return ht, ct

    def backward(self, grad_output, grad_state):
        # This would need to handle gradients properly
        raise NotImplementedError("Backward pass for LSTMCell is not implemented.")

# Example usage
if __name__ == "__main__":
    np.random.seed(0)
    input_size = 5
    hidden_size = 10
    x = Tensor(np.random.randn(1, input_size), requires_grad=True)
    h = Tensor(np.zeros((1, hidden_size)), requires_grad=True)
    c = Tensor(np.zeros((1, hidden_size)), requires_grad=True)

    lstm_cell = LSTMCell(input_size, hidden_size)
    h_new, c_new = lstm_cell.forward(x, h, c)
    print("LSTM Cell Output (Hidden State):", h_new)
    print("LSTM Cell Output (Cell State):", c_new)
