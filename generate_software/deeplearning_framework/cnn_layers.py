# cnn_layers.py

import numpy as np
from tensor import Tensor
from layers import Layer

def im2col(input_data, filter_h, filter_w, stride=1, pad=0):
    N, C, H, W = input_data.shape
    out_h = (H + 2 * pad - filter_h) // stride + 1
    out_w = (W + 2 * pad - filter_w) // stride + 1

    img = np.pad(input_data, [(0, 0), (0, 0), (pad, pad), (pad, pad)], 'constant')
    col = np.zeros((N, C, filter_h, filter_w, out_h, out_w))

    for y in range(filter_h):
        y_max = y + stride * out_h
        for x in range(filter_w):
            x_max = x + stride * out_w
            col[:, :, y, x, :, :] = img[:, :, y:y_max:stride, x:x_max:stride]

    col = col.transpose(0, 4, 5, 1, 2, 3).reshape(N * out_h * out_w, -1)
    return col

def col2im(col, input_shape, filter_h, filter_w, stride=1, pad=0):
    N, C, H, W = input_shape
    out_h = (H + 2 * pad - filter_h) // stride + 1
    out_w = (W + 2 * pad - filter_w) // stride + 1
    col = col.reshape(N, out_h, out_w, C, filter_h, filter_w).transpose(0, 3, 4, 5, 1, 2)

    img = np.zeros((N, C, H + 2 * pad + stride - 1, W + 2 * pad + stride - 1))
    for y in range(filter_h):
        y_max = y + stride * out_h
        for x in range(filter_w):
            x_max = x + stride * out_w
            img[:, :, y:y_max:stride, x:x_max:stride] += col[:, :, y, x, :, :]

    return img[:, :, pad:H + pad, pad:W + pad]

class Convolution(Layer):
    def __init__(self, input_channels, output_channels, kernel_size, stride=1, pad=0):
        self.input_channels = input_channels
        self.output_channels = output_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.pad = pad
        self.weights = Tensor(np.random.randn(output_channels, input_channels, kernel_size, kernel_size) * 0.01, requires_grad=True)
        self.biases = Tensor(np.zeros(output_channels), requires_grad=True)

    def forward(self, inputs):
        self.inputs = inputs
        col = im2col(inputs.data, self.kernel_size, self.kernel_size, self.stride, self.pad)
        col_w = self.weights.data.reshape(self.output_channels, -1).T
        out = np.dot(col, col_w) + self.biases.data
        N, H, W, C = inputs.data.shape
        out_h = (H + 2 * self.pad - self.kernel_size) // self.stride + 1
        out_w = (W + 2 * self.pad - self.kernel_size) // self.stride + 1
        out = out.reshape(N, out_h, out_w, self.output_channels).transpose(0, 3, 1, 2)
        return Tensor(out, requires_grad=True)

    def backward(self, grad_output):
        # This section would need implementation for gradients if model training is performed.
        raise NotImplementedError("Backward pass is not implemented.")

class MaxPooling(Layer):
    def __init__(self, pool_size, stride=None, pad=0):
        self.pool_size = pool_size
        self.stride = stride or pool_size
        self.pad = pad

    def forward(self, inputs):
        self.inputs = inputs
        N, C, H, W = inputs.data.shape
        out_h = (H - self.pool_size) // self.stride + 1
        out_w = (W - self.pool_size) // self.stride + 1

        col = im2col(inputs.data, self.pool_size, self.pool_size, self.stride, self.pad)
        col = col.reshape(-1, self.pool_size * self.pool_size)
        arg_max = np.argmax(col, axis=1)
        out = np.max(col, axis=1)
        out = out.reshape(N, out_h, out_w, C).transpose(0, 3, 1, 2)
        self.arg_max = arg_max
        return Tensor(out, requires_grad=True)

    def backward(self, grad_output):
        # This section would need implementation for gradients if model training is performed.
        raise NotImplementedError("Backward pass is not implemented.")

# Example usage
if __name__ == "__main__":
    from tensor import Tensor

    # Input image batch of size (batch_size, channels, height, width)
    x = Tensor(np.random.randn(1, 1, 4, 4), requires_grad=True)
    
    # Convolution layer
    conv = Convolution(1, 1, 3, stride=1, pad=0)
    output = conv.forward(x)
    print("Convolution output:", output)
    
    # MaxPooling layer
    pool = MaxPooling(2)
    pooled_output = pool.forward(output)
    print("Pooled output:", pooled_output)
