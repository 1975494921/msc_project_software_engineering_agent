# transformer_layers.py

import numpy as np
from tensor import Tensor
from layers import Layer
from utils import softmax

def scaled_dot_product_attention(query, key, value):
    d_k = key.data.shape[-1]
    scores = query.data @ key.data.transpose(0, 2, 1) / np.sqrt(d_k)
    weights = softmax(scores)
    output = weights @ value.data
    return Tensor(output)

class MultiHeadAttention(Layer):
    def __init__(self, num_heads, model_dim):
        self.num_heads = num_heads
        self.model_dim = model_dim
        self.head_dim = model_dim // num_heads
        
        # Initialize weights for query, key, value for all heads
        self.Wq = [Tensor(np.random.randn(model_dim, self.head_dim) * 0.01, requires_grad=True) for _ in range(num_heads)]
        self.Wk = [Tensor(np.random.randn(model_dim, self.head_dim) * 0.01, requires_grad=True) for _ in range(num_heads)]
        self.Wv = [Tensor(np.random.randn(model_dim, self.head_dim) * 0.01, requires_grad=True) for _ in range(num_heads)]
        
        # Output layer weights
        self.Wo = Tensor(np.random.randn(num_heads * self.head_dim, model_dim) * 0.01, requires_grad=True)

    def forward(self, query, key, value):
        batch_size = query.data.shape[0]
        heads = []
        
        for i in range(self.num_heads):
            q = query @ self.Wq[i]
            k = key @ self.Wk[i]
            v = value @ self.Wv[i]
            heads.append(scaled_dot_product_attention(q, k, v))
        
        # Concatenate all the head outputs
        concatenated = np.concatenate([head.data for head in heads], axis=-1)
        output = Tensor(concatenated) @ self.Wo
        return output

    def backward(self, grad_output):
        # Proper backward implementation would be needed here.
        raise NotImplementedError("Backward pass is not implemented.")

class PositionalEncoding(Layer):
    def __init__(self, model_dim, max_length=5000):
        self.model_dim = model_dim
        pos_encoding = np.array([
            [pos / np.power(10000, 2 * (j // 2) / model_dim) for j in range(model_dim)]
            for pos in range(max_length)
        ])
        pos_encoding[:, 0::2] = np.sin(pos_encoding[:, 0::2])
        pos_encoding[:, 1::2] = np.cos(pos_encoding[:, 1::2])
        self.pos_encoding = Tensor(pos_encoding)

    def forward(self, x):
        length = x.data.shape[1]
        return x + Tensor(self.pos_encoding.data[:length])

    def backward(self, grad_output):
        return grad_output

class TransformerBlock(Layer):
    def __init__(self, num_heads, model_dim, forward_expansion=4, dropout=0.1):
        self.attention = MultiHeadAttention(num_heads, model_dim)
        self.norm1 = Layer()  # Placeholder for Layer normalization
        self.norm2 = Layer()  # Placeholder for Layer normalization
        self.feed_forward = Layer()  # Placeholder for feed forward network

    def forward(self, query, key, value):
        attention_output = self.attention.forward(query, key, value)
        # x = self.norm1.forward(attention_output + query)
        forward_output = self.feed_forward.forward(attention_output)
        # output = self.norm2.forward(forward_output + x)
        return output

    def backward(self, grad_output):
        # Proper backward implementation would be needed here.
        raise NotImplementedError("Backward pass is not implemented.")

# Example usage
if __name__ == "__main__":
    num_heads = 8
    model_dim = 512
    input = Tensor(np.random.randn(10, 20, model_dim), requires_grad=True)
    
    mha = MultiHeadAttention(num_heads, model_dim)
    output = mha.forward(input, input, input)
    print("Multi-Head Attention Output:", output)

    pe = PositionalEncoding(model_dim)
    encoded = pe.forward(input)
    print("Positional Encoding Output:", encoded)

    transformer_block = TransformerBlock(num_heads, model_dim)
    transformer_output = transformer_block.forward(input, input, input)
    print("Transformer Block Output:", transformer_output)
