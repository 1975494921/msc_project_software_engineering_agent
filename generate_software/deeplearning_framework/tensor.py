# tensor.py

import numpy as np

class Tensor:
    def __init__(self, data, requires_grad=False):
        self.data = np.array(data, dtype=np.float32)
        self.requires_grad = requires_grad
        self.grad = None
        self._backward = lambda: None
        
    def __repr__(self):
        return f"Tensor({self.data}, requires_grad={self.requires_grad})"
    
    def zero_grad(self):
        self.grad = None
    
    def backward(self, grad=None):
        if not self.requires_grad:
            raise RuntimeError("Called backward on non-grad tensor.")
        
        if grad is None:
            if self.data.size == 1:
                grad = np.array(1.0)
            else:
                raise RuntimeError("grad must be specified for non-scalar outputs")
        
        if self.grad is None:
            self.grad = grad
        else:
            self.grad += grad
        
        self._backward()
    
    def __add__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other)
        data = self.data + other.data
        requires_grad = self.requires_grad or other.requires_grad
        result = Tensor(data, requires_grad=requires_grad)
        
        if requires_grad:
            def backward():
                if self.requires_grad:
                    self.backward(result.grad)
                if other.requires_grad:
                    other.backward(result.grad)
            result._backward = backward

        return result
    
    def __mul__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other)
        data = self.data * other.data
        requires_grad = self.requires_grad or other.requires_grad
        result = Tensor(data, requires_grad=requires_grad)
        
        if requires_grad:
            def backward():
                if self.requires_grad:
                    self.backward(result.grad * other.data)
                if other.requires_grad:
                    other.backward(result.grad * self.data)
            result._backward = backward
        
        return result

    def __matmul__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other)
        data = self.data @ other.data
        requires_grad = self.requires_grad or other.requires_grad
        result = Tensor(data, requires_grad=requires_grad)
        
        if requires_grad:
            def backward():
                if self.requires_grad:
                    self.backward(result.grad @ other.data.T)
                if other.requires_grad:
                    other.backward(self.data.T @ result.grad)
            result._backward = backward
        
        return result

if __name__ == "__main__":
    # Test Tensor operations with autograd
    a = Tensor([[1.0, 2.0], [3.0, 4.0]], requires_grad=True)
    b = Tensor([[2.0, 0.0], [0.0, 2.0]], requires_grad=True)
    c = a + b
    d = a * b
    e = a @ b
    
    print("Addition:", c)
    print("Multiplication:", d)
    print("Matrix multiplication:", e)

    e.zero_grad()
    e.backward(np.array([[1.0, 0.0], [0.0, 1.0]]))
    print("Gradient w.r.t a after matrix multiplication:", a.grad)
    d.zero_grad()
    d.backward(np.array([[1.0, 1.0], [1.0, 1.0]]))
    print("Gradient w.r.t a after multiplication:", a.grad)