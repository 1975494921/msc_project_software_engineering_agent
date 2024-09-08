# autograd.py

from tensor import Tensor

class Operation:
    def __call__(self, *inputs):
        self.inputs = [x if isinstance(x, Tensor) else Tensor(x) for x in inputs]
        self.outputs = self.forward(*self.inputs)
        if not isinstance(self.outputs, list):
            self.outputs = [self.outputs] if isinstance(self.outputs, Tensor) else [Tensor(self.outputs)]

        requires_grad = any(input.requires_grad for input in self.inputs)

        if requires_grad:
            for output in self.outputs:
                output.requires_grad = True
                output._backward = lambda grad_output, output=output: self.backward(grad_output, output)

        return self.outputs[0] if len(self.outputs) == 1 else self.outputs

    def forward(self, *inputs):
        raise NotImplementedError

    def backward(self, grad_output, output):
        raise NotImplementedError

class Add(Operation):
    def forward(self, x, y):
        return x.data + y.data
    
    def backward(self, grad_output, output):
        return grad_output, grad_output

class Multiply(Operation):
    def forward(self, x, y):
        return x.data * y.data
    
    def backward(self, grad_output, output):
        x, y = self.inputs
        return grad_output * y.data, grad_output * x.data

class MatMul(Operation):
    def forward(self, x, y):
        return x.data @ y.data
    
    def backward(self, grad_output, output):
        x, y = self.inputs
        return grad_output @ y.data.T, x.data.T @ grad_output

# Example usage
if __name__ == "__main__":
    # Create tensors with requires_grad=True to participate in gradient computation
    a = Tensor([[1.0, 2.0], [3.0, 4.0]], requires_grad=True)
    b = Tensor([[2.0, 0.0], [0.0, 2.0]], requires_grad=True)
    
    # Operations
    add_op = Add()
    mul_op = Multiply()
    matmul_op = MatMul()
    
    # Perform operations
    c = add_op(a, b)
    d = mul_op(a, b)
    e = matmul_op(a, b)
    
    print("Addition:", c)
    print("Multiplication:", d)
    print("Matrix Multiplication:", e)

    # Backward pass
    e.zero_grad()
    e.backward(Tensor([[1.0, 0.0], [0.0, 1.0]]))  # Backpropagate from this point
    print("Gradient w.r.t a after matrix multiplication:", a.grad)
    d.zero_grad()
    d.backward(Tensor([[1.0, 1.0], [1.0, 1.0]]))
    print("Gradient w.r.t a after multiplication:", a.grad)
