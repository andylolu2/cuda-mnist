from torch import nn
import torch
from test_utils import cuda_2d_arange

B, D1, D2 = 8, 8, 3

linear = nn.Linear(D1, D2)
linear.weight.data = cuda_2d_arange(D1, D2).T
linear.bias.data = cuda_2d_arange(1, D2)[0]

x = cuda_2d_arange(B, D1, requires_grad=True)
x.retain_grad()
y = linear(x)
y_grad = cuda_2d_arange(B, D2)
y.backward(y_grad)

print("x:\n", x)
print("w:\n", linear.weight.data.T)
print("b:\n", linear.bias.data)
print("y:\n", y)
print("y.grad:\n", y_grad)
print("w.grad:\n", linear.weight.grad.T)
print("b.grad:\n", linear.bias.grad)
print("x.grad:\n", x.grad)

