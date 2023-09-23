import torch
from torch import nn

from test_utils import cuda_2d_arange

B, D1, D2, D3 = 8, 8, 8, 2

mlp = nn.Sequential(
    nn.Linear(D1, D2),
    nn.ReLU(),
    nn.Linear(D2, D3),
)
mlp[0].weight.data = cuda_2d_arange(D1, D2).T
mlp[0].bias.data = cuda_2d_arange(1, D2)[0]
mlp[2].weight.data = cuda_2d_arange(D2, D3).T
mlp[2].bias.data = cuda_2d_arange(1, D3)[0]

x = cuda_2d_arange(B, D1, requires_grad=True)
x.retain_grad()
y = mlp(x)
y_grad = cuda_2d_arange(B, D3)
y.backward(y_grad)

print("x:\n", x)
for i, layer in enumerate(mlp):
    if isinstance(layer, nn.Linear):
        print(f"mlp[{i}].weight:\n", layer.weight.T)
        print(f"mlp[{i}].bias:\n", layer.bias)
print("y:\n", y)
print("y.grad:\n", y_grad)
for i, layer in enumerate(mlp):
    if isinstance(layer, nn.Linear):
        print(f"mlp[{i}].weight.grad:\n", layer.weight.grad.T)
        print(f"mlp[{i}].bias.grad:\n", layer.bias.grad)
print("x.grad:\n", x.grad)

