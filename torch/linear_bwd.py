import torch

M = 12
N = 13
K = 7

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

x = torch.arange(M * K, dtype=torch.float16, device=device).reshape(K, M).T / 10
w = torch.arange(N * K, dtype=torch.float16, device=device).reshape(K, N).T / 20
b = torch.arange(N, dtype=torch.float16, device=device)

x.requires_grad_(True)
w.requires_grad_(True)
b.requires_grad_(True)

y = torch.matmul(x, w.T) + b
loss = y.sum()

loss.backward()

print(f"{x=}")
print(f"{w=}")
print(f"{b=}")
print(f"{x.grad=}")
print(f"{w.grad=}")
print(f"{b.grad=}")

