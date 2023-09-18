import torch
import torch.nn.functional as F

from test_utils import cuda_2d_arange

B, C = 3, 5

y_pred = cuda_2d_arange(B, C, requires_grad=True)
y_pred.retain_grad()
y_true = torch.arange(B, dtype=torch.long, device=y_pred.device)

loss = F.cross_entropy(y_pred, y_true)
loss.backward()

print("loss\n", loss.item())
print("y_pred\n", y_pred)
print("y_pred_softmax\n", F.softmax(y_pred, dim=1))
print("y_true\n", y_true)
print("y_pred.grad\n", y_pred.grad)

