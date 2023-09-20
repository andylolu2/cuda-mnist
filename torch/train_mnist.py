import time

from torchvision import datasets, transforms

import torch
import torch.nn.functional as F
from torch import nn
from torch.optim import SGD
from torch.utils.data import DataLoader, TensorDataset

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
log_every = 500
dtype = torch.float16


def infinite_loader(loader: DataLoader):
    while True:
        yield from loader


if __name__ == "__main__":
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
        ]
    )

    mnist = datasets.MNIST(
        ".pytorch/MNIST", download=True, train=True, transform=transform
    )
    trainset = TensorDataset(
        mnist.data.to(device, dtype) / 255,
        mnist.targets.to(device),
    )
    train_loader = DataLoader(trainset, batch_size=32, shuffle=True)
    train_loader = infinite_loader(train_loader)

    model = nn.Sequential(
        nn.Flatten(),
        nn.Linear(784, 1024),
        nn.ReLU(),
        nn.Linear(1024, 1024),
        nn.ReLU(),
        nn.Linear(1024, 10),
    ).to(device, dtype)

    optimizer = SGD(model.parameters(), lr=0.003)

    start = time.time()

    step = 0
    loss = 0
    while (step := step + 1) < 10000:
        x, y = next(train_loader)

        y_pred = model(x)
        loss = F.cross_entropy(y_pred, y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if step % log_every == 0:
            print(f"Step: {step}, loss: {loss.item():.4f}")

    if torch.cuda.is_available():
        torch.cuda.synchronize()

    print(f"Final loss: {loss.item():.4f}")
    print(f"Duration: {time.time() - start:.4f}s")
