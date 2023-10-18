import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import nn
from torch.optim import SGD
from torch.utils.data import DataLoader, TensorDataset
from torchvision import datasets, transforms

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
log_every = 100
dtype = torch.float16


def infinite_loader(loader: DataLoader):
    while True:
        yield from loader


if __name__ == "__main__":
    n_steps = int(sys.argv[1]) if len(sys.argv) > 1 else 10000
    hidden_size = int(sys.argv[2]) if len(sys.argv) > 2 else 128
    seed = int(sys.argv[3]) if len(sys.argv) > 3 else 0

    torch.manual_seed(seed)


    dataset_path = Path(__file__).parents[1] / "data"
    mnist = datasets.MNIST(dataset_path, download=True, train=True)
    
    # Move data to GPU pre-emptively
    trainset = TensorDataset(
        mnist.data.to(device, dtype) / 255,
        mnist.targets.to(device),
    )
    train_loader = DataLoader(trainset, batch_size=32, shuffle=True)
    train_loader = infinite_loader(train_loader)

    model = nn.Sequential(
        nn.Flatten(),
        nn.Linear(784, hidden_size),
        nn.ReLU(),
        nn.Linear(hidden_size, hidden_size),
        nn.ReLU(),
        nn.Linear(hidden_size, hidden_size),
        nn.ReLU(),
        nn.Linear(hidden_size, 10),
    ).to(device)
    model = torch.compile(
        model, mode="max-autotune", fullgraph=True
    )

    optimizer = SGD(model.parameters(), lr=0.003)

    def train_step():
        x, y = next(train_loader)
        with torch.autocast(str(device), dtype=dtype):
            y_pred = model(x)
            loss = F.cross_entropy(y_pred, y)

        optimizer.zero_grad()
        loss.backward()
        return loss

    # Warmup
    for _ in range(50):
        train_step()
    
    torch.cuda.synchronize()

    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)

    step = 0
    loss = 0
    start_event.record()
    while (step := step + 1) <= n_steps:
        loss = train_step()
        optimizer.step()

        if step % log_every == 0:
            print(f"Step: {step}, loss: {loss.item():.4f}")
    end_event.record()

    torch.cuda.synchronize()

    print(f"Duration: {start_event.elapsed_time(end_event) / 1000:.4f}s")


