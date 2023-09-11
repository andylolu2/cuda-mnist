import time
import torch
import torch.nn.functional as F
from torch import nn
from torch.optim import SGD
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def infinite_loader(loader: DataLoader):
    while True:
        yield from loader

if __name__ == "__main__":
    transform = transforms.Compose([
        transforms.ToTensor(),
    ])

    trainset = datasets.MNIST(".pytorch/MNIST", download=True, train=True, transform=transform)
    train_loader = DataLoader(trainset, batch_size=16, shuffle=True)
    train_loader = infinite_loader(train_loader)


    model = nn.Sequential(
        nn.Flatten(),
        nn.Linear(784, 128),
        nn.ReLU(),
        nn.Linear(128, 10),
    ).to(device, dtype=torch.float16)

    optimizer = SGD(model.parameters(), lr=0.003)

    start = time.time()

    step = 0
    while (step := step + 1) < 5000:
        x, y = next(train_loader)
        x = x.to(device, dtype=torch.float16)
        y = y.to(device)

        y_pred = model(x)
        loss = F.cross_entropy(y_pred, y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    
    print(f"Final loss: {loss.item():.4f}")
    print(f"Duration: {time.time() - start:.4f}s")

