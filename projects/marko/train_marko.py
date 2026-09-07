import argparse
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


DATA_DIR = Path(__file__).resolve().parents[2] / "data"
CLASSES = (
    "airplane",
    "automobile",
    "bird",
    "cat",
    "deer",
    "dog",
    "frog",
    "horse",
    "ship",
    "truck",
)


def get_device():
    if hasattr(torch, "xpu") and torch.xpu.is_available():
        return torch.device("xpu")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def load_cifar10_data(batch_size=64):
    transform = transforms.ToTensor()

    print("Loading CIFAR-10 training data...")
    train_dataset = datasets.CIFAR10(
        root=DATA_DIR,
        train=True,
        download=False,
        transform=transform,
    )

    print("Loading CIFAR-10 test data...")
    test_dataset = datasets.CIFAR10(
        root=DATA_DIR,
        train=False,
        download=False,
        transform=transform,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
    )
    return train_loader, test_loader


def show_sample(train_loader):
    import matplotlib.pyplot as plt

    images, labels = next(iter(train_loader))

    print(f"Image batch shape: {images.shape}")
    print(f"Label batch shape: {labels.shape}")
    for label in labels[:10]:
        print(label.item(), CLASSES[label.item()])

    image = images[0].permute(1, 2, 0)
    plt.imshow(image)
    plt.title(CLASSES[labels[0].item()])
    plt.axis("off")
    plt.show()


class CNN(nn.Module):
    def __init__(self):
        super().__init__()

        self.dropout = nn.Dropout(0.25)

        # Convolution layers
        self.conv1 = nn.Conv2d(3, 32, 3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, 3, padding=1)

        self.pool = nn.MaxPool2d(2, 2)

        # Fully connected layers
        self.fc1 = nn.Linear(128 * 4 * 4, 256)
        self.fc2 = nn.Linear(256, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))

        x = torch.flatten(x, 1)

        # Dropout is applied around the first fully connected layer.
        x = self.dropout(x)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        return self.fc2(x)


def evaluate(model, test_loader, loss_fn, device):
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            total_loss += loss_fn(outputs, labels).item()
            correct += (outputs.argmax(dim=1) == labels).sum().item()
            total += labels.size(0)

    average_loss = total_loss / len(test_loader)
    accuracy = 100 * correct / total
    return average_loss, accuracy


def train(model, train_loader, test_loader, loss_fn, optimizer, device, epochs):
    for epoch in range(epochs):
        model.train()
        total_loss = 0.0

        for images, labels in train_loader:
            images = images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = loss_fn(outputs, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        train_loss = total_loss / len(train_loader)
        test_loss, test_accuracy = evaluate(model, test_loader, loss_fn, device)
        print(
            f"Epoch {epoch + 1}/{epochs} - "
            f"train loss: {train_loss:.4f} - "
            f"test loss: {test_loss:.4f} - "
            f"test accuracy: {test_accuracy:.2f}%"
        )


def parse_args():
    parser = argparse.ArgumentParser(description="Train a CNN on CIFAR-10.")
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument(
        "--show-sample",
        action="store_true",
        help="show one training image before training",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    device = get_device()
    train_loader, test_loader = load_cifar10_data(args.batch_size)

    if args.show_sample:
        show_sample(train_loader)

    model = CNN().to(device)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    print(f"Using device: {device}")
    train(
        model,
        train_loader,
        test_loader,
        loss_fn,
        optimizer,
        device,
        args.epochs,
    )
    model_path = Path(__file__).with_name("marko_cifar10.pth")
    torch.save(model.state_dict(), model_path)
    print(f"Model saved to: {model_path}")


if __name__ == "__main__":
    main()
