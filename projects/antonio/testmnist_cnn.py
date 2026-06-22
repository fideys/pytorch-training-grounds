import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

# Define device globally
if torch.xpu.is_available():
    device = torch.device("xpu")
elif torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

# Define model class
class CNN(nn.Module):
    def __init__(self):
        super().__init__()
        
        self.dropout = nn.Dropout(0.25)

        # Convolution layers
        self.conv1 = nn.Conv2d(1, 16, 3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
        # new layer (woah)
        self.conv3 = nn.Conv2d(32, 64, 3, padding=1)

        # Fully connected layers
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2)

        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2)

        # new layer but in the forward propagation
        x = F.relu(self.conv3(x))
        
        x = x.view(x.size(0), -1)

        # Dropout is applied before and after the first fully connected layer for stronger regularization.
        x = self.dropout(x)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        return self.fc2(x)

# Define smoothgrad function
def smooth_grad(image, model, device, n_samples=20, noise_level=0.1):
    grads = []

    for _ in range(n_samples):
        noise = torch.randn_like(image) * noise_level
        noisy_image = (image + noise).to(device)
        noisy_image.requires_grad = True

        output = model(noisy_image)
        pred = output.argmax(dim=1).item()

        model.zero_grad()
        output[0, pred].backward()

        grads.append(noisy_image.grad.abs())

    return torch.stack(grads).mean(dim=0)

#   epochs = int(input("epochs amount: "))
epochs = 15

transform = transforms.ToTensor()

def load_mnist_data():
    print("Downloading MNIST training data...")
    train_data = datasets.MNIST(
        root='./data',
        train=True,
        download=True,
        transform=transform
    )

    print("Downloading MNIST test data...")
    test_data = datasets.MNIST(
        root='./data',
        train=False,
        download=True,
        transform=transform
    )

    train_data_loader = DataLoader(
        dataset=train_data,
        batch_size=64,
        shuffle=True,
        num_workers=0,
        pin_memory=device.type in ('cuda', 'xpu')
    )

    test_data_loader = DataLoader(
        dataset=test_data,
        batch_size=64,
        shuffle=False,
        num_workers=0,
        pin_memory=device.type in ('cuda', 'xpu')
    )
    return train_data_loader, test_data_loader

if __name__ == '__main__':
    model = CNN()
    model = model.to(device)

    print(f"Using device: {device}")



# Load MNIST data loaders
    train_data_loader, test_data_loader = load_mnist_data()

    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)


    # Training loop
    losses = []
    def train_model():
        for epoch in range(epochs):
            model.train()
            epoch_loss = 0

            for images, labels in train_data_loader:
                images = images.to(device)
                labels = labels.to(device)

                outputs = model(images)
                loss = loss_fn(outputs, labels)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                epoch_loss += loss.item()

            avg_loss = epoch_loss / len(train_data_loader)
            losses.append(avg_loss)
            print(f"Epoch {epoch+1}/{epochs} completed - Loss: {avg_loss:.4f}")

# See test accuracy
    def test_and_save():
        correct = 0
        total = 0

        model.eval()
    
        with torch.no_grad():
            for images, labels in test_data_loader:
                images = images.to(device)
                labels = labels.to(device)
                outputs = model(images)
                predictions = outputs.argmax(dim=1)
                correct += (predictions == labels).sum().item()
                total += labels.size(0)

        accuracy = correct / total
        print(f"Test Accuracy: {accuracy * 100:.2f}%")

        torch.save(model.state_dict(), "mnist_cnn.pth")
        print("Model saved as mnist_cnn.pth")
    
    train_model()
    test_and_save()