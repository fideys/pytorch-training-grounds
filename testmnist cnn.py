import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

if torch.xpu.is_available():
    device = torch.device("xpu")
elif torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")
    

transform = transforms.ToTensor()

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
    num_workers=0 if device.type in ('cpu', 'xpu') else 4,  # Reduce workers for CPU
    pin_memory=device.type in ('cuda', 'xpu') # Only use pin_memory with GPU or XPU
)

test_data_loader = DataLoader(
    dataset=test_data,
    batch_size=64,
    shuffle=False,
    num_workers=0 if device.type in ('cpu', 'xpu') else 2,  # Reduce workers for CPU
    pin_memory=device.type in ('cuda', 'xpu') # Only use pin_memory with GPU or XPU
)

# define the model

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
model = CNN()
model = model.to(device)  # Fixed: reassign the model

print(f"Using device: {device}")  # Add this to confirm device

loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# Training loop
losses = []

for epoch in range(5):
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
    print(f"Epoch {epoch+1}/5 completed - Loss: {avg_loss:.4f}")
    
# training loop ends here
# define smoothgrad
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

# see test accuracy
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


# # Plot training loss
# plt.plot(losses)
# plt.xlabel("Epoch")
# plt.ylabel("Loss")
# plt.title("Training Loss")
# plt.show()

# Get a test image for saliency visualization
images, labels = next(iter(test_data_loader))
image = images[0].unsqueeze(0)  # Add batch dimension: [1, 28, 28] -> [1, 1, 28, 28]
label = labels[0]
model.eval()

saliency = smooth_grad(image, model, device)
saliency = saliency.view(28, 28).detach().cpu()
image = image.cpu()

plt.figure(figsize=(6,3))

plt.subplot(1,2,1)
plt.imshow(image.detach().view(1, 28, 28)[0], cmap="gray")
plt.title("Input")
plt.axis("off")

plt.subplot(1,2,2)
plt.imshow(saliency, cmap="hot")
plt.title("Pixel importance")
plt.axis("off")

plt.show()