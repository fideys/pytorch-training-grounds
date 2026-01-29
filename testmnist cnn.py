import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt


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
    shuffle=True
)

test_data_loader = DataLoader(
    dataset=test_data,
    batch_size=64,
    shuffle=False
)

# define the model

class CNN(nn.Module):
    def __init__(self):
        super().__init__()

        # Convolution layers
        self.conv1 = nn.Conv2d(1, 16, 3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, 3, padding=1)

        # Fully connected layers
        self.fc1 = nn.Linear(32 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2)

        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2)

        x = x.view(x.size(0), -1)

        x = F.relu(self.fc1(x))
        return self.fc2(x)

model = CNN()

loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# Training loop
losses = []

for epoch in range(5):
    epoch_loss = 0

    for images, labels in train_data_loader:
        
        outputs = model(images)
        loss = loss_fn(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()

    avg_loss = epoch_loss / len(train_data_loader)
    losses.append(avg_loss)
    print(f"Epoch {epoch+1}/5 completed - Loss: {avg_loss:.4f}")

# Plot training loss
plt.plot(losses)
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Training Loss")
plt.show()


model.eval()

images, labels = next(iter(test_data_loader))

image = images[0].unsqueeze(0)  # Add batch dimension: [1, 28, 28] -> [1, 1, 28, 28]
label = labels[0]

image.requires_grad = True

output = model(image)

predicted_class = output.argmax(dim=1)
print("Predicted:", predicted_class.item())
model.zero_grad()
output[0, predicted_class].backward()

saliency = image.grad.abs()

saliency = saliency.view(28, 28).detach()

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



print(f"Final image shape: {images.shape}")
print(f"Final label shape: {labels.shape}")