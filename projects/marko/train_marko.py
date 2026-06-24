import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

def load_cifar10_data():
    transform = transforms.ToTensor()

    print("Loading CIFAR-10 training data...")
    train_dataset = datasets.CIFAR10(
        root="./data",
        train=True,
        download=False,
        transform=transform
    )

    print("Loading CIFAR-10 test data...")
    test_dataset = datasets.CIFAR10(
        root="./data",
        train=False,
        download=False,
        transform=transform
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=64,
        shuffle=True
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=64,
        shuffle=False
    )
    return train_loader, test_loader

train_loader, test_loader = load_cifar10_data()

images, labels = next(iter(train_loader))
 
print(images.shape)
print(labels.shape)

classes = (
    "airplane",
    "automobile",
    "bird",
    "cat",
    "deer",
    "dog",
    "frog",
    "horse",
    "ship",
    "truck"
)

for i in range(10):
    print(labels[i].item(), classes[labels[i]])
    
image = images[0]
image = image.permute(1, 2, 0)

plt.imshow(image)
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
        x = F.relu(self.conv1(x))
        x = self.pool(x)

        x = F.relu(self.conv2(x))
        x = self.pool(x)

        x = F.relu(self.conv3(x))
        x = self.pool(x)
        
        x = torch.flatten(x, 1) 

        # Dropout is applied before and after the first fully connected layer for stronger regularization.
        x = self.dropout(x)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        return self.fc2(x)
    
epochs = 15

# add the training loop here... i'm burnt out.