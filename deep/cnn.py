import random

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torchmetrics.classification import Accuracy

from deep.utils import split_dataset, train, get_full_dataset, plot_accuracy


class CNN(nn.Module):
    def __init__(self, hidden_size: int = 128, num_classes: int = 10, input_channels: int = 1):
        super(CNN, self).__init__()

        # Activation function and pooling
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # First convolutional layer: input_channels -> 16 output filters, kernel size 3x3
        self.conv1 = nn.Conv2d(in_channels=input_channels, out_channels=16, kernel_size=3, padding=1)

        # Second convolutional layer: 16 -> 32 output filters
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, padding=1)

        # Fully connected layers
        self.fc1 = nn.Linear(32 * 7 * 7, hidden_size)  # Make sure input size matches here
        self.fc2 = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        # Convolution -> ReLU -> MaxPool (Reduces size from 28x28 -> 14x14)
        out = self.conv1(x)
        out = self.relu(out)
        out = self.pool(out)

        # Convolution -> ReLU -> MaxPool (Reduces size from 14x14 -> 7x7)
        out = self.conv2(out)
        out = self.relu(out)
        out = self.pool(out)

        # Flatten for fully connected layers
        out = out.view(out.size(0), -1)  # Flatten (batch_size, 32*7*7)

        # Fully connected layers
        out = self.fc1(out)
        out = self.relu(out)
        out = self.fc2(out)

        return out


# Set device
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device: {device}")

# Model parameters
input_size = 28 * 28
hidden_size = 128
num_classes = 10

# Hyperparameters
seed = 42
learning_rate = 1e-3
batch_size = 256
num_epochs = 15

# Set seed
random.seed(seed)
torch.manual_seed(seed)

# Get full dataset (train and test)
dataset = get_full_dataset(datasets.MNIST, transform=transforms.ToTensor(), download=True)

# Load Data
train_loader, test_loader, val_loader = split_dataset(
    dataset,
    batch_size=batch_size,
    batch_size_acc=8192,
    shuffle=True,
)

# Initialize network
model = CNN(hidden_size=hidden_size, num_classes=num_classes).to(device)

# Loss and optimizer
metrics = Accuracy(task="multiclass", num_classes=num_classes).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Train Network
list_train_acc, list_test_acc = train(
    model=model,
    train_loader=train_loader,
    test_loader=test_loader,
    val_loader=val_loader,
    num_epochs=num_epochs,
    metrics=metrics,
    criterion=criterion,
    optimizer=optimizer,
)

# Plot the accuracy (torchmetrics)
plot_accuracy(list_train_acc, list_test_acc)
