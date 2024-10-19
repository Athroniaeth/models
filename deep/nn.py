import random

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torchmetrics.classification import Accuracy

from deep.utils import split_dataset, train


class NN(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(NN, self).__init__()
        self.relu = nn.ReLU()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        # Shape {batch_size} {canal} 28x28 -> {batch_size} 768 (28x28)
        out = x.reshape(x.shape[0], -1)

        out = self.fc1(out)
        out = self.relu(out)
        out = self.fc2(out)

        return out


# Set device
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device: {device}")

# Hyperparameters
seed = 42
input_size = 28 * 28
hidden_size = 5
num_classes = 10
learning_rate = 1e-3
batch_size = 256
num_epochs = 15

# Set seed
random.seed(seed)
torch.manual_seed(seed)

# Load Data
train_loader, test_loader, val_loader = split_dataset(
    datasets.MNIST,
    transform=transforms.ToTensor(),
    batch_size=batch_size,
    batch_size_acc=4096,
    shuffle=True,
    download=True,
)

# Initialize network
model = NN(input_size=input_size, hidden_size=hidden_size, num_classes=num_classes).to(device)

# Loss and optimizer
metrics = Accuracy(task="multiclass", num_classes=num_classes).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Train Network
train(
    model=model,
    train_loader=train_loader,
    test_loader=test_loader,
    val_loader=val_loader,
    num_epochs=num_epochs,
    metrics=metrics,
    criterion=criterion,
    optimizer=optimizer,
)
