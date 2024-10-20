import random

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torchmetrics.classification import Accuracy

from deep.utils import split_dataset, train, get_full_dataset, plot_accuracy


class RNN(nn.Module):
    def __init__(
            self,
            input_size: int,
            hidden_size: int,
            num_layers: int,
            num_classes: int,
            sequence_length: int = 28
    ):
        super(RNN, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.rnn = nn.RNN(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size*sequence_length, num_classes)

    def forward(self, x):
        # Remove channel dimension
        x = x.squeeze(1)

        # Set initial hidden and cell states
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)

        # Forward propagate RNN
        out, _ = self.rnn(x, h0)
        out = out.reshape(out.shape[0], -1)
        out = self.fc(out)
        return out


# Set device
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device: {device}")

# Model parameters
input_size = 28
num_layers = 2
hidden_size = 256
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
model = RNN(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers, num_classes=num_classes).to(device)

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
