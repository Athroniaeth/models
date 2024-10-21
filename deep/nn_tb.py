import itertools
import random

import mlflow
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torchmetrics.classification import Accuracy

from deep.utils import split_dataset, get_full_dataset, train, plot_accuracy


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

# Model parameters
input_size = 28 * 28
hidden_size = 8
num_classes = 10

# Hyperparameters
seed = 42
num_epochs = 5

# Set seed
random.seed(seed)
torch.manual_seed(seed)

# Test this hyperparameters
batch_sizes = [256, 512, 1024]
learning_rates = [1e-2, 1e-3, 1e-4]

generator = itertools.product(batch_sizes, learning_rates, repeat=1)

mlflow.set_experiment("/neural_networks_mnist")

for batch_size, learning_rate in generator:
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
    model = NN(input_size=input_size, hidden_size=hidden_size, num_classes=num_classes).to(device)

    # Loss and optimizer
    metrics = Accuracy(task="multiclass", num_classes=num_classes).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    with mlflow.start_run():
        # Train Network (with tensorboard)
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
