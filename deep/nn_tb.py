import itertools
import random
from functools import partial

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.tensorboard import SummaryWriter
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


def callback_batch_save(
        step: int,
        acc: torch.Tensor,
        loss: torch.Tensor,
        writer: SummaryWriter
):
    writer.add_scalar("Loss", loss, global_step=step)
    writer.add_scalar("Accuracy", acc, global_step=step)


def callback_epoch_save(
        epoch: int,
        epoch_train_acc: torch.Tensor,
        epoch_test_acc: torch.Tensor,
        epoch_loss: torch.Tensor,
        lr: float,
        batch_size: int,
        writer: SummaryWriter
):
    writer.add_hparams(
        {
            "lr": lr,
            "bsize": batch_size
        },
        {
            f"hparams/train_acc": epoch_train_acc.item(),
            f"hparams/test_acc": epoch_test_acc.item(),
            f"hparams/loss": epoch_loss.item(),
        },
    )


# Set device
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device: {device}")

# Model parameters
input_size = 28 * 28
hidden_size = 128
num_classes = 10

# Hyperparameters
seed = 42
num_epochs = 2

# Set seed
random.seed(seed)
torch.manual_seed(seed)

# Test this hyperparameters
batch_sizes = [256, 512] #, 1024, 2048, 4096]
learning_rates = [5e-2, 1e-2, 5e-3,]# 1e-3, 5e-4, 1e-4]

generator = itertools.product(batch_sizes, learning_rates, repeat=1)

for batch_size, learning_rate in generator:
    writer = SummaryWriter(f"runs/MNIST/Batch_{batch_size}_LR_{learning_rate}")

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
        callbacks_batch=[partial(callback_batch_save, writer=writer)],
        callbacks_epoch=[partial(callback_epoch_save, lr=learning_rate, batch_size=batch_size, writer=writer)]
    )

    plot_accuracy(list_train_acc, list_test_acc)
    writer.close()
