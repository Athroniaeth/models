import random

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch import Tensor
from torch.utils.data import Dataset

from deep.utils import split_dataset, train, get_full_dataset, plot_accuracy


class AutoEncoder(nn.Module):
    """
    Simple Autoencoder with a single hidden layer for encoding and decoding.

    Args:
        input_size (int): The size of the flattened input image (e.g. 28*28 for MNIST).
        hidden_size (int): The size of the bottleneck layer.
    """

    def __init__(self, input_size: int, hidden_size: int):
        super(AutoEncoder, self).__init__()

        # Encoder of input
        self.encoder = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU()
        )

        # Decoder of latent representation
        self.decoder = nn.Sequential(
            nn.Linear(hidden_size, input_size),
            nn.ReLU()
        )

    def forward(self, x):
        """
        Forward pass through the autoencoder. The input is first encoded, then decoded.

        Args:
            x (Tensor): Input tensor with shape (batch_size, 1, 28, 28).

        Returns:
            Tensor: Reconstructed output with shape (batch_size, 28*28).
        """
        # Encoding the input
        encoded = self.encoder(x)

        # Decoding the latent representation
        decoded = self.decoder(encoded)

        return decoded


class AEDataset(Dataset):
    def __init__(self, dataset):
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        image, _ = self.dataset[idx]  # Ignore the label

        # Flatten the input from (batch_size, 1, 28, 28) -> (batch_size, 28*28)
        image = image.view(-1)

        return image, image


# Set device
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device: {device}")

# Model hyperparameters
input_size = 28 * 28
hidden_size = 784

# Hyperparameters
seed = 42
learning_rate = 1e-3
batch_size = 256
num_epochs = 15

# Set seed
random.seed(seed)
torch.manual_seed(seed)

# Convert classification dataset to Autoencoder dataset
full_dataset = get_full_dataset(datasets.MNIST, transform=transforms.ToTensor(), download=True)
full_dataset = AEDataset(full_dataset)

# Load Data (train, test, validation)
train_loader, test_loader, val_loader = split_dataset(
    full_dataset=full_dataset,
    batch_size=batch_size,
    batch_size_acc=8192,
    shuffle=True,
)

# Initialize the Autoencoder model
model = AutoEncoder(input_size=input_size, hidden_size=hidden_size).to(device)


def metrics(x: Tensor, y: Tensor, epsilon: float = 1e-12) -> Tensor:
    # Want ignore black pixel
    mask = y > 0
    x, y = x[mask], y[mask]

    # Get the percentage error
    output = (x + epsilon) / (y + epsilon)  # epsilon to avoid division by zero

    # Inverse bad calcul percentage (max/min = +1, min/max = 0 - 0.999...)
    output = torch.where(output >= 1, 1 / output, output)

    # Return the mean of the percentage error
    return output.mean()


criterion = nn.MSELoss()
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

# Show the first image and its reconstruction
import matplotlib.pyplot as plt

image, _ = full_dataset[0]
image = image.view(1, -1).to(device)

plt.subplot(1, 3, 1)
plt.title("Original Image")
plt.imshow(image.cpu().view(28, 28), cmap="gray")

reconstructed = model(image)
image_reconstructed = reconstructed.cpu().detach().view(28, 28)

plt.subplot(1, 3, 2)
plt.title("Reconstructed Image")
plt.imshow(image_reconstructed, cmap="gray")

# Repass construct image to model (3 third image)
reconstructed_2 = model(reconstructed)
image_reconstructed_2 = reconstructed_2.cpu().detach().view(28, 28)

plt.subplot(1, 3, 3)
plt.title("Reconstructed Image (2)")
plt.imshow(image_reconstructed_2, cmap="gray")

plt.show()
