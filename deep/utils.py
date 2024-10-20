import contextlib
from typing import Callable, List

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from torch import optim
from torch.utils.data import DataLoader, random_split
from torchmetrics import Metric
from tqdm import tqdm


def get_full_dataset(
        dataset: Callable,
        transform: Callable,
        download: bool,
):
    """
    Get full dataset with both training and test data

    Args:
        dataset (Callable): Dataset to download
        transform (Callable): Transformation to apply to the dataset
        download (bool): Download the dataset

    Returns:
        torch.utils.data.ConcatDataset: Full dataset with both training and test data
    """
    # Download datasets via API
    train_dataset = dataset(root="dataset/", train=True, transform=transform, download=download)
    test_dataset = dataset(root="dataset/", train=False, transform=transform, download=download)

    # Merging training and test datasets
    full_dataset = torch.utils.data.ConcatDataset([train_dataset, test_dataset])

    return full_dataset


def split_dataset(
        full_dataset,
        shuffle: bool,
        batch_size: int,
        batch_size_acc: int,
        train_size: float = 0.6,
        test_size: float = 0.2,
        val_size: float = 0.2,
):
    """
    Split dataset into training, validation and test sets

    Args:
        full_dataset (Datasets): Dataset to split
        shuffle (bool): Shuffle the dataset
        batch_size (int): Batch size for training
        batch_size_acc (int): Batch size for accuracy
        train_size (float): Size of the training set
        test_size (float): Size of the test set
        val_size (float): Size of the validation set

    Returns:
        tuple: Training, validation and test DataLoaders
    """
    # Define partition sizes of the dataset
    length = len(full_dataset)
    train_size = int(train_size * length)
    test_size = int(test_size * length)
    val_size = int(val_size * length)

    # Random split of train, val and test data
    train_dataset, val_dataset, test_dataset = random_split(full_dataset, [train_size, val_size, test_size])

    # Creating DataLoaders for each partition
    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=shuffle)
    test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size_acc, shuffle=shuffle)
    val_loader = DataLoader(dataset=val_dataset, batch_size=batch_size_acc, shuffle=shuffle)

    return train_loader, val_loader, test_loader


def check_accuracy(loader: DataLoader, model: nn.Module, metrics: Metric):
    """
    Check accuracy of model on data in loader

    Args:
        loader (DataLoader): DataLoader for the data to check accuracy
        model (nn.Module): Model to check accuracy
        metrics (Metric): Metric to use for accuracy

    Returns:
        float: Accuracy of the model on the data in loader
    """
    acc = 0
    model.eval()  # Set model to evaluation mode
    device = next(model.parameters()).device  # Get model device

    # Don't calculate gradients
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device=device)
            y = y.to(device=device)

            scores = model(x)
            acc += metrics(scores, y)

    model.train()  # Set model back to train mode
    return acc / len(loader)


def train(
        model: nn.Module,
        train_loader: DataLoader,
        test_loader: DataLoader,
        val_loader: DataLoader,
        num_epochs: int,
        metrics: Metric,
        criterion: nn.Module,
        optimizer: optim.Optimizer,
):
    """
    Complete function to train a model

    Args:
        model (nn.Module): Model to train
        train_loader (DataLoader): DataLoader for training data
        test_loader (DataLoader): DataLoader for test data
        val_loader (DataLoader): DataLoader for validation data
        num_epochs (int): Number of epochs to train
        metrics (Metric): Metric to use for accuracy
        criterion (nn.Module): Loss function
        optimizer (optim.Optimizer): Optimizer to use for training

    Returns:
        float: Accuracy of the model on the test data
    """
    list_train_acc = []
    list_test_acc = []

    # Get device of model
    device = next(model.parameters()).device

    # Allow interruption of training
    with contextlib.suppress(KeyboardInterrupt):
        for epoch in range(num_epochs):
            total_accuracy = 0
            pbar = tqdm(train_loader, total=len(train_loader), desc=f"Epoch {epoch + 1}/{num_epochs}")

            for batch_idx, (data, targets) in enumerate(pbar):
                # Get data to cuda if possible
                data = data.to(device=device)
                targets = targets.to(device=device)

                # Forward step
                scores = model(data)
                acc = metrics(scores, targets)
                loss = criterion(scores, targets)

                # Backward step
                optimizer.zero_grad()
                loss.backward()

                # Gradient descent or Adam step
                optimizer.step()
                total_accuracy += acc

            # Check test accuracy after each epoch
            train_acc = total_accuracy / len(train_loader)
            test_acc = check_accuracy(test_loader, model, metrics)

            # Update progress bar
            pbar.write(f"Epoch {epoch}, Train: {train_acc * 100:.2f}%, Test: {test_acc * 100:.2f}%")

            # Save accuracy for each epoch
            list_train_acc.append(train_acc)
            list_test_acc.append(test_acc)

    # Check validation accuracy after training
    val_acc = check_accuracy(val_loader, model, metrics)
    pbar.write(f"Validation accuracy: {val_acc * 100:.2f}%")

    return list_train_acc, list_test_acc


def plot_accuracy(train_acc_list: List[torch.Tensor], test_acc_list: List[torch.Tensor]):
    """
    Plot the training and testing accuracy over epochs.

    Plots the training and testing accuracy over epochs
    on a single graph. Epoch numbers will be displayed
    as integers. The function will display the plot.

    Parameters:
        train_acc_list (List[torch.Tensor]): List of training accuracy values for each epoch.
        test_acc_list (List[torch.Tensor]): List of testing accuracy values for each epoch.

    """

    # Convert the tensors to numpy arrays (move to CPU first if on GPU)
    train_acc_cpu = [acc.cpu().numpy() for acc in train_acc_list]
    test_acc_cpu = [acc.cpu().numpy() for acc in test_acc_list]

    # Create the plot
    plt.figure(figsize=(10, 6))

    # Plot training accuracy
    plt.plot(train_acc_cpu, label="Training Accuracy", color='orange')

    # Plot testing accuracy
    plt.plot(test_acc_cpu, label="Testing Accuracy", color='blue')

    # Add labels and title
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.title("Training and Testing Accuracy Over Epochs")

    # Set the ticks on the x-axis to be integers (epochs)
    epochs = np.arange(1, len(train_acc_cpu))  # Create an array of epochs (1, 2, ..., N)
    plt.xticks(epochs)  # Force x-axis ticks to be integers

    # Show legend
    plt.legend()

    # Display the plot
    plt.show()
