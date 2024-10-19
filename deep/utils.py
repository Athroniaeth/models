from typing import Callable

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from torchmetrics import Metric


def split_dataset(
    dataset,
    transform: Callable,
    shuffle: bool,
    download: bool,
    batch_size: int,
    batch_size_acc: int,
    train_size: float = 0.6,
    test_size: float = 0.2,
    val_size: float = 0.2,
):
    """
    Split dataset into training, validation and test sets

    Args:
        dataset (Datasets): Dataset to split
        transform (Callable): Transform to apply to the dataset
        shuffle (bool): Shuffle the dataset
        download (bool): Download the dataset
        batch_size (int): Batch size for training
        batch_size_acc (int): Batch size for accuracy
        train_size (float): Size of the training set
        test_size (float): Size of the test set
        val_size (float): Size of the validation set

    Returns:
        tuple: Training, validation and test DataLoaders
    """
    # Download datasets via API
    train_dataset = dataset(root="dataset/", train=True, transform=transform, download=download)
    test_dataset = dataset(root="dataset/", train=False, transform=transform, download=download)

    # Merging training and test datasets
    full_dataset = torch.utils.data.ConcatDataset([train_dataset, test_dataset])

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
