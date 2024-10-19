import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchmetrics import Metric


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
