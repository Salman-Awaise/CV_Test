"""
CNN training loop and loss function for color constancy.
"""

import torch
import torch.nn as nn
from typing import Any

def angular_error_loss(pred_illuminant: torch.Tensor, true_illuminant: torch.Tensor) -> torch.Tensor:
    """
    Compute angular error loss in degrees.
    """
    dot = torch.sum(pred_illuminant * true_illuminant, dim=1)
    dot = torch.clamp(dot, -1.0, 1.0)
    angle = torch.acos(dot)
    angle_deg = angle * 180.0 / torch.pi
    return torch.mean(angle_deg)

def euclidean_loss(pred_illuminant: torch.Tensor, true_illuminant: torch.Tensor) -> torch.Tensor:
    """
    Compute mean Euclidean (L2) loss between predicted and true illuminant.
    """
    return torch.mean(torch.norm(pred_illuminant - true_illuminant, dim=1))


def train_model(model: nn.Module, dataloader: Any, epochs: int = 50, val_dataloader: Any = None, loss_fn=angular_error_loss) -> None:
    """
    Standard PyTorch training loop.
    Save best model based on validation angular error.
    Use Adam optimizer, lr=0.001 initially.
    """
    import torch.optim as optim
    best_val_error = float('inf')
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    for epoch in range(epochs):
        model.train()
        for batch in dataloader:
            images, gt = batch
            optimizer.zero_grad()
            pred = model(images)
            loss = loss_fn(pred, gt)
            loss.backward()
            optimizer.step()
        if val_dataloader:
            model.eval()
            val_errors = []
            with torch.no_grad():
                for val_batch in val_dataloader:
                    val_images, val_gt = val_batch
                    val_pred = model(val_images)
                    val_error = loss_fn(val_pred, val_gt)
                    val_errors.append(val_error.item())
            mean_val_error = sum(val_errors) / len(val_errors)
            if mean_val_error < best_val_error:
                best_val_error = mean_val_error
                torch.save(model.state_dict(), 'best_model.pth')
