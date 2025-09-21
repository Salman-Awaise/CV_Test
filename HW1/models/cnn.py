"""CNN model definition for color constancy estimation."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Any, Tuple

from utils import setup_logging

logger = setup_logging()

class ColorConstancyCNN(nn.Module):
    def __init__(self, dropout_rate: float = 0.4) -> None:
        super(ColorConstancyCNN, self).__init__()
        # First convolutional block
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)  # 3x3 kernel for better features
        self.bn1 = nn.BatchNorm2d(32)
        self.pool1 = nn.MaxPool2d(4)  # 4x4 pooling
        
        # Second convolutional block
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.pool2 = nn.MaxPool2d(4)  # 4x4 pooling
        
        # Global average pooling to reduce overfitting
        self.global_avg_pool = nn.AdaptiveAvgPool2d(1)
        
        # Fully connected layers
        self.fc1 = nn.Linear(64, 128)
        self.dropout1 = nn.Dropout(dropout_rate)
        self.fc2 = nn.Linear(128, 64)
        self.dropout2 = nn.Dropout(dropout_rate)
        self.fc3 = nn.Linear(64, 3)
        
        # Initialize weights
        self._initialize_weights()
    
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # First conv block
        x = self.pool1(F.relu(self.bn1(self.conv1(x))))
        
        # Second conv block
        x = self.pool2(F.relu(self.bn2(self.conv2(x))))
        
        # Global average pooling
        x = self.global_avg_pool(x)
        x = x.view(x.size(0), -1)
        
        # Fully connected layers with dropout
        x = F.relu(self.fc1(x))
        x = self.dropout1(x)
        x = F.relu(self.fc2(x))
        x = self.dropout2(x)
        x = self.fc3(x)
        
        # Normalize to unit vector
        x = x / (torch.norm(x, dim=1, keepdim=True) + 1e-8)
        return x
