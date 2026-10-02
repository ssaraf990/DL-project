"""
2D Convolutional Neural Network (CNN) Architecture for ESC-50 Environmental Sound Classification.

Assigned Team Member: Satyam Saraf (Student ID: 230953178)

Architecture Overview:
    Input: Log-Mel Spectrogram tensor of shape (Batch, 1, 128, 216).
    Backbone: 4 Convolutional blocks with BatchNorm, ReLU activations, and MaxPool2d layers.
    Global Pooling: AdaptiveAvgPool2d((1, 1)) to condense spatial/temporal dimensions.
    Classifier: Fully connected layers with Dropout for regularization, outputting 50 class logits.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ESC50CNN(nn.Module):
    """
    2D CNN model for ESC-50 Environmental Sound Classification.
    """

    def __init__(self, num_classes: int = 50, dropout_rate: float = 0.4) -> None:
        super(ESC50CNN, self).__init__()

        # Conv Block 1: (1, 128, 216) -> (32, 64, 108)
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        # Conv Block 2: (32, 64, 108) -> (64, 32, 54)
        self.conv2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        # Conv Block 3: (64, 32, 54) -> (128, 16, 27)
        self.conv3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        # Conv Block 4: (128, 16, 27) -> (256, 8, 13)
        self.conv4 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        # Global Average Pooling
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))

        # FC Classifier
        self.classifier = nn.Sequential(
            nn.Dropout(p=dropout_rate),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(p=dropout_rate / 2.0),
            nn.Linear(128, num_classes)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x (torch.Tensor): Input log-Mel spectrogram tensor of shape (B, 1, 128, T).

        Returns:
            torch.Tensor: Logits tensor of shape (B, 50).
        """
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.global_pool(x)
        x = torch.flatten(x, 1)
        logits = self.classifier(x)
        return logits
