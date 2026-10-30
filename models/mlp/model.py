"""Multilayer perceptron for ESC-50 environmental sound classification.

Assigned team member: Mridul Arora (registration number 230953330).
The network consumes one 40-dimensional, time-averaged MFCC vector per clip
and returns 50 unnormalized class logits for use with CrossEntropyLoss.
"""

import torch
from torch import nn


class ESC50MLP(nn.Module):
    """Fully connected ESC-50 classifier for fixed-length MFCC vectors."""

    def __init__(
        self,
        input_dim: int = 40,
        num_classes: int = 50,
        dropout_rate: float = 0.4,
    ) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, num_classes),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """Return raw class logits with shape ``(batch_size, 50)``."""
        if features.ndim != 2 or features.shape[1] != 40:
            raise ValueError(
                "ESC50MLP expects input shape (batch_size, 40), "
                f"received {tuple(features.shape)}"
            )
        return self.network(features)
