"""
LSTM Neural Network Architecture for ESC-50 Environmental Sound Classification.

Assigned Team Member: Ryan Gupta (Student ID: 230953288)

Architecture Overview:
    1. Input Normalization: BatchNorm1d over the 40 MFCC coefficients.
       Raw MFCCs have very different scales per coefficient (c0 is in the hundreds,
       higher coefficients are close to zero). BatchNorm learns its mean/std only from
       training batches, so no statistics from the held-out fold leak into the model.
       Input shape: (B, T=216, 40) -> (B, 216, 40)
    2. Bidirectional LSTM: 2 stacked layers (hidden_size=128 per direction, batch_first=True).
       The forward direction reads the clip start -> end, the backward direction end -> start,
       so every time step sees context from the whole 5-second clip.
       Output shape: (B, 216, 256)
    3. Temporal Pooling: concatenation of mean pooling and max pooling across time.
       Mean pooling summarises stationary sounds (rain, engine), max pooling keeps short
       events (door knock, dog bark) that would otherwise be averaged away.
       Output shape: (B, 512)
    4. 50-Class Classifier: Linear projection with ReLU and Dropout regularizers -> (B, 50).
"""

import torch
import torch.nn as nn


class ESC50LSTM(nn.Module):
    """
    Bidirectional LSTM Architecture for ESC-50 Environmental Sound Classification.
    """

    def __init__(
        self,
        num_classes: int = 50,
        n_mfcc: int = 40,
        hidden_size: int = 128,
        num_lstm_layers: int = 2,
        bidirectional: bool = True,
        dropout_rate: float = 0.4,
    ) -> None:
        super(ESC50LSTM, self).__init__()

        # 1. Per-coefficient input normalization
        self.input_norm = nn.BatchNorm1d(n_mfcc)

        # 2. LSTM Sequence Modeling
        self.lstm = nn.LSTM(
            input_size=n_mfcc,
            hidden_size=hidden_size,
            num_layers=num_lstm_layers,
            batch_first=True,
            bidirectional=bidirectional,
            dropout=dropout_rate / 2.0 if num_lstm_layers > 1 else 0.0,
        )

        lstm_out_dim = hidden_size * (2 if bidirectional else 1)

        # 3. Classifier Head (input = mean-pool + max-pool features)
        self.classifier = nn.Sequential(
            nn.Dropout(p=dropout_rate),
            nn.Linear(lstm_out_dim * 2, 128),
            nn.ReLU(),
            nn.Dropout(p=dropout_rate / 2.0),
            nn.Linear(128, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x (torch.Tensor): Sequential MFCC input tensor of shape (B, T=216, 40).

        Returns:
            torch.Tensor: Logits output of shape (B, 50).
        """
        # Step 1: Normalize each MFCC coefficient
        # BatchNorm1d expects (B, C, T), so transpose in and back out
        x = self.input_norm(x.transpose(1, 2)).transpose(1, 2)  # -> (B, 216, 40)

        # Step 2: LSTM Temporal Sequence Processing
        lstm_out, _ = self.lstm(x)  # -> (B, 216, 256)

        # Step 3: Mean + Max Temporal Pooling
        mean_pool = torch.mean(lstm_out, dim=1)  # -> (B, 256)
        max_pool, _ = torch.max(lstm_out, dim=1)  # -> (B, 256)
        seq_features = torch.cat([mean_pool, max_pool], dim=1)  # -> (B, 512)

        # Step 4: Classifier Output
        logits = self.classifier(seq_features)  # -> (B, 50)
        return logits
