"""
CNN-LSTM Hybrid Neural Network Architecture for ESC-50 Environmental Sound Classification.

Assigned Team Member: Karan Shah (Student ID: 230953540)

Architecture Overview:
    1. CNN Feature Extractor: 3 Convolutional blocks with BatchNorm, ReLU activations, and MaxPool2d layers.
       Input shape: (B, 1, 128, 216) -> Output shape: (B, 128, 16, 27)
    2. CNN-to-LSTM Tensor Transformation:
       Permutes and flattens CNN spatial-frequency channels to convert 2D feature maps into a sequential time-series representation.
       Shape transformation: (B, 128, 16, 27) -> Permute (B, 27, 128, 16) -> Reshape (B, 27, 2048),
       where T=27 sequence steps and feature dimension per step is 128*16 = 2048.
    3. LSTM Temporal Modeling: 2-layer Bidirectional/Unidirectional LSTM (hidden_size=128, batch_first=True).
       Output shape: (B, 27, 128) -> Temporal Mean Pooling -> (B, 128)
    4. 50-Class Classifier: Linear projection with ReLU and Dropout regularizers -> (B, 50).
"""

import torch
import torch.nn as nn


class ESC50CNNLSTM(nn.Module):
    """
    CNN-LSTM Hybrid Architecture for ESC-50 Environmental Sound Classification.
    """

    def __init__(self, num_classes: int = 50, hidden_size: int = 128, num_lstm_layers: int = 2, dropout_rate: float = 0.4) -> None:
        super(ESC50CNNLSTM, self).__init__()

        # 1. CNN Feature Extraction Blocks
        # Input: (B, 1, 128, 216) -> (B, 32, 64, 108)
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        # (B, 32, 64, 108) -> (B, 64, 32, 54)
        self.conv2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        # (B, 64, 32, 54) -> (B, 128, 16, 27)
        self.conv3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        # 2. LSTM Sequence Modeling
        # CNN output feature dimension per time step = 128 channels * 16 frequency bands = 2048
        cnn_feature_dim = 128 * 16
        self.lstm = nn.LSTM(
            input_size=cnn_feature_dim,
            hidden_size=hidden_size,
            num_layers=num_lstm_layers,
            batch_first=True,
            dropout=dropout_rate / 2.0 if num_lstm_layers > 1 else 0.0
        )

        # 3. Classifier Head
        self.classifier = nn.Sequential(
            nn.Dropout(p=dropout_rate),
            nn.Linear(hidden_size, 64),
            nn.ReLU(),
            nn.Dropout(p=dropout_rate / 2.0),
            nn.Linear(64, num_classes)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x (torch.Tensor): Log-Mel spectrogram input tensor of shape (B, 1, 128, 216).

        Returns:
            torch.Tensor: Logits output of shape (B, 50).
        """
        # Step 1: CNN Spatial-Frequency Extraction
        # Input: (B, 1, 128, 216)
        x = self.conv1(x)  # -> (B, 32, 64, 108)
        x = self.conv2(x)  # -> (B, 64, 32, 54)
        x = self.conv3(x)  # -> (B, 128, 16, 27)

        # Step 2: CNN-to-LSTM Tensor Transformation
        # Current shape: (B, Channels=128, Freq=16, Time=27)
        B, C, F, T = x.shape
        # Permute time to 2nd dimension: (B, Time=27, Channels=128, Freq=16)
        x = x.permute(0, 3, 1, 2)
        # Flatten spatial/frequency features per time step: (B, T=27, C*F=2048)
        x = x.reshape(B, T, C * F)

        # Step 3: LSTM Temporal Sequence Processing
        # Output shape: (B, T=27, Hidden=128)
        lstm_out, _ = self.lstm(x)

        # Temporal Mean Pooling across sequence length T=27
        # Shape: (B, Hidden=128)
        seq_features = torch.mean(lstm_out, dim=1)

        # Step 4: Classifier Output
        # Logits shape: (B, 50)
        logits = self.classifier(seq_features)
        return logits
