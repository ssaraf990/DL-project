"""
ESC-50 Dataset Loader Implementation.

Provides PyTorch Dataset implementation for loading ESC-50 audio recordings,
parsing CSV metadata, enforcing official 5-fold cross-validation splits,
and extracting features for MLP, CNN, LSTM, and CNN-LSTM architectures.
"""

import os
from typing import Optional, Callable, Tuple
import pandas as pd
import torch
from torch.utils.data import Dataset
import numpy as np

from .audio_utils import load_audio, pad_or_truncate_audio, normalize_waveform
from .feature_extraction import (
    extract_mfcc,
    extract_sequential_mfcc,
    extract_mel_spectrogram,
    extract_hybrid_spectrogram,
)


class ESC50Dataset(Dataset):
    """
    PyTorch Dataset wrapper for the ESC-50 Environmental Sound Classification dataset.

    Fold Splitting & Leakage Prevention:
        ESC-50 defines an official 5-fold cross-validation scheme. Clips from the same original
        recording session (src_file) are grouped into the same fold.
        - When is_train=True and val_fold=k (1 <= k <= 5), samples from fold != k are loaded for training.
        - When is_train=False and val_fold=k, samples from fold == k are loaded for validation/testing.
        This strict partition guarantees zero data overlap between training and evaluation splits.
    """

    SUPPORTED_FEATURES = {"mfcc", "mel_spectrogram", "seq_mfcc", "hybrid"}

    def __init__(
        self,
        meta_csv: str,
        audio_dir: str,
        val_fold: Optional[int] = None,
        is_train: bool = True,
        feature_type: str = "mel_spectrogram",
        target_sr: int = 22050,
        target_length: int = 110250,
        transform: Optional[Callable] = None,
    ) -> None:
        """
        Args:
            meta_csv (str): Path to esc50.csv metadata file.
            audio_dir (str): Path to raw audio directory containing .wav files.
            val_fold (Optional[int]): Held-out validation fold index (1 to 5).
                                      If None, uses all dataset rows.
            is_train (bool): If True and val_fold is set, selects training folds (fold != val_fold).
                             If False and val_fold is set, selects evaluation fold (fold == val_fold).
            feature_type (str): Feature extraction algorithm: 'mfcc', 'mel_spectrogram', 'seq_mfcc', 'hybrid'.
            target_sr (int): Target sampling rate in Hz (default: 22,050 Hz).
            target_length (int): Fixed waveform length in samples (default: 110,250 samples for 5s).
            transform (Optional[Callable]): Optional transformation function applied to feature tensor.
        """
        if feature_type not in self.SUPPORTED_FEATURES:
            raise ValueError(
                f"Invalid feature_type '{feature_type}'. Supported feature types: {self.SUPPORTED_FEATURES}"
            )

        if not os.path.exists(meta_csv):
            raise FileNotFoundError(f"Metadata CSV file not found at: {meta_csv}")

        self.meta_csv = meta_csv
        self.audio_dir = audio_dir
        self.val_fold = val_fold
        self.is_train = is_train
        self.feature_type = feature_type
        self.target_sr = target_sr
        self.target_length = target_length
        self.transform = transform

        # Parse CSV metadata
        df = pd.read_csv(meta_csv)
        required_cols = {"filename", "fold", "target", "category", "src_file", "take"}
        if not required_cols.issubset(set(df.columns)):
            missing = required_cols - set(df.columns)
            raise ValueError(f"Metadata CSV is missing required columns: {missing}")

        # Filter by fold
        if val_fold is not None:
            if not (1 <= val_fold <= 5):
                raise ValueError(f"val_fold must be between 1 and 5 inclusive, got: {val_fold}")
            
            if is_train:
                self.df = df[df["fold"] != val_fold].reset_index(drop=True)
            else:
                self.df = df[df["fold"] == val_fold].reset_index(drop=True)
        else:
            self.df = df.reset_index(drop=True)

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        row = self.df.iloc[idx]
        filename = row["filename"]
        target = int(row["target"])  # Class label index 0 to 49

        file_path = os.path.join(self.audio_dir, filename)

        # 1. Load audio signal
        waveform, _ = load_audio(file_path, target_sr=self.target_sr)

        # 2. Enforce fixed length (5s -> 110,250 samples)
        waveform = pad_or_truncate_audio(waveform, target_length=self.target_length)

        # 3. Apply sample z-score normalization
        waveform = normalize_waveform(waveform)

        # 4. Extract feature representation
        if self.feature_type == "mfcc":
            # Output shape: (40,) for MLP
            feature = extract_mfcc(waveform, sr=self.target_sr)
            tensor_feature = torch.tensor(feature, dtype=torch.float32)

        elif self.feature_type == "mel_spectrogram":
            # Output shape: (128, T) -> add channel dimension -> (1, 128, T) for CNN
            feature = extract_mel_spectrogram(waveform, sr=self.target_sr)
            tensor_feature = torch.tensor(feature, dtype=torch.float32).unsqueeze(0)

        elif self.feature_type == "seq_mfcc":
            # Output shape: (T, 40) for LSTM
            feature = extract_sequential_mfcc(waveform, sr=self.target_sr)
            tensor_feature = torch.tensor(feature, dtype=torch.float32)

        elif self.feature_type == "hybrid":
            # Output shape: (128, T) -> add channel dimension -> (1, 128, T) for CNN-LSTM
            feature = extract_hybrid_spectrogram(waveform, sr=self.target_sr)
            tensor_feature = torch.tensor(feature, dtype=torch.float32).unsqueeze(0)

        else:
            raise ValueError(f"Unsupported feature type: {self.feature_type}")

        if self.transform is not None:
            tensor_feature = self.transform(tensor_feature)

        return tensor_feature, target
