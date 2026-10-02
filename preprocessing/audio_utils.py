"""
Audio Utility Functions for ESC-50 Environmental Sound Classification.

Provides functions for loading audio files, resampling to 22,050 Hz,
mono conversion, padding/truncating to fixed 5-second length (110,250 samples),
and sample-wise z-score normalization.
"""

import os
from typing import Tuple
import numpy as np
import librosa


def load_audio(file_path: str, target_sr: int = 22050) -> Tuple[np.ndarray, int]:
    """
    Load an audio file, convert to mono, and resample to the target sampling rate.

    Args:
        file_path (str): Path to the target audio file (.wav).
        target_sr (int): Target sampling rate in Hz (default: 22,050 Hz).

    Returns:
        Tuple[np.ndarray, int]: 1D audio waveform array and target sampling rate.

    Raises:
        FileNotFoundError: If the specified audio file path does not exist.
        RuntimeError: If librosa fails to read/resample the audio file.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Audio file not found at path: {file_path}")

    try:
        waveform, sr = librosa.load(file_path, sr=target_sr, mono=True)
        return waveform, sr
    except Exception as e:
        raise RuntimeError(f"Error loading audio file '{file_path}': {str(e)}")


def pad_or_truncate_audio(waveform: np.ndarray, target_length: int = 110250) -> np.ndarray:
    """
    Ensure the audio waveform has exact target length via zero-padding or truncation.

    For standard ESC-50:
        target_sr = 22,050 Hz
        duration = 5 seconds
        target_length = 22,050 * 5 = 110,250 samples.

    Args:
        waveform (np.ndarray): 1D input audio waveform array.
        target_length (int): Target length in audio samples (default: 110,250).

    Returns:
        np.ndarray: Fixed-length 1D audio array of shape (target_length,).
    """
    current_length = len(waveform)

    if current_length < target_length:
        pad_width = target_length - current_length
        return np.pad(waveform, (0, pad_width), mode='constant', constant_values=0.0)
    elif current_length > target_length:
        return waveform[:target_length]
    
    return waveform


def normalize_waveform(waveform: np.ndarray) -> np.ndarray:
    """
    Apply per-audio sample z-score amplitude normalization.

    Normalization Formula:
        normalized_waveform = (waveform - mean) / (std + 1e-8)

    Rationale & Documentation:
        Per-audio normalization is chosen over global dataset-wide scaling to prevent
        data leakage across cross-validation folds. It standardizes the dynamic range
        and signal power per recording regardless of recording gain or background noise level.

    Args:
        waveform (np.ndarray): Input 1D audio waveform array.

    Returns:
        np.ndarray: Normalized 1D audio array with zero mean and unit variance.
    """
    mean = np.mean(waveform)
    std = np.std(waveform)
    return (waveform - mean) / (std + 1e-8)
