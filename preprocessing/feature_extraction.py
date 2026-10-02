"""
Feature Extraction Utilities for Environmental Sound Classification.

Provides functions to compute acoustic representations for four deep learning models:
1. extract_mfcc: Averaged 1D MFCC vector for MLP (shape: (40,))
2. extract_sequential_mfcc: 2D Frame-level sequence for LSTM (shape: (T, 40))
3. extract_mel_spectrogram: Log-Mel spectrogram for CNN (shape: (128, T))
4. extract_hybrid_spectrogram: Spectrogram representation for CNN-LSTM (shape: (128, T))
"""

import numpy as np
import librosa


def extract_mfcc(
    waveform: np.ndarray,
    sr: int = 22050,
    n_mfcc: int = 40,
    n_fft: int = 2048,
    hop_length: int = 512
) -> np.ndarray:
    """
    Extract aggregated 1D MFCC feature vector for MLP model input.

    Computes frame-level MFCC coefficients and averages across the temporal dimension,
    collapsing time to yield a fixed-size 1D descriptor.

    Args:
        waveform (np.ndarray): 1D audio waveform array.
        sr (int): Sampling rate in Hz (default: 22,050).
        n_mfcc (int): Number of MFCC coefficients to extract (default: 40).
        n_fft (int): FFT window size (default: 2,048).
        hop_length (int): Hop length between frames (default: 512).

    Returns:
        np.ndarray: 1D MFCC vector of shape (n_mfcc,) e.g. (40,).
    """
    mfcc = librosa.feature.mfcc(
        y=waveform,
        sr=sr,
        n_mfcc=n_mfcc,
        n_fft=n_fft,
        hop_length=hop_length
    )
    # Average across time frames (axis 1)
    mfcc_mean = np.mean(mfcc, axis=1)
    return mfcc_mean


def extract_sequential_mfcc(
    waveform: np.ndarray,
    sr: int = 22050,
    n_mfcc: int = 40,
    n_fft: int = 2048,
    hop_length: int = 512
) -> np.ndarray:
    """
    Extract sequential frame-level MFCC feature matrix for LSTM model input.

    Preserves temporal sequence order without averaging across time. Transposes
    the matrix to shape (time_steps, n_mfcc) for PyTorch recurrent neural networks.

    Args:
        waveform (np.ndarray): 1D audio waveform array.
        sr (int): Sampling rate in Hz (default: 22,050).
        n_mfcc (int): Number of MFCC coefficients per frame (default: 40).
        n_fft (int): FFT window size (default: 2,048).
        hop_length (int): Hop length between frames (default: 512).

    Returns:
        np.ndarray: 2D sequence matrix of shape (T, n_mfcc) e.g. (~216, 40).
    """
    mfcc = librosa.feature.mfcc(
        y=waveform,
        sr=sr,
        n_mfcc=n_mfcc,
        n_fft=n_fft,
        hop_length=hop_length
    )
    # Transpose from (n_mfcc, T) to (T, n_mfcc)
    return mfcc.T


def extract_mel_spectrogram(
    waveform: np.ndarray,
    sr: int = 22050,
    n_mels: int = 128,
    n_fft: int = 2048,
    hop_length: int = 512
) -> np.ndarray:
    """
    Extract 2D log-Mel Spectrogram matrix for CNN model input.

    Computes power Mel-spectrogram and applies logarithmic decibel scaling (power_to_db).

    Args:
        waveform (np.ndarray): 1D audio waveform array.
        sr (int): Sampling rate in Hz (default: 22,050).
        n_mels (int): Number of Mel frequency bands (default: 128).
        n_fft (int): FFT window size (default: 2,048).
        hop_length (int): Hop length between frames (default: 512).

    Returns:
        np.ndarray: 2D log-Mel spectrogram matrix of shape (n_mels, T) e.g. (128, ~216).
    """
    mel_spec = librosa.feature.melspectrogram(
        y=waveform,
        sr=sr,
        n_mels=n_mels,
        n_fft=n_fft,
        hop_length=hop_length
    )
    log_mel_spec = librosa.power_to_db(mel_spec, ref=np.max)
    return log_mel_spec


def extract_hybrid_spectrogram(
    waveform: np.ndarray,
    sr: int = 22050,
    n_mels: int = 128,
    n_fft: int = 2048,
    hop_length: int = 512
) -> np.ndarray:
    """
    Extract time-frequency log-Mel spectrogram for CNN-LSTM hybrid model input.

    Architecture Usage & Documentation:
        The CNN-LSTM model receives the time-frequency spectrogram as a 2D/3D tensor (1, n_mels, T).
        Convolutional 2D layers perform local feature extraction across time-frequency regions to
        produce spatial feature maps. These feature maps are subsequently reshaped into a sequence of
        feature vectors across time frames T, which are fed into LSTM recurrent layers for temporal
        sequence modeling.

    Args:
        waveform (np.ndarray): 1D audio waveform array.
        sr (int): Sampling rate in Hz (default: 22,050).
        n_mels (int): Number of Mel frequency bands (default: 128).
        n_fft (int): FFT window size (default: 2,048).
        hop_length (int): Hop length between frames (default: 512).

    Returns:
        np.ndarray: 2D log-Mel spectrogram matrix of shape (n_mels, T) e.g. (128, ~216).
    """
    return extract_mel_spectrogram(
        waveform=waveform,
        sr=sr,
        n_mels=n_mels,
        n_fft=n_fft,
        hop_length=hop_length
    )
