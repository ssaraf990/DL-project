# Preprocessing Module

This module houses all audio pre-processing and feature extraction tools required for the 4 model architectures.

## Sub-modules Overview

- `audio_utils.py`: Audio loading, sampling rate conversions, channel reduction, padding, and truncation.
- `feature_extraction.py`: Acoustic feature extraction routines:
  - **MFCC:** Global aggregated MFCC vector generation (MLP model).
  - **Mel-spectrogram:** 2D time-frequency log-Mel spectrogram generation (CNN model).
  - **Sequential MFCC:** Frame-level MFCC feature sequence matrices (LSTM model).
  - **Hybrid Spectrogram:** High-resolution time-frequency maps (CNN-LSTM model).
- `dataset_loader.py`: Dataset indexing, fold splitting (ESC-50 5-fold scheme), and PyTorch DataLoader integration.
