# Data Directory

This directory manages the raw ESC-50 audio data and processed feature datasets.

## Directory Layout

- `raw/`: Place the original ESC-50 dataset files here (audio files in `.wav` format along with `meta/esc50.csv`).
- `processed/`: Saved preprocessed representations (e.g. extracted MFCC arrays, Mel-spectrogram tensors, sequential feature matrices).

## ESC-50 Dataset Details

- **Source:** [ESC-50 GitHub Repository](https://github.com/karolpiczak/ESC-50)
- **Audio Spec:** 5 seconds long, sampled at 44.1 kHz (resampled as needed during preprocessing).
- **Metadata File:** `esc50.csv` containing columns `filename`, `fold`, `target`, `category`, `esc10`, `src_file`, `take`.

> **Note:** Raw audio files and processed binary feature matrices (`.npy`, `.pt`, etc.) are ignored by git to maintain repository performance.
