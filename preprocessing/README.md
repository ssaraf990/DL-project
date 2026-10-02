# Preprocessing & Infrastructure Module

This module houses the common audio pre-processing, reproducibility controls, feature extraction algorithms, and dataset loaders for the ESC-50 Deep Learning project.

---

## Technical Specifications

### 1. Audio Utilities (`audio_utils.py`)
- **Sampling Rate:** All audio recordings are converted to single-channel (mono) and resampled to **22,050 Hz**.
- **Fixed Duration:** 5.0 seconds per audio clip.
- **Waveform Target Length:** Exactly **110,250 samples** ($22,050 \times 5$).
- **Padding & Truncation:** Signals shorter than 5 seconds are zero-padded; longer signals are truncated.
- **Amplitude Normalization:** Sample-wise z-score normalization: $(x - \mu) / (\sigma + 1e-8)$. Performed per recording to prevent global dataset data leakage.

### 2. Reproducibility Control (`reproducibility.py`)
- `set_seed(seed=42)` initializes seeds across Python `random`, `numpy`, `torch`, `torch.cuda`, and sets CuDNN to deterministic execution mode (`deterministic = True`, `benchmark = False`).

---

## Model Feature Representations & Shapes

| Model | Feature Type | Extract Function | Output Tensor Shape | Model Consumption Description |
| :--- | :--- | :--- | :--- | :--- |
| **MLP** | `"mfcc"` | `extract_mfcc` | `torch.Size([40])` | 1D global mean-aggregated MFCC feature vector fed into dense Linear layers. |
| **CNN** | `"mel_spectrogram"` | `extract_mel_spectrogram` | `torch.Size([1, 128, T])` | 2D log-Mel spectrogram image ($128 \times T$) with single channel dimension, processed by 2D Convolutional layers (`nn.Conv2d`). |
| **LSTM** | `"seq_mfcc"` | `extract_sequential_mfcc` | `torch.Size([T, 40])` | Frame-level sequence of MFCC feature vectors across $T \approx 216$ time steps, processed by recurrent LSTM layers (`batch_first=True`). |
| **CNN-LSTM** | `"hybrid"` | `extract_hybrid_spectrogram` | `torch.Size([1, 128, T])` | 2D log-Mel spectrogram processed by CNN spatial feature maps, reshaped into frame sequence $(T, C)$ for LSTM temporal sequence modeling. |

*Note: For $22,050\text{ Hz}$ sampling rate, $110,250\text{ samples}$, and `hop_length=512`, the temporal dimension $T = \lfloor 110250 / 512 \rfloor + 1 = 216$ frames.*

---

## Dataset Loading & Fold Handling (`dataset_loader.py`)

- **Class Labels:** 50 distinct environmental sound classes mapped to integer labels $0$ through $49$.
- **ESC-50 Official 5-Fold Scheme:**
  - `val_fold=k` (where $k \in \{1, 2, 3, 4, 5\}$).
  - **Training Split (`is_train=True`):** Selects all samples where `fold != k` (1,600 audio samples).
  - **Evaluation Split (`is_train=False`):** Selects all samples where `fold == k` (400 audio samples).
  - **Leakage Prevention:** Grouping clips by the official `fold` column prevents clips from the same original recording session (`src_file`) from appearing in both training and test sets.
