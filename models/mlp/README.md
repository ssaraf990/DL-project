# MLP Model

**Assigned team member:** Mridul Arora  
**Registration number:** 230953330  
**Input feature:** 40 time-averaged MFCC coefficients

`ESC50MLP` is a PyTorch classifier for the 50 ESC-50 categories. It accepts a
batch shaped `(N, 40)` and returns raw logits shaped `(N, 50)`.

## Architecture

1. Linear 40 → 256, BatchNorm, ReLU, Dropout(0.4)
2. Linear 256 → 128, BatchNorm, ReLU, Dropout(0.4)
3. Linear 128 → 64, ReLU
4. Linear 64 → 50

The training pipeline uses `CrossEntropyLoss`, so the model deliberately does
not apply softmax in `forward`.

## Features and normalization

The shared preprocessing code loads each clip at 22,050 Hz, pads or truncates
it to five seconds, applies waveform z-score normalization, and extracts 40
MFCCs with `n_fft=2048` and `hop_length=512`. MFCCs are averaged over time to
form the `(40,)` input vector. A `StandardScaler` is fitted on folds 1–4 only;
its mean and scale are stored in the checkpoint for inference.
