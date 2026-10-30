# LSTM Model Workspace

- **Member Name:** Ryan Gupta
- **Registration Number:** 230953288
- **Assigned Model:** Long Short-Term Memory (LSTM)
- **Model Role:** Models temporal dependencies and sequential transitions in audio recordings using sequential frame-level MFCC feature vectors.

## Architecture (`ESC50LSTM` in `model.py`)

| Stage | Layer | Output Shape |
| :--- | :--- | :--- |
| Input | Sequential MFCC | `(B, 216, 40)` |
| 1 | `BatchNorm1d(40)` per-coefficient normalisation | `(B, 216, 40)` |
| 2 | 2-layer Bidirectional `LSTM(hidden=128)`, inter-layer dropout 0.2 | `(B, 216, 256)` |
| 3 | Mean pooling ‖ Max pooling over time | `(B, 512)` |
| 4 | Dropout(0.4) → Linear(512→128) → ReLU → Dropout(0.2) → Linear(128→50) | `(B, 50)` |

**Total trainable parameters:** 641,538

### Design Choices

- **BatchNorm on the input:** raw MFCC coefficients differ in scale by orders of magnitude (c0 ≈ hundreds, higher coefficients ≈ ±10). Normalising each coefficient lets the LSTM gates work in a sensible range. BatchNorm statistics come only from training batches, so nothing from the held-out fold leaks in.
- **Bidirectional:** a clip is classified only after the whole clip is seen, so reading it in both directions is free context.
- **Mean + max pooling instead of the last hidden state:** the last state of a 216-step sequence mostly reflects the end of the clip. Mean pooling captures stationary background texture, and max pooling keeps short events such as a knock or bark.
- **Gradient clipping (norm 5.0):** standard for LSTMs to prevent exploding gradients over long sequences.

See [`experiments/lstm/README.md`](../../experiments/lstm/README.md) for results.
