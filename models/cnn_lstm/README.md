# CNN-LSTM Hybrid Model Workspace

- **Assigned Team Member:** Karan Shah (Student ID: 230953540)
- **Git Identity:** Karan Shah <kp.shah098@gmail.com> (GitHub: `Karan-Shah7`)
- **Assigned Model:** CNN-LSTM Hybrid (`ESC50CNNLSTM`)
- **Input Feature Representation:** 2D Log-Mel Spectrogram (`128 x 216`)

---

## 1. Model Architecture Overview (`ESC50CNNLSTM`)

The CNN-LSTM model combines 2D convolutional feature extraction with recurrent sequence modeling:
- **Input Shape:** `(Batch, 1, 128, 216)` (Channel x Mel Bands x Time Frames)
- **CNN Backbone:**
  - Conv Block 1: `Conv2d(1 -> 32, k=3)` + `BatchNorm2d` + `ReLU` + `MaxPool2d(2, 2)` $\to$ `(B, 32, 64, 108)`
  - Conv Block 2: `Conv2d(32 -> 64, k=3)` + `BatchNorm2d` + `ReLU` + `MaxPool2d(2, 2)` $\to$ `(B, 64, 32, 54)`
  - Conv Block 3: `Conv2d(64 -> 128, k=3)` + `BatchNorm2d` + `ReLU` + `MaxPool2d(2, 2)` $\to$ `(B, 128, 16, 27)`
- **CNN-to-LSTM Tensor Transformation:**
  - Permute channels to time dimension: `(B, 128, 16, 27)` $\to$ `(B, 27, 128, 16)`
  - Flatten spatial/frequency features per time step: `(B, 27, 2048)` where sequence length $T'=27$ and feature dimension per step is $128 \times 16 = 2048$.
- **LSTM Sequence Modeling:**
  - 2-layer `LSTM` (`input_size=2048`, `hidden_size=128`, `batch_first=True`, `dropout=0.2`) $\to$ `(B, 27, 128)`
  - Temporal Mean Pooling across time frames $T'=27$ $\to$ `(B, 128)`
- **Classifier:** Linear layers `(128 -> 64 -> 50)` with `Dropout(0.4)`

---

## 2. Training Configuration & Performance Summary

- **Held-Out Evaluation Fold:** `Fold 5` (Folds 1..4 for training)
- **Epochs:** 40
- **Batch Size:** 32
- **Optimizer:** Adam (`lr=0.001`, `weight_decay=1e-4`)
- **Hardware Device:** Apple Silicon MPS GPU
- **Held-out Fold 5 Accuracy:** **57.75%**
- **Macro Precision:** **57.02%**
- **Macro Recall:** **57.75%**
- **Macro F1-Score:** **55.02%**

---

## 3. Directory & File References

- **Model Implementation:** [`models/cnn_lstm/model.py`](model.py)
- **Training Script:** [`experiments/cnn_lstm/train_cnn_lstm.py`](../../experiments/cnn_lstm/train_cnn_lstm.py)
- **Experiment Documentation:** [`experiments/cnn_lstm/README.md`](../../experiments/cnn_lstm/README.md)
- **Detailed Error Analysis:** [`experiments/cnn_lstm/error_analysis.md`](../../experiments/cnn_lstm/error_analysis.md)
- **Results Directory:** [`results/cnn_lstm/`](../../results/cnn_lstm/)
