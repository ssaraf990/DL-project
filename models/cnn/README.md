# CNN Model Architecture Workspace

- **Assigned Team Member:** Satyam Saraf (Student ID: 230953178)
- **Assigned Model:** Convolutional Neural Network (CNN)
- **Input Feature Representation:** 2D Log-Mel Spectrogram (`128 x 216`)

---

## 1. Model Architecture Overview (`ESC50CNN`)

The CNN processes 2D log-Mel spectrogram images as spatial-frequency representations:
- **Input Shape:** `(Batch, 1, 128, 216)` (Channel x Mel Frequency Bands x Time Frames)
- **Conv Block 1:** `Conv2d(1 -> 32, kernel=3)` + `BatchNorm2d` + `ReLU` + `MaxPool2d(2, 2)`
- **Conv Block 2:** `Conv2d(32 -> 64, kernel=3)` + `BatchNorm2d` + `ReLU` + `MaxPool2d(2, 2)`
- **Conv Block 3:** `Conv2d(64 -> 128, kernel=3)` + `BatchNorm2d` + `ReLU` + `MaxPool2d(2, 2)`
- **Conv Block 4:** `Conv2d(128 -> 256, kernel=3)` + `BatchNorm2d` + `ReLU` + `MaxPool2d(2, 2)`
- **Global Pooling:** `AdaptiveAvgPool2d((1, 1))` to condense spatial & temporal features
- **Classifier:** Fully Connected Linear layers `(256 -> 128 -> 50)` with `Dropout(0.4)`

---

## 2. Training Configuration & Performance Summary

- **Held-Out Test Fold:** `Fold 5` (Folds 1..4 for training)
- **Epochs:** 40
- **Batch Size:** 32
- **Optimizer:** Adam (`lr=0.001`, `weight_decay=1e-4`)
- **Hardware Device:** Apple Silicon MPS GPU
- **Overall Test Accuracy (Fold 5):** **63.25%**
- **Macro Precision:** **65.60%**
- **Macro Recall:** **63.25%**
- **Macro F1-Score:** **62.16%**

---

## 3. Directory & File References

- **Model Definition:** [`models/cnn/model.py`](model.py)
- **Training Script:** [`experiments/cnn/train_cnn.py`](../../experiments/cnn/train_cnn.py)
- **Experiment Documentation:** [`experiments/cnn/README.md`](../../experiments/cnn/README.md)
- **Detailed Error Analysis:** [`experiments/cnn/error_analysis.md`](../../experiments/cnn/error_analysis.md)
- **Results Directory:** [`results/cnn/`](../../results/cnn/)
