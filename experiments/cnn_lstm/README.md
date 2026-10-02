# CNN-LSTM Hybrid Model Experiment Setup & Performance Summary

- **Assigned Member:** Karan Shah (Student ID: 230953540)
- **Git Identity:** Karan Shah <kp.shah098@gmail.com> (GitHub: `Karan-Shah7`)
- **Model Architecture:** 3-Block 2D CNN + 2-Layer LSTM Hybrid (`ESC50CNNLSTM`)
- **Input Feature Representation:** Log-Mel Spectrogram (`128 x 216`)
- **Dataset Protocol:** ESC-50 Environmental Sound Classification (Held-out Fold 5 Evaluation)

---

## 1. Experiment Configuration

| Parameter | Value |
| :--- | :--- |
| **Framework** | PyTorch 2.14 |
| **Hardware Device** | Apple Silicon MPS GPU |
| **Random Seed** | `42` (`set_seed`) |
| **Held-Out Evaluation Fold** | `Fold 5` (Folds 1, 2, 3, 4 for training) |
| **Training Samples** | 1,600 audio clips |
| **Held-Out Evaluation Samples** | 400 audio clips |
| **Batch Size** | 32 |
| **Optimizer** | Adam (`lr=0.001`, `weight_decay=1e-4`) |
| **Loss Function** | CrossEntropyLoss |
| **Epochs Trained** | 40 |
| **Total Training Duration** | 16.07 minutes (964.32 seconds) |

---

## 2. Quantitative Experimental Results (Held-out Fold 5 Evaluation)

| Metric | Value |
| :--- | :--- |
| **Overall Classification Accuracy** | **57.75%** |
| **Macro Precision** | **57.02%** |
| **Weighted Precision** | **57.02%** |
| **Macro Recall** | **57.75%** |
| **Weighted Recall** | **57.75%** |
| **Macro F1-Score** | **55.02%** |
| **Weighted F1-Score** | **55.02%** |
| **Final Validation Loss** | **2.0921** |

---

## 3. Generated Artifacts & Pointers

All experiment results, metrics, plots, and checkpoints are stored in `results/cnn_lstm/`:

- **Metrics JSON:** [`results/cnn_lstm/metrics.json`](../../results/cnn_lstm/metrics.json)
- **Classification Report:** [`results/cnn_lstm/classification_report.txt`](../../results/cnn_lstm/classification_report.txt)
- **Confusion Matrix Plot:** [`results/cnn_lstm/confusion_matrix.png`](../../results/cnn_lstm/confusion_matrix.png)
- **Accuracy Curve:** [`results/cnn_lstm/training_vs_validation_accuracy.png`](../../results/cnn_lstm/training_vs_validation_accuracy.png)
- **Loss Curve:** [`results/cnn_lstm/training_vs_validation_loss.png`](../../results/cnn_lstm/training_vs_validation_loss.png)
- **Trained Model Checkpoint:** `results/cnn_lstm/cnn_lstm_esc50_fold5.pt`
- **Detailed Error Analysis:** [`experiments/cnn_lstm/error_analysis.md`](error_analysis.md)

---

## 4. Reproduction Command

To reproduce this experiment from the root directory:

```bash
PYTHONPATH=. python3 experiments/cnn_lstm/train_cnn_lstm.py
```
