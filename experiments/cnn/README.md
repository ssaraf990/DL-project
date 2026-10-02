# CNN Model Experiment Setup & Performance Summary

- **Assigned Member:** Satyam Saraf (Student ID: 230953178)
- **Model Architecture:** 4-Block 2D Convolutional Neural Network (`ESC50CNN`)
- **Input Feature Representation:** Log-Mel Spectrogram (`128 x 216`)
- **Dataset:** ESC-50 Environmental Sound Classification (Official Fold 5 held-out)

---

## 1. Experiment Configuration

| Parameter | Value |
| :--- | :--- |
| **Framework** | PyTorch 2.14 |
| **Hardware Device** | Apple Silicon MPS GPU |
| **Random Seed** | `42` (`set_seed`) |
| **Held-Out Validation Fold** | `Fold 5` (Folds 1, 2, 3, 4 for training) |
| **Training Samples** | 1,600 audio clips |
| **Test/Validation Samples** | 400 audio clips |
| **Batch Size** | 32 |
| **Optimizer** | Adam (`lr=0.001`, `weight_decay=1e-4`) |
| **Loss Function** | CrossEntropyLoss |
| **Epochs Trained** | 40 |
| **Total Training Duration** | 16.14 minutes (968.24 seconds) |

---

## 2. Quantitative Experimental Results (Fold 5 Test Set)

| Metric | Value |
| :--- | :--- |
| **Overall Classification Accuracy** | **63.25%** |
| **Macro Precision** | **65.60%** |
| **Weighted Precision** | **65.60%** |
| **Macro Recall** | **63.25%** |
| **Weighted Recall** | **63.25%** |
| **Macro F1-Score** | **62.16%** |
| **Weighted F1-Score** | **62.16%** |
| **Final Validation Loss** | **1.2408** |

---

## 3. Generated Artifacts & Pointers

All experiment results, metrics, plots, and checkpoints are stored in `results/cnn/`:

- **Metrics JSON:** [`results/cnn/metrics.json`](../../results/cnn/metrics.json)
- **Classification Report:** [`results/cnn/classification_report.txt`](../../results/cnn/classification_report.txt)
- **Confusion Matrix Plot:** [`results/cnn/confusion_matrix.png`](../../results/cnn/confusion_matrix.png)
- **Accuracy Curve:** [`results/cnn/training_vs_validation_accuracy.png`](../../results/cnn/training_vs_validation_accuracy.png)
- **Loss Curve:** [`results/cnn/training_vs_validation_loss.png`](../../results/cnn/training_vs_validation_loss.png)
- **Trained Model Checkpoint:** `results/cnn/cnn_esc50_fold5.pt`
- **Detailed Error Analysis:** [`experiments/cnn/error_analysis.md`](error_analysis.md)

---

## 4. Observations & Preliminary Status

- **Status:** Preliminary baseline experiment completed for Phase 2.
- **Observations:** The 2D CNN model effectively captures spatial-frequency representations from log-Mel spectrograms, achieving **63.25% test accuracy** on 50-class environmental sound classification without pretraining.
- **Potential Enhancements:** Data augmentation (SpecAugment, Mixup, Random Time Shift), learning rate scheduling, and depth tuning can further boost performance in Phase 3.
