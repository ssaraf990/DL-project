# LSTM Model Experiment Setup & Performance Summary

- **Assigned Member:** Ryan Gupta (Student ID: 230953288)
- **Model Architecture:** 2-Layer Bidirectional LSTM with Mean + Max Temporal Pooling (`ESC50LSTM`)
- **Input Feature Representation:** Sequential frame-level MFCC (`216 time steps x 40 coefficients`)
- **Dataset Protocol:** ESC-50 Environmental Sound Classification (Held-out Fold 5 Evaluation, plus full 5-fold CV)

---

## 1. Experiment Configuration

Same protocol as the CNN and CNN-LSTM experiments so results are directly comparable.

| Parameter | Value |
| :--- | :--- |
| **Framework** | PyTorch 2.14 |
| **Hardware Device** | Apple Silicon MPS GPU |
| **Random Seed** | `42` (`set_seed`, re-applied before every fold) |
| **Held-Out Evaluation Fold** | `Fold 5` (Folds 1, 2, 3, 4 for training) |
| **Training Samples** | 1,600 audio clips |
| **Held-Out Evaluation Samples** | 400 audio clips |
| **Batch Size** | 32 |
| **Optimizer** | Adam (`lr=0.001`, `weight_decay=1e-4`) |
| **Loss Function** | CrossEntropyLoss |
| **Gradient Clipping** | Max norm 5.0 (stabilises backpropagation through 216 time steps) |
| **Epochs Trained** | 40 (checkpoint kept from best epoch) |
| **Trainable Parameters** | 641,538 |
| **Total Training Duration** | 1.51 minutes (90.6 seconds) |

Sequential MFCCs are extracted once through the shared `ESC50Dataset(feature_type="seq_mfcc")` pipeline and cached to `data/processed/seq_mfcc.npz` (gitignored), so each epoch takes ~2.3 s instead of re-running librosa.

---

## 2. Quantitative Experimental Results (Held-out Fold 5 Evaluation)

| Metric | Value |
| :--- | :--- |
| **Overall Classification Accuracy** | **48.75%** |
| **Macro Precision** | **48.30%** |
| **Weighted Precision** | **48.30%** |
| **Macro Recall** | **48.75%** |
| **Weighted Recall** | **48.75%** |
| **Macro F1-Score** | **46.75%** |
| **Weighted F1-Score** | **46.75%** |
| **Best Epoch** | 24 |
| **Final-Epoch (40) Accuracy** | 47.25% |
| **Final Validation Loss** | 3.3486 |

### 5-Fold Cross-Validation

| Fold 1 | Fold 2 | Fold 3 | Fold 4 | Fold 5 | **Mean ± Std** |
| :---: | :---: | :---: | :---: | :---: | :---: |
| 46.25% | 48.75% | 45.00% | 50.75% | 48.75% | **47.90% ± 2.03%** |

> **Note on model selection:** like the CNN and CNN-LSTM runs, the reported number is the best epoch on the held-out fold. The final-epoch accuracy (47.25%) is reported alongside it to show the selection effect is small (1.5 points).

### Observations

- **Strong overfitting:** training accuracy reaches 95.4% while validation accuracy plateaus around 46–48% from epoch ~10 onward, and validation loss rises after epoch 10. With only 32 clips per class for training, the LSTM memorises sequences rather than generalising.
- **Stable across folds:** the 2-point standard deviation over 5 folds shows the fold-5 result is representative, not a lucky split.
- **Below the spectrogram models:** the CNN (63.25%) and CNN-LSTM (57.75%) both outperform the LSTM on fold 5. A 40-coefficient MFCC frame is a compressed description of the spectrum, while the Mel-spectrogram models see all 128 frequency bands and learn local time-frequency patterns directly.

---

## 3. Generated Artifacts & Pointers

All experiment results, metrics, plots, and checkpoints are stored in `results/lstm/`:

- **Metrics JSON:** [`results/lstm/metrics.json`](../../results/lstm/metrics.json)
- **Classification Report:** [`results/lstm/classification_report.txt`](../../results/lstm/classification_report.txt)
- **Confusion Matrix Plot:** [`results/lstm/confusion_matrix.png`](../../results/lstm/confusion_matrix.png)
- **Accuracy Curve:** [`results/lstm/training_vs_validation_accuracy.png`](../../results/lstm/training_vs_validation_accuracy.png)
- **Loss Curve:** [`results/lstm/training_vs_validation_loss.png`](../../results/lstm/training_vs_validation_loss.png)
- **Per-clip Fold 5 Predictions:** [`results/lstm/fold5_predictions.csv`](../../results/lstm/fold5_predictions.csv)
- **Full Training Log:** [`results/lstm/train_log.txt`](../../results/lstm/train_log.txt)
- **Trained Model Checkpoints:** `results/lstm/lstm_esc50_fold{1..5}.pt` (gitignored)
- **Detailed Error Analysis:** [`experiments/lstm/error_analysis.md`](error_analysis.md)

---

## 4. Reproduction Command

To reproduce this experiment from the root directory:

```bash
PYTHONPATH=. python3 experiments/lstm/train_lstm.py --all-folds
```

Omit `--all-folds` to run only the Fold 5 experiment.
