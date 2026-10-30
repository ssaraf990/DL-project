"""
Training and Evaluation Pipeline for LSTM Model on ESC-50 (Held-out Fold 5 Evaluation).

Assigned Team Member: Ryan Gupta (Student ID: 230953288)

Uses the same protocol as the CNN and CNN-LSTM experiments (seed 42, folds 1-4 train /
fold 5 held-out, batch 32, Adam lr=1e-3 wd=1e-4, 40 epochs, best-epoch checkpoint) so the
results go straight into the common comparison table.

Sequential MFCC features (216 x 40 per clip) are extracted once through the shared
ESC50Dataset pipeline and cached to data/processed/seq_mfcc.npz, since recomputing
MFCCs with librosa every epoch is the slowest part of training.

Usage (from repo root):
    PYTHONPATH=. python3 experiments/lstm/train_lstm.py              # Fold 5 (main result)
    PYTHONPATH=. python3 experiments/lstm/train_lstm.py --all-folds  # + 5-fold cross-validation
"""

import os
import time
import json
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import classification_report

from preprocessing.reproducibility import set_seed
from preprocessing.dataset_loader import ESC50Dataset
from models.lstm.model import ESC50LSTM
from evaluation.metrics import compute_metrics
from evaluation.confusion_matrix import plot_confusion_matrix


META_CSV = "data/raw/meta/esc50.csv"
AUDIO_DIR = "data/raw/audio"
CACHE_PATH = "data/processed/seq_mfcc.npz"
RESULTS_DIR = "results/lstm"

SEED = 42
BATCH_SIZE = 32
EPOCHS = 40
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4
GRAD_CLIP_NORM = 5.0


def load_or_extract_features():
    """Return (features (2000, 216, 40), targets (2000,), folds (2000,)) using the shared pipeline."""
    if os.path.exists(CACHE_PATH):
        cache = np.load(CACHE_PATH)
        return cache["features"], cache["targets"], cache["folds"]

    print(f"Extracting sequential MFCC features for all clips (one-time, cached to {CACHE_PATH})...")
    dataset = ESC50Dataset(META_CSV, AUDIO_DIR, val_fold=None, feature_type="seq_mfcc")
    features = np.zeros((len(dataset), 216, 40), dtype=np.float32)
    targets = np.zeros(len(dataset), dtype=np.int64)
    for i in range(len(dataset)):
        x, y = dataset[i]
        features[i] = x.numpy()
        targets[i] = y
        if (i + 1) % 200 == 0:
            print(f"  {i + 1}/{len(dataset)} clips processed")
    folds = dataset.df["fold"].to_numpy().astype(np.int64)

    os.makedirs(os.path.dirname(CACHE_PATH), exist_ok=True)
    np.savez(CACHE_PATH, features=features, targets=targets, folds=folds)
    return features, targets, folds


def train_epoch(model, dataloader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for inputs, labels in dataloader:
        inputs = inputs.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        # Clip gradients to avoid exploding gradients through 216 LSTM time steps
        nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP_NORM)
        optimizer.step()

        running_loss += loss.item() * inputs.size(0)
        _, preds = torch.max(outputs, 1)
        correct += (preds == labels).sum().item()
        total += inputs.size(0)

    return running_loss / total, correct / total


def validate_epoch(model, dataloader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)
            loss = criterion(outputs, labels)

            running_loss += loss.item() * inputs.size(0)
            _, preds = torch.max(outputs, 1)
            correct += (preds == labels).sum().item()
            total += inputs.size(0)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    return running_loss / total, correct / total, np.array(all_labels), np.array(all_preds)


def run_fold(val_fold, features, targets, folds, device, checkpoint_path, verbose=True):
    """Train on all folds except val_fold, evaluate on val_fold. Returns (history, best_acc, y_true, y_pred, seconds)."""
    set_seed(SEED)

    train_mask = folds != val_fold
    train_ds = TensorDataset(torch.from_numpy(features[train_mask]), torch.from_numpy(targets[train_mask]))
    val_ds = TensorDataset(torch.from_numpy(features[~train_mask]), torch.from_numpy(targets[~train_mask]))
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    model = ESC50LSTM(num_classes=50, n_mfcc=40, hidden_size=128, num_lstm_layers=2, dropout_rate=0.4).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)

    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}
    best_val_acc = 0.0
    start_time = time.time()

    for epoch in range(1, EPOCHS + 1):
        t0 = time.time()
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc, _, _ = validate_epoch(model, val_loader, criterion, device)
        t1 = time.time()

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)

        if verbose:
            print(
                f"Epoch [{epoch:02d}/{EPOCHS:02d}] ({t1 - t0:.1f}s) - "
                f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc * 100:.2f}% | "
                f"Fold {val_fold} Val Loss: {val_loss:.4f} | Fold {val_fold} Val Acc: {val_acc * 100:.2f}%"
            )

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), checkpoint_path)

    total_time = time.time() - start_time

    model.load_state_dict(torch.load(checkpoint_path))
    _, _, y_true, y_pred = validate_epoch(model, val_loader, criterion, device)
    return history, best_val_acc, y_true, y_pred, total_time


def save_plots(history, val_fold):
    epochs = range(1, len(history["train_acc"]) + 1)

    plt.figure(figsize=(8, 6))
    plt.plot(epochs, [acc * 100 for acc in history["train_acc"]], label="Training Accuracy", color="blue", linewidth=2)
    plt.plot(epochs, [acc * 100 for acc in history["val_acc"]], label=f"Fold {val_fold} Val Accuracy", color="orange", linewidth=2)
    plt.title(f"LSTM Training vs. Fold {val_fold} Validation Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy (%)")
    plt.legend(loc="lower right")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "training_vs_validation_accuracy.png"), dpi=300)
    plt.close()

    plt.figure(figsize=(8, 6))
    plt.plot(epochs, history["train_loss"], label="Training Loss", color="blue", linewidth=2)
    plt.plot(epochs, history["val_loss"], label=f"Fold {val_fold} Val Loss", color="orange", linewidth=2)
    plt.title(f"LSTM Training vs. Fold {val_fold} Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Cross-Entropy Loss")
    plt.legend(loc="upper right")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "training_vs_validation_loss.png"), dpi=300)
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="Train the ESC-50 LSTM model.")
    parser.add_argument("--all-folds", action="store_true", help="Also run full 5-fold cross-validation.")
    args = parser.parse_args()

    os.makedirs(RESULTS_DIR, exist_ok=True)

    # Hardware selection
    if torch.cuda.is_available():
        device = torch.device("cuda")
        hardware_str = f"GPU ({torch.cuda.get_device_name(0)})"
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
        hardware_str = "Apple Silicon MPS GPU"
    else:
        device = torch.device("cpu")
        hardware_str = "CPU"

    features, targets, folds = load_or_extract_features()

    val_fold = 5
    print("=== PRE-TRAINING HARDWARE & EXPERIMENT STATUS ===")
    print(f"Device Selected: {device}")
    print(f"Training Dataset Size (Folds 1..4): {int((folds != val_fold).sum())} samples")
    print(f"Held-out Fold 5 Evaluation Size: {int((folds == val_fold).sum())} samples")
    print(f"Input Feature Tensor Shape: {tuple(features.shape[1:])} per sample (T, n_mfcc)")
    print("=================================================\n")

    # 1. Main result: held-out Fold 5 (same protocol as CNN / CNN-LSTM)
    checkpoint_path = os.path.join(RESULTS_DIR, "lstm_esc50_fold5.pt")
    print(f"Starting LSTM training for {EPOCHS} epochs on Fold {val_fold}...")
    history, best_val_acc, y_true, y_pred, total_time = run_fold(
        val_fold, features, targets, folds, device, checkpoint_path
    )
    print(f"\nLSTM Training Complete in {total_time / 60.0:.2f} minutes.")
    print(f"Best Held-out Fold 5 Evaluation Accuracy: {best_val_acc * 100:.2f}%")

    save_plots(history, val_fold)
    print("✓ Plots saved: training_vs_validation_accuracy.png & training_vs_validation_loss.png")

    metrics_dict = compute_metrics(y_true, y_pred)
    metrics_dict["best_epoch_val_acc"] = float(best_val_acc)
    metrics_dict["best_epoch"] = int(np.argmax(history["val_acc"]) + 1)
    metrics_dict["final_epoch_val_acc"] = float(history["val_acc"][-1])
    metrics_dict["final_val_loss"] = float(history["val_loss"][-1])
    metrics_dict["total_training_time_seconds"] = float(total_time)
    metrics_dict["hardware_used"] = hardware_str
    metrics_dict["epochs_trained"] = EPOCHS
    metrics_dict["held_out_fold"] = val_fold
    metrics_dict["trainable_parameters"] = sum(
        p.numel() for p in ESC50LSTM().parameters() if p.requires_grad
    )

    df_meta = pd.read_csv(META_CSV)
    class_names = df_meta[["target", "category"]].drop_duplicates().sort_values("target")["category"].tolist()

    report_str = classification_report(y_true, y_pred, target_names=class_names, digits=4, zero_division=0)
    with open(os.path.join(RESULTS_DIR, "classification_report.txt"), "w") as f:
        f.write("ESC-50 LSTM Model Classification Report (Held-out Fold 5 Evaluation)\n")
        f.write("=" * 65 + "\n\n")
        f.write(report_str)
    print("✓ Classification report saved")

    plot_confusion_matrix(
        y_true=y_true,
        y_pred=y_pred,
        class_names=class_names,
        save_path=os.path.join(RESULTS_DIR, "confusion_matrix.png"),
        title="LSTM Model - Normalized 50x50 Confusion Matrix (Fold 5)",
    )
    print("✓ Confusion matrix plot saved")

    # Save per-clip predictions for error analysis
    pd.DataFrame({
        "filename": df_meta[df_meta["fold"] == val_fold]["filename"].to_numpy(),
        "true": [class_names[i] for i in y_true],
        "pred": [class_names[i] for i in y_pred],
    }).to_csv(os.path.join(RESULTS_DIR, "fold5_predictions.csv"), index=False)

    # 2. Optional: full 5-fold cross-validation
    if args.all_folds:
        print("\n=== 5-FOLD CROSS-VALIDATION ===")
        fold_accs = {}
        for k in range(1, 6):
            fold_ckpt = os.path.join(RESULTS_DIR, f"lstm_esc50_fold{k}.pt")
            _, acc, _, _, secs = run_fold(k, features, targets, folds, device, fold_ckpt, verbose=False)
            fold_accs[k] = float(acc)
            print(f"Fold {k}: best val acc {acc * 100:.2f}% ({secs / 60.0:.1f} min)")
        accs = np.array(list(fold_accs.values()))
        metrics_dict["cv_fold_accuracies"] = fold_accs
        metrics_dict["cv_mean_accuracy"] = float(accs.mean())
        metrics_dict["cv_std_accuracy"] = float(accs.std())
        print(f"5-Fold CV Accuracy: {accs.mean() * 100:.2f}% ± {accs.std() * 100:.2f}%")

    with open(os.path.join(RESULTS_DIR, "metrics.json"), "w") as f:
        json.dump(metrics_dict, f, indent=4)
    print("✓ Metrics saved to results/lstm/metrics.json")


if __name__ == "__main__":
    main()
