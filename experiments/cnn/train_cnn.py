"""
Training and Evaluation Pipeline for CNN Model on ESC-50 (Fold 5).

Assigned Team Member: Satyam Saraf (Student ID: 230953178)

Executes actual training, validation, checkpointing, plot generation,
metrics computation, and confusion matrix export on official ESC-50 Fold 5.
"""

import os
import time
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report

from preprocessing.reproducibility import set_seed
from preprocessing.dataset_loader import ESC50Dataset
from models.cnn.model import ESC50CNN
from evaluation.metrics import compute_metrics
from evaluation.confusion_matrix import plot_confusion_matrix


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
        optimizer.step()

        running_loss += loss.item() * inputs.size(0)
        _, preds = torch.max(outputs, 1)
        correct += (preds == labels).sum().item()
        total += inputs.size(0)

    epoch_loss = running_loss / total
    epoch_acc = correct / total
    return epoch_loss, epoch_acc


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

    epoch_loss = running_loss / total
    epoch_acc = correct / total
    return epoch_loss, epoch_acc, np.array(all_labels), np.array(all_preds)


def main():
    # 1. Reproducibility
    seed = 42
    set_seed(seed)

    # 2. Paths
    meta_csv = "data/raw/meta/esc50.csv"
    audio_dir = "data/raw/audio"
    results_dir = "results/cnn"
    os.makedirs(results_dir, exist_ok=True)

    # 3. Hardware device
    if torch.cuda.is_available():
        device = torch.device("cuda")
        hardware_str = f"GPU ({torch.cuda.get_device_name(0)})"
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
        hardware_str = "Apple Silicon MPS GPU"
    else:
        device = torch.device("cpu")
        hardware_str = "CPU"
    print(f"Device selected: {device} [{hardware_str}]")

    # 4. Dataset & DataLoaders (Fold 5 held-out)
    val_fold = 5
    batch_size = 32

    train_dataset = ESC50Dataset(
        meta_csv=meta_csv,
        audio_dir=audio_dir,
        val_fold=val_fold,
        is_train=True,
        feature_type="mel_spectrogram"
    )
    val_dataset = ESC50Dataset(
        meta_csv=meta_csv,
        audio_dir=audio_dir,
        val_fold=val_fold,
        is_train=False,
        feature_type="mel_spectrogram"
    )

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    print(f"Training samples (folds 1..4): {len(train_dataset)}")
    print(f"Validation/Test samples (fold 5): {len(val_dataset)}")

    # 5. Model, Loss, Optimizer
    num_classes = 50
    model = ESC50CNN(num_classes=num_classes, dropout_rate=0.4).to(device)
    criterion = nn.CrossEntropyLoss()
    learning_rate = 1e-3
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-4)

    # 6. Training Loop
    epochs = 40
    checkpoint_path = os.path.join(results_dir, "cnn_esc50_fold5.pt")

    history = {
        "train_loss": [],
        "train_acc": [],
        "val_loss": [],
        "val_acc": [],
    }

    best_val_acc = 0.0
    start_time = time.time()

    print(f"\nStarting CNN training for {epochs} epochs on Fold {val_fold}...")
    for epoch in range(1, epochs + 1):
        t0 = time.time()
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc, _, _ = validate_epoch(model, val_loader, criterion, device)
        t1 = time.time()

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)

        print(
            f"Epoch [{epoch:02d}/{epochs:02d}] ({t1 - t0:.1f}s) - "
            f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc * 100:.2f}% | "
            f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc * 100:.2f}%"
        )

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), checkpoint_path)

    total_training_time = time.time() - start_time
    print(f"\nTraining Complete in {total_training_time / 60.0:.2f} minutes.")
    print(f"Best Validation Accuracy: {best_val_acc * 100:.2f}%")

    # 7. Generate Training & Validation Plots
    plt.figure(figsize=(8, 6))
    plt.plot(range(1, epochs + 1), [acc * 100 for acc in history["train_acc"]], label="Training Accuracy", color="blue", linewidth=2)
    plt.plot(range(1, epochs + 1), [acc * 100 for acc in history["val_acc"]], label="Validation Accuracy", color="orange", linewidth=2)
    plt.title("CNN Training vs. Validation Accuracy (ESC-50 Fold 5)")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy (%)")
    plt.legend(loc="lower right")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, "training_vs_validation_accuracy.png"), dpi=300)
    plt.close()

    plt.figure(figsize=(8, 6))
    plt.plot(range(1, epochs + 1), history["train_loss"], label="Training Loss", color="blue", linewidth=2)
    plt.plot(range(1, epochs + 1), history["val_loss"], label="Validation Loss", color="orange", linewidth=2)
    plt.title("CNN Training vs. Validation Loss (ESC-50 Fold 5)")
    plt.xlabel("Epoch")
    plt.ylabel("Cross-Entropy Loss")
    plt.legend(loc="upper right")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, "training_vs_validation_loss.png"), dpi=300)
    plt.close()
    print("✓ Plots saved: training_vs_validation_accuracy.png & training_vs_validation_loss.png")

    # 8. Evaluation of Best Model Checkpoint
    print("\nEvaluating Best Saved Checkpoint on Fold 5 Test Set...")
    model.load_state_dict(torch.load(checkpoint_path))
    val_loss, val_acc, y_true, y_pred = validate_epoch(model, val_loader, criterion, device)

    # Metrics
    metrics_dict = compute_metrics(y_true, y_pred)
    metrics_dict["best_epoch_val_acc"] = float(best_val_acc)
    metrics_dict["final_val_loss"] = float(val_loss)
    metrics_dict["total_training_time_seconds"] = float(total_training_time)
    metrics_dict["hardware_used"] = hardware_str
    metrics_dict["epochs_trained"] = epochs
    metrics_dict["held_out_fold"] = val_fold

    metrics_path = os.path.join(results_dir, "metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(metrics_dict, f, indent=4)
    print(f"✓ Metrics saved to {metrics_path}")

    # Load 50 class names from esc50.csv
    df_meta = pd.read_csv(meta_csv)
    class_map = df_meta[["target", "category"]].drop_duplicates().sort_values("target")
    class_names = class_map["category"].tolist()

    # Classification report text
    report_str = classification_report(y_true, y_pred, target_names=class_names, digits=4, zero_division=0)
    report_path = os.path.join(results_dir, "classification_report.txt")
    with open(report_path, "w") as f:
        f.write("ESC-50 CNN Model Classification Report (Fold 5 Test Set)\n")
        f.write("=" * 60 + "\n\n")
        f.write(report_str)
    print(f"✓ Classification report saved to {report_path}")

    # Confusion matrix heatmap plot
    cm_path = os.path.join(results_dir, "confusion_matrix.png")
    plot_confusion_matrix(
        y_true=y_true,
        y_pred=y_pred,
        class_names=class_names,
        save_path=cm_path,
        title="CNN Model - Normalized 50x50 Confusion Matrix (Fold 5)"
    )
    print(f"✓ Confusion matrix plot saved to {cm_path}")


if __name__ == "__main__":
    main()
