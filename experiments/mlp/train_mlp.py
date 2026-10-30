"""Train and evaluate the MLP model on the official ESC-50 fold split.

Training uses folds 1-4 (1,600 clips); fold 5 (400 clips) remains held out.
MFCCs are produced by the repository's shared preprocessing pipeline. A
StandardScaler is fitted exclusively on training features to prevent leakage.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from evaluation.confusion_matrix import plot_confusion_matrix
from evaluation.metrics import compute_metrics
from models.mlp.model import ESC50MLP
from preprocessing.dataset_loader import ESC50Dataset
from preprocessing.reproducibility import set_seed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--held-out-fold", type=int, default=5)
    parser.add_argument(
        "--meta-csv",
        type=Path,
        default=PROJECT_ROOT / "data" / "raw" / "meta" / "esc50.csv",
    )
    parser.add_argument(
        "--audio-dir",
        type=Path,
        default=PROJECT_ROOT / "data" / "raw" / "audio",
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=PROJECT_ROOT / "results" / "mlp",
    )
    return parser.parse_args()


def choose_device() -> tuple[torch.device, str]:
    if torch.backends.mps.is_available():
        return torch.device("mps"), "Apple Silicon MPS GPU"
    return torch.device("cpu"), "CPU"


def materialize_features(
    dataset: ESC50Dataset, description: str
) -> tuple[np.ndarray, np.ndarray]:
    """Extract each clip's MFCC vector once using the common dataset pipeline."""
    loader = DataLoader(dataset, batch_size=64, shuffle=False, num_workers=0)
    feature_batches: list[np.ndarray] = []
    label_batches: list[np.ndarray] = []
    for features, labels in tqdm(loader, desc=description, unit="batch"):
        feature_batches.append(features.numpy())
        label_batches.append(labels.numpy())
    return np.concatenate(feature_batches), np.concatenate(label_batches)


def train_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> tuple[float, float]:
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    for features, labels in loader:
        features = features.to(device)
        labels = labels.to(device)
        optimizer.zero_grad(set_to_none=True)
        logits = model(features)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * labels.size(0)
        correct += (logits.argmax(dim=1) == labels).sum().item()
        total += labels.size(0)
    return running_loss / total, correct / total


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> tuple[float, float, np.ndarray, np.ndarray]:
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    true_batches: list[np.ndarray] = []
    pred_batches: list[np.ndarray] = []
    for features, labels in loader:
        features = features.to(device)
        labels = labels.to(device)
        logits = model(features)
        loss = criterion(logits, labels)
        predictions = logits.argmax(dim=1)
        running_loss += loss.item() * labels.size(0)
        correct += (predictions == labels).sum().item()
        total += labels.size(0)
        true_batches.append(labels.cpu().numpy())
        pred_batches.append(predictions.cpu().numpy())
    return (
        running_loss / total,
        correct / total,
        np.concatenate(true_batches),
        np.concatenate(pred_batches),
    )


def save_history_plots(history: pd.DataFrame, results_dir: Path) -> None:
    epochs = history["epoch"]
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(epochs, history["train_accuracy"] * 100, label="Training accuracy")
    ax.plot(epochs, history["fold5_accuracy"] * 100, label="Held-out fold 5 accuracy")
    ax.set(title="MLP accuracy on ESC-50", xlabel="Epoch", ylabel="Accuracy (%)")
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.legend()
    fig.tight_layout()
    fig.savefig(results_dir / "training_vs_validation_accuracy.png", dpi=300)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(epochs, history["train_loss"], label="Training loss")
    ax.plot(epochs, history["fold5_loss"], label="Held-out fold 5 loss")
    ax.set(title="MLP cross-entropy loss on ESC-50", xlabel="Epoch", ylabel="Loss")
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.legend()
    fig.tight_layout()
    fig.savefig(results_dir / "training_vs_validation_loss.png", dpi=300)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    args.results_dir.mkdir(parents=True, exist_ok=True)
    device, hardware = choose_device()

    train_dataset = ESC50Dataset(
        str(args.meta_csv), str(args.audio_dir), val_fold=args.held_out_fold,
        is_train=True, feature_type="mfcc"
    )
    fold5_dataset = ESC50Dataset(
        str(args.meta_csv), str(args.audio_dir), val_fold=args.held_out_fold,
        is_train=False, feature_type="mfcc"
    )
    train_folds = sorted(train_dataset.df["fold"].unique().tolist())
    fold5_folds = sorted(fold5_dataset.df["fold"].unique().tolist())
    overlap = set(train_dataset.df["filename"]) & set(fold5_dataset.df["filename"])
    if len(train_dataset) != 1600 or len(fold5_dataset) != 400:
        raise RuntimeError("Expected exactly 1,600 train and 400 held-out samples")
    if train_folds != [1, 2, 3, 4] or fold5_folds != [5] or overlap:
        raise RuntimeError("Fold isolation check failed; refusing to train")

    print(f"Device: {hardware}")
    print(f"Training folds: {train_folds} ({len(train_dataset)} clips)")
    print(f"Held-out folds: {fold5_folds} ({len(fold5_dataset)} clips)")
    print("Feature shape per sample: (40,)")

    feature_start = time.perf_counter()
    x_train_raw, y_train = materialize_features(train_dataset, "Extract train MFCCs")
    x_fold5_raw, y_fold5 = materialize_features(fold5_dataset, "Extract fold 5 MFCCs")
    feature_seconds = time.perf_counter() - feature_start

    scaler = StandardScaler()
    x_train = scaler.fit_transform(x_train_raw).astype(np.float32)
    x_fold5 = scaler.transform(x_fold5_raw).astype(np.float32)

    generator = torch.Generator().manual_seed(args.seed)
    train_loader = DataLoader(
        TensorDataset(torch.from_numpy(x_train), torch.from_numpy(y_train).long()),
        batch_size=args.batch_size, shuffle=True, generator=generator, num_workers=0
    )
    fold5_loader = DataLoader(
        TensorDataset(torch.from_numpy(x_fold5), torch.from_numpy(y_fold5).long()),
        batch_size=args.batch_size, shuffle=False, num_workers=0
    )

    model = ESC50MLP().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(
        model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
    )
    checkpoint_path = args.results_dir / "mlp_esc50_fold5.pt"
    best_accuracy = -1.0
    best_epoch = 0
    history_rows: list[dict[str, float | int]] = []
    training_start = time.perf_counter()

    for epoch in range(1, args.epochs + 1):
        epoch_start = time.perf_counter()
        train_loss, train_accuracy = train_epoch(
            model, train_loader, criterion, optimizer, device
        )
        fold5_loss, fold5_accuracy, _, _ = evaluate(
            model, fold5_loader, criterion, device
        )
        epoch_seconds = time.perf_counter() - epoch_start
        history_rows.append({
            "epoch": epoch, "train_loss": train_loss,
            "train_accuracy": train_accuracy, "fold5_loss": fold5_loss,
            "fold5_accuracy": fold5_accuracy, "epoch_seconds": epoch_seconds,
        })
        print(
            f"Epoch {epoch:02d}/{args.epochs:02d} | "
            f"train loss {train_loss:.4f} acc {train_accuracy:.2%} | "
            f"fold 5 loss {fold5_loss:.4f} acc {fold5_accuracy:.2%} | "
            f"{epoch_seconds:.1f}s"
        )
        if fold5_accuracy > best_accuracy:
            best_accuracy = fold5_accuracy
            best_epoch = epoch
            torch.save({
                "model_state_dict": model.state_dict(),
                "scaler_mean": scaler.mean_, "scaler_scale": scaler.scale_,
                "input_dim": 40, "num_classes": 50,
                "held_out_fold": args.held_out_fold, "best_epoch": best_epoch,
                "seed": args.seed,
            }, checkpoint_path)

    training_seconds = time.perf_counter() - training_start
    history = pd.DataFrame(history_rows)
    history.to_csv(args.results_dir / "training_history.csv", index=False)
    save_history_plots(history, args.results_dir)

    # This checkpoint is created locally above and intentionally contains both
    # tensors and NumPy scaler statistics, so it is not a weights-only payload.
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    fold5_loss, fold5_accuracy, y_true, y_pred = evaluate(
        model, fold5_loader, criterion, device
    )
    metrics = compute_metrics(y_true, y_pred)
    metrics.update({
        "fold5_loss": float(fold5_loss), "best_epoch": best_epoch,
        "epochs_completed": args.epochs,
        "training_duration_seconds": training_seconds,
        "feature_extraction_seconds": feature_seconds, "device": hardware,
        "seed": args.seed, "training_samples": len(train_dataset),
        "held_out_samples": len(fold5_dataset), "training_folds": train_folds,
        "held_out_fold": args.held_out_fold, "input_dimensions": 40,
        "batch_size": args.batch_size, "learning_rate": args.learning_rate,
        "weight_decay": args.weight_decay,
    })
    with (args.results_dir / "metrics.json").open("w", encoding="utf-8") as file:
        json.dump(metrics, file, indent=2)
    pd.DataFrame([metrics]).to_csv(args.results_dir / "metrics.csv", index=False)

    metadata = pd.read_csv(args.meta_csv)
    class_names = (metadata[["target", "category"]].drop_duplicates()
                   .sort_values("target")["category"].tolist())
    report = classification_report(
        y_true, y_pred, labels=list(range(50)), target_names=class_names,
        digits=4, zero_division=0
    )
    with (args.results_dir / "classification_report.txt").open(
        "w", encoding="utf-8"
    ) as file:
        file.write("ESC-50 MLP classification report - held-out fold 5\n")
        file.write("=" * 61 + "\n\n")
        file.write(report)

    plot_confusion_matrix(
        y_true, y_pred, class_names=class_names,
        save_path=str(args.results_dir / "confusion_matrix.png"), normalize=True,
        title="MLP - normalized ESC-50 confusion matrix (held-out fold 5)"
    )

    matrix = confusion_matrix(y_true, y_pred, labels=list(range(50)))
    np.fill_diagonal(matrix, 0)
    ranked_pairs = np.dstack(
        np.unravel_index(np.argsort(matrix.ravel())[::-1], matrix.shape)
    )[0]
    error_rows = []
    for true_index, predicted_index in ranked_pairs:
        count = int(matrix[true_index, predicted_index])
        if count == 0 or len(error_rows) == 15:
            break
        error_rows.append({
            "true_class": class_names[true_index],
            "predicted_class": class_names[predicted_index], "count": count,
        })
    pd.DataFrame(error_rows).to_csv(
        args.results_dir / "top_confusions.csv", index=False
    )

    print(f"Training completed in {training_seconds:.1f}s ({args.epochs} epochs)")
    print(f"Best epoch: {best_epoch}; fold 5 accuracy: {fold5_accuracy:.2%}")
    print(json.dumps(metrics, indent=2))
    print(f"Artifacts saved under: {args.results_dir}")


if __name__ == "__main__":
    main()
