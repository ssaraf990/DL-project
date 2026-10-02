"""
Confusion Matrix Visualization Utilities.

Generates, plots, and saves class-level confusion matrix heatmaps
for 50-class ESC-50 predictions using Seaborn and Matplotlib.
"""

import os
from typing import List, Optional
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix


def plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: Optional[List[str]] = None,
    save_path: Optional[str] = None,
    normalize: bool = True,
    title: str = "ESC-50 Confusion Matrix",
    figsize: tuple = (16, 14),
) -> np.ndarray:
    """
    Generate and plot a formatted confusion matrix heatmap.

    Args:
        y_true (np.ndarray): Ground truth integer labels (0 to 49).
        y_pred (np.ndarray): Predicted integer labels (0 to 49).
        class_names (Optional[List[str]]): List of 50 ESC-50 class names.
                                            If None, uses class indices 0..49.
        save_path (Optional[str]): Path to save the generated image file.
        normalize (bool): Whether to normalize rows by class support count.
        title (str): Title header for the plot.
        figsize (tuple): Figure dimensions (width, height) in inches.

    Returns:
        np.ndarray: Computed 2D confusion matrix array.
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    n_classes = 50
    if class_names is None:
        labels = list(range(n_classes))
        class_names = [str(i) for i in range(n_classes)]
    else:
        labels = list(range(len(class_names)))

    cm = confusion_matrix(y_true, y_pred, labels=labels)

    cm_display = cm.astype("float")
    if normalize:
        row_sums = cm_display.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1.0  # Prevent division by zero
        cm_display = cm_display / row_sums

    fig, ax = plt.subplots(figsize=figsize)

    sns.heatmap(
        cm_display,
        annot=False,
        fmt=".2f" if normalize else "d",
        cmap="Blues",
        cbar=True,
        xticklabels=class_names,
        yticklabels=class_names,
        ax=ax,
    )

    ax.set_title(title, fontsize=14, pad=12)
    ax.set_xlabel("Predicted Class Label", fontsize=12)
    ax.set_ylabel("True Class Label", fontsize=12)
    plt.xticks(rotation=90, fontsize=8)
    plt.yticks(rotation=0, fontsize=8)
    plt.tight_layout()

    if save_path:
        out_dir = os.path.dirname(save_path)
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)
        plt.savefig(save_path, bbox_inches="tight", dpi=300)

    plt.close(fig)
    return cm
