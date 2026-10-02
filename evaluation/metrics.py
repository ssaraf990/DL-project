"""
Evaluation Metrics Utilities for Environmental Sound Classification.

Computes overall classification accuracy, macro/weighted precision,
macro/weighted recall, and macro/weighted F1-scores for ESC-50 predictions.
"""

from typing import Dict
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
)


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """
    Compute comprehensive classification metrics for model evaluation.

    Args:
        y_true (np.ndarray): 1D array of ground truth class integer labels (0 to 49).
        y_pred (np.ndarray): 1D array of predicted class integer labels (0 to 49).

    Returns:
        Dict[str, float]: Dictionary containing computed evaluation metrics:
            - 'accuracy': Overall classification accuracy.
            - 'precision_macro': Unweighted macro-averaged precision.
            - 'precision_weighted': Support-weighted precision.
            - 'recall_macro': Unweighted macro-averaged recall.
            - 'recall_weighted': Support-weighted recall.
            - 'f1_macro': Unweighted macro-averaged F1-score.
            - 'f1_weighted': Support-weighted F1-score.
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    acc = float(accuracy_score(y_true, y_pred))
    p_macro = float(precision_score(y_true, y_pred, average="macro", zero_division=0))
    p_weighted = float(precision_score(y_true, y_pred, average="weighted", zero_division=0))
    r_macro = float(recall_score(y_true, y_pred, average="macro", zero_division=0))
    r_weighted = float(recall_score(y_true, y_pred, average="weighted", zero_division=0))
    f1_mac = float(f1_score(y_true, y_pred, average="macro", zero_division=0))
    f1_weight = float(f1_score(y_true, y_pred, average="weighted", zero_division=0))

    return {
        "accuracy": acc,
        "precision_macro": p_macro,
        "precision_weighted": p_weighted,
        "recall_macro": r_macro,
        "recall_weighted": r_weighted,
        "f1_macro": f1_mac,
        "f1_weighted": f1_weight,
    }
