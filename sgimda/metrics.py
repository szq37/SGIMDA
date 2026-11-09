from typing import Dict

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)

def select_f1_threshold(labels: np.ndarray, scores: np.ndarray) -> float:
    precision, recall, thresholds = precision_recall_curve(labels, scores)
    f1 = 2.0 * precision[:-1] * recall[:-1] / np.maximum(
        precision[:-1] + recall[:-1], 1e-12
    )
    return float(thresholds[int(np.nanargmax(f1))]) if len(thresholds) else 0.5

def classification_metrics(
    labels: np.ndarray, scores: np.ndarray, threshold: float
) -> Dict[str, float]:
    if not np.isfinite(scores).all():
        raise ValueError("predictions contain nonfinite values")
    prediction = (scores >= threshold).astype(np.int64)
    return {
        "auc": float(roc_auc_score(labels, scores)),
        "aupr": float(average_precision_score(labels, scores)),
        "accuracy": float(accuracy_score(labels, prediction)),
        "precision": float(precision_score(labels, prediction, zero_division=0)),
        "recall": float(recall_score(labels, prediction, zero_division=0)),
        "f1": float(f1_score(labels, prediction, zero_division=0)),
    }
