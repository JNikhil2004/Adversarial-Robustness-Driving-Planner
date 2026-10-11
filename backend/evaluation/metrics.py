"""
metrics.py
-----------
Robustness benchmarking and classification metrics.
Computes:
  - Accuracy, Precision, Recall, F1 Score, Confusion Matrix
  - False Acceptance Rate (FAR) - critical safety failure: accepting adversarial inputs as safe
  - False Rejection Rate (FRR) - operational inefficiency: rejecting benign inputs
  - Score drops and summary statistics
"""

from typing import Dict, List, Any
import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix


def classification_report(y_true: List[str], y_pred: List[str], labels: List[str]) -> Dict[str, Any]:
    """Compute accuracy, per-class precision/recall/F1, confusion matrix, FAR, and FRR."""
    if not y_true or not y_pred:
        return {
            "accuracy": 0.0,
            "per_class": {},
            "confusion_matrix": [],
            "labels": labels,
            "far": 0.0,
            "frr": 0.0,
        }

    accuracy = accuracy_score(y_true, y_pred)
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, zero_division=0
    )
    cm = confusion_matrix(y_true, y_pred, labels=labels)

    per_class = {}
    for i, label in enumerate(labels):
        per_class[label] = {
            "precision": round(float(precision[i]), 3),
            "recall": round(float(recall[i]), 3),
            "f1_score": round(float(f1[i]), 3),
            "support": int(support[i]),
        }

    # FAR and FRR calculations
    far, frr = compute_far_frr(y_true, y_pred, safe_label="trusted")

    return {
        "accuracy": round(float(accuracy), 3),
        "per_class": per_class,
        "confusion_matrix": cm.tolist(),
        "labels": labels,
        "far": round(far, 4),
        "frr": round(frr, 4),
    }


def compute_far_frr(y_true: List[str], y_pred: List[str], safe_label: str = "trusted") -> tuple:
    """
    Calculate False Acceptance Rate (FAR) and False Rejection Rate (FRR).
    FAR: Malicious/attacked inputs accepted as safe (Safety Critical).
    FRR: Safe inputs rejected as unsafe/suspicious.
    """
    malicious_count = sum(1 for yt in y_true if yt != safe_label)
    false_acceptances = sum(1 for yt, yp in zip(y_true, y_pred) if yt != safe_label and yp == safe_label)
    far = float(false_acceptances / max(1, malicious_count)) if malicious_count > 0 else 0.0

    benign_count = sum(1 for yt in y_true if yt == safe_label)
    false_rejections = sum(1 for yt, yp in zip(y_true, y_pred) if yt == safe_label and yp != safe_label)
    frr = float(false_rejections / max(1, benign_count)) if benign_count > 0 else 0.0

    return far, frr


def robustness_drop(clean_score: float, attacked_score: float) -> float:
    """Percentage-point drop in robustness score after an attack is applied."""
    return round(clean_score - attacked_score, 2)


def summarize_scores(scores: List[float]) -> Dict[str, float]:
    """Mean/min/max helper used for aggregate benchmark reporting."""
    if not scores:
        return {"mean": 0.0, "min": 0.0, "max": 0.0}
    return {
        "mean": round(float(np.mean(scores)), 2),
        "min": round(float(np.min(scores)), 2),
        "max": round(float(np.max(scores)), 2),
    }
