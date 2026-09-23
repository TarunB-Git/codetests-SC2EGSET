import math

import numpy as np
from scipy.stats import spearmanr
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    brier_score_loss,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    roc_auc_score,
)


def classification_metrics(
    targets: np.ndarray,
    probabilities: np.ndarray,
    threshold: float = 0.5,
) -> dict[str, float]:
    targets = np.asarray(targets, dtype=int)
    probabilities = np.asarray(probabilities, dtype=float)
    predictions = (probabilities >= threshold).astype(int)
    has_two_classes = np.unique(targets).size == 2
    auroc = (
        float(roc_auc_score(targets, probabilities)) if has_two_classes else math.nan
    )
    balanced_accuracy = (
        float(balanced_accuracy_score(targets, predictions))
        if has_two_classes
        else math.nan
    )
    return {
        "accuracy": float(accuracy_score(targets, predictions)),
        "balanced_accuracy": balanced_accuracy,
        "auroc": auroc,
        "f1": float(f1_score(targets, predictions, zero_division=0)),
        "brier": float(brier_score_loss(targets, probabilities)),
    }


def regression_metrics(
    targets: np.ndarray,
    predictions: np.ndarray,
) -> dict[str, float]:
    targets = np.asarray(targets, dtype=float)
    predictions = np.asarray(predictions, dtype=float)
    correlation = (
        spearmanr(targets, predictions).statistic
        if np.unique(targets).size > 1 and np.unique(predictions).size > 1
        else math.nan
    )
    return {
        "mae": float(mean_absolute_error(targets, predictions)),
        "rmse": float(mean_squared_error(targets, predictions) ** 0.5),
        "spearman": float(correlation),
    }


def reliability_bins(
    targets: np.ndarray,
    probabilities: np.ndarray,
    n_bins: int = 10,
) -> list[dict[str, float | int]]:
    targets = np.asarray(targets, dtype=int)
    probabilities = np.asarray(probabilities, dtype=float)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    assignments = np.minimum(np.digitize(probabilities, edges[1:-1]), n_bins - 1)
    result = []

    for index in range(n_bins):
        selected = assignments == index
        if not selected.any():
            continue
        result.append(
            {
                "lower": float(edges[index]),
                "upper": float(edges[index + 1]),
                "count": int(selected.sum()),
                "mean_probability": float(probabilities[selected].mean()),
                "observed_rate": float(targets[selected].mean()),
            }
        )

    return result
