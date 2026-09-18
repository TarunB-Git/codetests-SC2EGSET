from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from sklearn.model_selection import KFold, cross_val_predict
from sklearn.preprocessing import StandardScaler

from latent_trainer.benchmarks.common.metrics import (
    classification_metrics,
    reliability_bins,
)
from latent_trainer.benchmarks.common.models import (
    make_classifier,
    positive_probabilities,
)


@dataclass(frozen=True)
class AlignedCache:
    train_features: np.ndarray
    train_labels: np.ndarray
    validation_features: np.ndarray
    validation_labels: np.ndarray
    test_features: np.ndarray
    test_labels: np.ndarray
    transform: str


def load_aligned_cache(path: Path) -> AlignedCache:
    cache = torch.load(path, map_location="cpu", weights_only=False)
    features = {
        split: cache[f"{split}_features"].cpu().numpy()
        for split in ("train", "val", "test")
    }
    labels = {
        split: cache[f"{split}_labels"].cpu().numpy()
        for split in ("train", "val", "test")
    }

    for split in ("train", "val", "test"):
        if features[split].ndim != 3 or features[split].shape[1] != 2:
            raise ValueError(f"{split} features must have shape [N, 2, F]")

    return AlignedCache(
        train_features=features["train"],
        train_labels=labels["train"],
        validation_features=features["val"],
        validation_labels=labels["val"],
        test_features=features["test"],
        test_labels=labels["test"],
        transform=str(cache.get("transform", "")),
    )


def tabular_view(
    features: np.ndarray,
    labels: np.ndarray,
    view: str,
) -> tuple[np.ndarray, np.ndarray]:
    if labels.ndim != 2 or labels.shape[1] != 2:
        raise ValueError("Aligned per-player labels with shape [N, 2] are required")
    if view == "one-player":
        return features.reshape(-1, features.shape[-1]), labels.reshape(-1)
    if view == "two-player":
        return features.reshape(len(features), -1), labels[:, 0]
    raise ValueError(f"Unknown tabular view: {view}")


def _historical_result(
    features: np.ndarray,
    labels: np.ndarray,
    model_name: str,
    seed: int,
) -> dict:
    scaled = StandardScaler().fit_transform(features)
    folds = min(5, len(labels))
    if folds < 2:
        raise ValueError("At least two samples are required")
    model = make_classifier(model_name, seed=seed, historical=True)
    predictions = cross_val_predict(
        model,
        scaled,
        labels,
        cv=KFold(n_splits=folds, shuffle=True, random_state=seed),
        method="predict_proba",
        n_jobs=1,
    )[:, 1]
    return {
        "metrics": classification_metrics(labels, predictions),
        "calibration": reliability_bins(labels, predictions),
        "samples": int(len(labels)),
    }


def _corrected_result(
    train_features: np.ndarray,
    train_labels: np.ndarray,
    evaluation_features: np.ndarray,
    evaluation_labels: np.ndarray,
    model_name: str,
    seed: int,
    calibrated: bool,
) -> dict:
    model = make_classifier(
        model_name, seed=seed, calibrated=calibrated, historical=False
    )
    model.fit(train_features, train_labels)
    probabilities = positive_probabilities(model, evaluation_features)
    return {
        "metrics": classification_metrics(evaluation_labels, probabilities),
        "calibration": reliability_bins(evaluation_labels, probabilities),
        "samples": int(len(evaluation_labels)),
    }


def run_static_benchmark(
    cache_path: Path,
    models: tuple[str, ...] = ("logistic", "svm", "xgboost", "mlp"),
    view: str = "one-player",
    protocol: str = "corrected",
    calibrated: bool = False,
    seed: int = 42,
    evaluation_cache_path: Path | None = None,
) -> dict:
    cache = load_aligned_cache(cache_path)

    if protocol == "historical":
        if cache.transform != "historical_averaged_economy":
            raise ValueError(
                "Historical results require a historical_averaged_economy cache"
            )
        all_features = np.concatenate(
            [
                cache.train_features,
                cache.validation_features,
                cache.test_features,
            ]
        )
        all_labels = np.concatenate(
            [cache.train_labels, cache.validation_labels, cache.test_labels]
        )
        features, labels = tabular_view(all_features, all_labels, view)
        results = {
            model: _historical_result(features, labels, model, seed) for model in models
        }
    elif protocol == "corrected":
        train_features, train_labels = tabular_view(
            cache.train_features, cache.train_labels, view
        )
        validation_features, validation_labels = tabular_view(
            cache.validation_features, cache.validation_labels, view
        )
        test_features, test_labels = tabular_view(
            cache.test_features, cache.test_labels, view
        )
        results = {}
        for model in models:
            results[model] = {
                "validation": _corrected_result(
                    train_features,
                    train_labels,
                    validation_features,
                    validation_labels,
                    model,
                    seed,
                    calibrated,
                ),
                "test": _corrected_result(
                    train_features,
                    train_labels,
                    test_features,
                    test_labels,
                    model,
                    seed,
                    calibrated,
                ),
            }
        if evaluation_cache_path is not None:
            evaluation_cache = load_aligned_cache(evaluation_cache_path)
            evaluation_features, evaluation_labels = tabular_view(
                evaluation_cache.test_features,
                evaluation_cache.test_labels,
                view,
            )
            for model in models:
                results[model]["cross_dataset"] = _corrected_result(
                    train_features,
                    train_labels,
                    evaluation_features,
                    evaluation_labels,
                    model,
                    seed,
                    calibrated,
                )
    else:
        raise ValueError(f"Unknown protocol: {protocol}")

    return {
        "task": "1A",
        "protocol": protocol,
        "view": view,
        "transform": cache.transform,
        "evaluation_cache": str(evaluation_cache_path)
        if evaluation_cache_path is not None
        else None,
        "models": results,
    }
