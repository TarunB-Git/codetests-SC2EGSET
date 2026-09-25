from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from sklearn.base import BaseEstimator
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold, cross_val_predict
from sklearn.preprocessing import StandardScaler

from latent_trainer.benchmarks.cache.dataset import ShardReplayDataset
from latent_trainer.benchmarks.cache.manifest import CacheManifest
from latent_trainer.benchmarks.common.metrics import (
    classification_metrics,
    reliability_bins,
)
from latent_trainer.benchmarks.common.models import (
    make_classifier,
    positive_probabilities,
)
from latent_trainer.benchmarks.common.provenance import software_provenance
from latent_trainer.benchmarks.splits.manifest import SplitManifest


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
        if labels[split].ndim != 2 or labels[split].shape[1] != 2:
            raise ValueError(f"{split} labels must have shape [N, 2]")
        if len(features[split]) != len(labels[split]):
            raise ValueError(f"{split} features and labels must have equal length")
        if not np.isin(labels[split], (0, 1)).all():
            raise ValueError(f"{split} labels must be binary")

    return AlignedCache(
        train_features=features["train"],
        train_labels=labels["train"],
        validation_features=features["val"],
        validation_labels=labels["val"],
        test_features=features["test"],
        test_labels=labels["test"],
        transform=str(cache.get("transform", "")),
    )


def load_sharded_aligned_cache(
    manifest_path: Path,
    split_path: Path | None,
    protocol: str,
) -> AlignedCache:
    dataset = ShardReplayDataset(manifest_path)
    if protocol == "historical":
        indices = [
            index
            for index in range(len(dataset))
            if dataset[index]["duration_loops"] > 12000
        ]
        partitions = {"train": indices, "validation": [], "test": []}
        transform = "historical_averaged_economy"
    else:
        if split_path is None:
            raise ValueError("Corrected sharded Task 1A requires a split manifest")
        split = SplitManifest.load(split_path)
        if split.cache_fingerprint != dataset.manifest.fingerprint():
            raise ValueError("Split does not belong to the Task 1A cache")
        partitions = {
            name: split.splits[name]["dataset_indices"]
            for name in ("train", "validation", "test")
        }
        transform = "averaged_economy"

    def arrays(indices: list[int]) -> tuple[np.ndarray, np.ndarray]:
        if not indices:
            return (
                np.empty((0, 2, len(dataset.manifest.feature_names)), dtype=np.float32),
                np.empty((0, 2), dtype=int),
            )
        features = []
        labels = []
        for index in sorted(indices):
            sample = dataset[index]
            features.append(sample["average"].clone())
            labels.append(sample["outcomes"].clone())
        return torch.stack(features).numpy(), torch.stack(labels).numpy()

    train_features, train_labels = arrays(partitions["train"])
    validation_features, validation_labels = arrays(partitions["validation"])
    test_features, test_labels = arrays(partitions["test"])
    return AlignedCache(
        train_features=train_features,
        train_labels=train_labels,
        validation_features=validation_features,
        validation_labels=validation_labels,
        test_features=test_features,
        test_labels=test_labels,
        transform=transform,
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


def _evaluate_model(
    model: BaseEstimator,
    evaluation_features: np.ndarray,
    evaluation_labels: np.ndarray,
) -> dict:
    probabilities = positive_probabilities(model, evaluation_features)
    return {
        "metrics": classification_metrics(evaluation_labels, probabilities),
        "calibration": reliability_bins(evaluation_labels, probabilities),
        "samples": int(len(evaluation_labels)),
        "class_counts": {
            str(value): int((evaluation_labels == value).sum())
            for value in np.unique(evaluation_labels)
        },
    }


def _validation_calibrated_probabilities(
    model: BaseEstimator,
    validation_features: np.ndarray,
    validation_labels: np.ndarray,
    test_features: np.ndarray,
) -> np.ndarray:
    if np.unique(validation_labels).size != 2:
        raise ValueError("Calibration requires two validation outcome classes")
    epsilon = 1e-6
    validation = positive_probabilities(model, validation_features)
    test = positive_probabilities(model, test_features)
    validation_logits = np.log(
        np.clip(validation, epsilon, 1 - epsilon)
        / np.clip(1 - validation, epsilon, 1 - epsilon)
    ).reshape(-1, 1)
    test_logits = np.log(
        np.clip(test, epsilon, 1 - epsilon) / np.clip(1 - test, epsilon, 1 - epsilon)
    ).reshape(-1, 1)
    calibrator = LogisticRegression().fit(validation_logits, validation_labels)
    return calibrator.predict_proba(test_logits)[:, 1]


def run_static_benchmark(
    cache_path: Path,
    models: tuple[str, ...] = ("logistic", "svm", "xgboost", "mlp"),
    view: str = "one-player",
    protocol: str = "corrected",
    calibrated: bool = False,
    seed: int = 42,
    evaluation_cache_path: Path | None = None,
    split_path: Path | None = None,
    evaluation_split_path: Path | None = None,
) -> dict:
    cache = (
        load_sharded_aligned_cache(cache_path, split_path, protocol)
        if cache_path.suffix == ".json"
        else load_aligned_cache(cache_path)
    )

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
        if cache.transform != "averaged_economy":
            raise ValueError("Corrected Task 1A requires an averaged_economy cache")
        train_features, train_labels = tabular_view(
            cache.train_features, cache.train_labels, view
        )
        validation_features, validation_labels = tabular_view(
            cache.validation_features, cache.validation_labels, view
        )
        test_features, test_labels = tabular_view(
            cache.test_features, cache.test_labels, view
        )
        if not len(validation_labels) or not len(test_labels):
            raise ValueError("Corrected Task 1A requires non-empty validation and test")
        if np.unique(train_labels).size != 2:
            raise ValueError("Task 1A training data requires both outcome classes")
        results = {}
        fitted_models = {}
        for model_name in models:
            model = make_classifier(
                model_name,
                seed=seed,
                calibrated=False,
                historical=False,
            )
            model.fit(train_features, train_labels)
            fitted_models[model_name] = model
            validation_result = _evaluate_model(
                model,
                validation_features,
                validation_labels,
            )
            if calibrated:
                test_probabilities = _validation_calibrated_probabilities(
                    model,
                    validation_features,
                    validation_labels,
                    test_features,
                )
                test_result = {
                    "metrics": classification_metrics(test_labels, test_probabilities),
                    "calibration": reliability_bins(test_labels, test_probabilities),
                    "samples": int(len(test_labels)),
                    "class_counts": {
                        str(value): int((test_labels == value).sum())
                        for value in np.unique(test_labels)
                    },
                }
            else:
                test_result = _evaluate_model(model, test_features, test_labels)
            results[model_name] = {
                "validation": validation_result,
                "test": test_result,
            }
        if evaluation_cache_path is not None:
            evaluation_cache = (
                load_sharded_aligned_cache(
                    evaluation_cache_path, evaluation_split_path, "corrected"
                )
                if evaluation_cache_path.suffix == ".json"
                else load_aligned_cache(evaluation_cache_path)
            )
            if evaluation_cache.transform != "averaged_economy":
                raise ValueError(
                    "Cross-dataset evaluation requires an averaged_economy cache"
                )
            if (
                evaluation_cache.test_features.shape[-1]
                != cache.train_features.shape[-1]
            ):
                raise ValueError("Training and evaluation feature widths differ")
            evaluation_features, evaluation_labels = tabular_view(
                evaluation_cache.test_features,
                evaluation_cache.test_labels,
                view,
            )
            for model_name, model in fitted_models.items():
                if calibrated:
                    probabilities = _validation_calibrated_probabilities(
                        model,
                        validation_features,
                        validation_labels,
                        evaluation_features,
                    )
                    results[model_name]["cross_dataset"] = {
                        "metrics": classification_metrics(
                            evaluation_labels, probabilities
                        ),
                        "calibration": reliability_bins(
                            evaluation_labels, probabilities
                        ),
                        "samples": int(len(evaluation_labels)),
                        "class_counts": {
                            str(value): int((evaluation_labels == value).sum())
                            for value in np.unique(evaluation_labels)
                        },
                    }
                else:
                    results[model_name]["cross_dataset"] = _evaluate_model(
                        model,
                        evaluation_features,
                        evaluation_labels,
                    )
    else:
        raise ValueError(f"Unknown protocol: {protocol}")

    return {
        "task": "1A",
        "protocol": protocol,
        "evaluation_design": (
            {
                "purpose": "historical-reproduction",
                "split": "shuffled-row-kfold",
                "preprocessing": "full-dataset-scaling",
            }
            if protocol == "historical"
            else {
                "purpose": "leakage-resistant-benchmark",
                "split": "replay-grouped-cache-splits",
                "preprocessing": "train-only",
                "calibration": "validation-fitted" if calibrated else "none",
            }
        ),
        "view": view,
        "transform": cache.transform,
        "evaluation_cache": str(evaluation_cache_path)
        if evaluation_cache_path is not None
        else None,
        "evaluation_split": str(evaluation_split_path)
        if evaluation_split_path is not None
        else None,
        "models": results,
        "provenance": {
            "software": software_provenance(),
            "cache_fingerprint": CacheManifest.load(cache_path).fingerprint()
            if cache_path.suffix == ".json"
            else None,
            "split_fingerprint": SplitManifest.load(split_path).fingerprint()
            if split_path is not None
            else None,
            "evaluation_cache_fingerprint": CacheManifest.load(
                evaluation_cache_path
            ).fingerprint()
            if evaluation_cache_path is not None
            and evaluation_cache_path.suffix == ".json"
            else None,
            "evaluation_split_fingerprint": SplitManifest.load(
                evaluation_split_path
            ).fingerprint()
            if evaluation_split_path is not None
            else None,
        },
    }
