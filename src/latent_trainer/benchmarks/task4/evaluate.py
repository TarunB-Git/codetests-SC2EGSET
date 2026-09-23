from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import accuracy_score

from latent_trainer.benchmarks.cache.dataset import ShardReplayDataset
from latent_trainer.benchmarks.common.metrics import (
    classification_metrics,
    regression_metrics,
)
from latent_trainer.benchmarks.task4.checkpoint import load_compatible_model
from latent_trainer.benchmarks.task4.data import (
    StreamingNormalization,
    load_compatible_split,
)


def _encode(
    dataset: ShardReplayDataset,
    indices: list[int],
    model: Any,
    normalization: StreamingNormalization,
) -> dict[str, Any]:
    latent = []
    reconstruction_errors = []
    outcomes = []
    ratings = []
    rating_validity = []
    identities = []
    replay_ids = []
    with torch.no_grad():
        for index in indices:
            sample = dataset[index]
            data = (sample["average"] - normalization.mean) / normalization.std
            mu, _ = model.model.encode(data.unsqueeze(0))
            reconstruction = model.model.decode(mu).squeeze(0)
            latent.append(mu.squeeze(0).reshape(-1).numpy())
            reconstruction_errors.append(
                torch.mean((reconstruction - data) ** 2).item()
            )
            outcomes.append(int(sample["outcomes"][0]))
            ratings.append(float(sample["raw_mmr"][0]))
            rating_validity.append(bool(sample["mmr_valid"][0]))
            identities.append(sample["toon_ids"][0])
            replay_ids.append(sample["replay_id"])
    return {
        "latent": np.asarray(latent),
        "reconstruction_errors": np.asarray(reconstruction_errors),
        "outcomes": np.asarray(outcomes),
        "ratings": np.asarray(ratings),
        "rating_validity": np.asarray(rating_validity),
        "identities": np.asarray(identities),
        "replay_ids": replay_ids,
    }


def _binary_probe(train: dict[str, Any], test: dict[str, Any]) -> dict[str, Any]:
    if np.unique(train["outcomes"]).size < 2:
        return {"status": "unavailable", "reason": "one training class"}
    probe = LogisticRegression(max_iter=1000).fit(train["latent"], train["outcomes"])
    probabilities = probe.predict_proba(test["latent"])[:, 1]
    return {
        "status": "ok",
        "metrics": classification_metrics(test["outcomes"], probabilities),
    }


def _mmr_probe(train: dict[str, Any], test: dict[str, Any]) -> dict[str, Any]:
    train_valid = train["rating_validity"]
    test_valid = test["rating_validity"]
    if train_valid.sum() < 2 or test_valid.sum() == 0:
        return {"status": "unavailable", "reason": "insufficient valid MMR"}
    probe = Ridge().fit(train["latent"][train_valid], train["ratings"][train_valid])
    predictions = probe.predict(test["latent"][test_valid])
    return {
        "status": "ok",
        "metrics": regression_metrics(test["ratings"][test_valid], predictions),
        "train_count": int(train_valid.sum()),
        "test_count": int(test_valid.sum()),
    }


def _skill_probe(
    train: dict[str, Any], test: dict[str, Any], class_count: int
) -> dict[str, Any]:
    train_valid = train["rating_validity"]
    test_valid = test["rating_validity"]
    ratings = train["ratings"][train_valid]
    if len(ratings) < class_count or test_valid.sum() == 0:
        return {"status": "unavailable", "reason": "insufficient valid MMR"}
    boundaries = np.unique(
        np.quantile(ratings, np.linspace(0, 1, class_count + 1)[1:-1])
    )
    train_labels = np.digitize(ratings, boundaries)
    test_labels = np.digitize(test["ratings"][test_valid], boundaries)
    if np.unique(train_labels).size < 2:
        return {"status": "unavailable", "reason": "one quantile class"}
    probe = LogisticRegression(max_iter=1000).fit(
        train["latent"][train_valid], train_labels
    )
    predictions = probe.predict(test["latent"][test_valid])
    support = np.bincount(test_labels, minlength=len(boundaries) + 1)
    return {
        "status": "ok",
        "accuracy": float(accuracy_score(test_labels, predictions)),
        "training_quantile_boundaries": boundaries.tolist(),
        "test_class_support": support.tolist(),
    }


def _identity_diagnostic(train: dict[str, Any], test: dict[str, Any]) -> dict[str, Any]:
    centroids = {}
    for identity in np.unique(train["identities"]):
        centroids[identity] = train["latent"][train["identities"] == identity].mean(0)
    known = np.asarray([value in centroids for value in test["identities"]])
    if not known.any():
        return {
            "status": "unavailable",
            "reason": "no test identities occur in training",
            "known_test_count": 0,
        }
    names = list(centroids)
    matrix = np.stack([centroids[name] for name in names])
    distances = ((test["latent"][known, None] - matrix[None]) ** 2).sum(-1)
    predicted = np.asarray(names)[distances.argmin(1)]
    return {
        "status": "ok",
        "accuracy": float(accuracy_score(test["identities"][known], predicted)),
        "known_test_count": int(known.sum()),
        "diagnostic_only": True,
    }


def _domain_probe(
    primary_train: dict[str, Any],
    primary_test: dict[str, Any],
    secondary_train: dict[str, Any],
    secondary_test: dict[str, Any],
) -> dict[str, Any]:
    train_features = np.concatenate(
        [primary_train["latent"], secondary_train["latent"]]
    )
    train_labels = np.concatenate(
        [
            np.zeros(len(primary_train["latent"])),
            np.ones(len(secondary_train["latent"])),
        ]
    )
    test_features = np.concatenate([primary_test["latent"], secondary_test["latent"]])
    test_labels = np.concatenate(
        [np.zeros(len(primary_test["latent"])), np.ones(len(secondary_test["latent"]))]
    )
    probe = LogisticRegression(max_iter=1000).fit(train_features, train_labels)
    return {
        "status": "ok",
        "accuracy": float(accuracy_score(test_labels, probe.predict(test_features))),
        "interpretation": "held-out SC2GGSet-versus-SC2EGSet separation diagnostic",
    }


def evaluate_guided_vae(
    cache_manifest_path: Path,
    split_path: Path,
    checkpoint_path: Path,
    normalization_path: Path,
    class_count: int = 4,
    secondary_cache_manifest_path: Path | None = None,
    secondary_split_path: Path | None = None,
) -> dict[str, Any]:
    dataset = ShardReplayDataset(cache_manifest_path)
    split = load_compatible_split(dataset, split_path)
    normalization = StreamingNormalization.load(normalization_path)
    model = load_compatible_model(
        checkpoint_path, dataset, split, normalization.fingerprint()
    )
    encoded = {
        name: _encode(
            dataset,
            split.splits[name]["dataset_indices"],
            model,
            normalization,
        )
        for name in ("train", "validation", "test")
    }
    domain: dict[str, Any] = {
        "status": "unavailable",
        "reason": "secondary dataset and held-out split were not supplied",
    }
    if secondary_cache_manifest_path is not None and secondary_split_path is not None:
        secondary = ShardReplayDataset(secondary_cache_manifest_path)
        secondary_split = load_compatible_split(secondary, secondary_split_path)
        secondary_encoded = {
            name: _encode(
                secondary,
                secondary_split.splits[name]["dataset_indices"],
                model,
                normalization,
            )
            for name in ("train", "test")
        }
        domain = _domain_probe(
            encoded["train"],
            encoded["test"],
            secondary_encoded["train"],
            secondary_encoded["test"],
        )
    test_errors = encoded["test"]["reconstruction_errors"]
    return {
        "provenance": {
            "cache_fingerprint": dataset.manifest.fingerprint(),
            "split_fingerprint": split.fingerprint(),
            "checkpoint": checkpoint_path.name,
        },
        "reconstruction": {
            "mse_mean": float(test_errors.mean()) if len(test_errors) else math.nan,
            "mse_median": float(np.median(test_errors))
            if len(test_errors)
            else math.nan,
            "sample_count": len(test_errors),
        },
        "win_loss_probe": _binary_probe(encoded["train"], encoded["test"]),
        "mmr_probe": _mmr_probe(encoded["train"], encoded["test"]),
        "quantile_skill_probe": _skill_probe(
            encoded["train"], encoded["test"], class_count
        ),
        "identity_retrieval": _identity_diagnostic(encoded["train"], encoded["test"]),
        "domain_separation": domain,
        "held_out_only": True,
    }


def latent_stability(reference: np.ndarray, comparison: np.ndarray) -> dict[str, float]:
    if reference.shape != comparison.shape or reference.ndim != 2:
        raise ValueError(
            "Latent stability arrays must have matching two-dimensional shape"
        )
    correlations = []
    for index in range(reference.shape[1]):
        left = reference[:, index]
        right = comparison[:, index]
        if np.std(left) and np.std(right):
            correlations.append(float(np.corrcoef(left, right)[0, 1]))
    return {
        "mean_dimension_correlation": float(np.mean(correlations))
        if correlations
        else math.nan,
        "compared_dimensions": len(correlations),
    }
