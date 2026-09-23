from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import accuracy_score, f1_score, recall_score

from latent_trainer.benchmarks.cache.manifest import CacheManifest
from latent_trainer.benchmarks.common.data import (
    PlayerReplaySample,
    load_cached_player_samples,
    load_player_samples,
)
from latent_trainer.benchmarks.common.metrics import regression_metrics
from latent_trainer.benchmarks.common.models import (
    make_multiclass_classifier,
    make_regressor,
    predict_sequence_model,
    train_sequence_model,
)
from latent_trainer.benchmarks.common.provenance import software_provenance
from latent_trainer.benchmarks.common.splits import (
    assert_disjoint_groups,
    grouped_split,
    indices_from_split_manifest,
    player_held_out_split,
)
from latent_trainer.benchmarks.data.source import InputFormat
from latent_trainer.benchmarks.splits.manifest import SplitManifest

RACES = ("Prot", "Terr", "Zerg")


@dataclass(frozen=True)
class SkillData:
    features: np.ndarray
    sequences: list[torch.Tensor]
    mmr: np.ndarray
    player_ids: list[str]
    replay_ids: list[str]


def _race_features(race: str) -> np.ndarray:
    return np.asarray([float(race.startswith(value)) for value in RACES])


def build_skill_data(
    samples: list[PlayerReplaySample],
    include_race: bool = False,
    mask_economy: bool = False,
    mask_race: bool = False,
) -> SkillData:
    valid = [sample for sample in samples if sample.mmr > 0]
    if not valid:
        raise ValueError("No samples have positive MMR")
    if any(not sample.player_toon for sample in valid):
        raise ValueError("Player-held-out evaluation requires persistent Toon IDs")
    if mask_economy and (not include_race or mask_race):
        raise ValueError("At least one metadata-blind feature group must remain")

    features = []
    sequences = []
    for sample in valid:
        static_parts = []
        if not mask_economy:
            static_parts.append(sample.average.numpy())
        if include_race and not mask_race:
            static_parts.append(_race_features(sample.race))
        features.append(np.concatenate(static_parts))

        sequence_parts = []
        if not mask_economy:
            sequence_parts.append(sample.sequence)
        if include_race and not mask_race:
            race = torch.tensor(_race_features(sample.race), dtype=torch.float32)
            sequence_parts.append(race.unsqueeze(0).expand(len(sample.sequence), -1))
        sequences.append(torch.cat(sequence_parts, dim=1))

    return SkillData(
        features=np.asarray(features, dtype=np.float32),
        sequences=sequences,
        mmr=np.asarray([sample.mmr for sample in valid], dtype=np.float32),
        player_ids=[sample.player_toon for sample in valid],
        replay_ids=[sample.replay_id for sample in valid],
    )


def quantile_classes(
    train_mmr: np.ndarray,
    values: np.ndarray,
    class_count: int = 4,
) -> tuple[np.ndarray, np.ndarray]:
    if class_count < 2:
        raise ValueError("class_count must be at least two")
    boundaries = np.unique(
        np.quantile(train_mmr, np.linspace(0, 1, class_count + 1)[1:-1])
    )
    return np.digitize(values, boundaries).astype(int), boundaries


def _classification_metrics(
    targets: np.ndarray,
    predictions: np.ndarray,
) -> dict[str, float]:
    labels = np.unique(targets)
    return {
        "accuracy": float(accuracy_score(targets, predictions)),
        "balanced_accuracy": float(
            recall_score(
                targets,
                predictions,
                labels=labels,
                average="macro",
                zero_division=0,
            )
        ),
        "macro_f1": float(
            f1_score(
                targets,
                predictions,
                labels=labels,
                average="macro",
                zero_division=0,
            )
        ),
    }


def run_skill_benchmark(
    json_path: Path | None,
    objective: str,
    representation: str,
    model_name: str,
    include_race: bool = False,
    mask_economy: bool = False,
    mask_race: bool = False,
    class_count: int = 4,
    epochs: int = 10,
    max_replays: int = 0,
    seed: int = 42,
    split_strategy: str = "player-held-out",
    offsets_path: Path | None = None,
    input_format: InputFormat = "auto",
    source_indices_path: Path | None = None,
    cache_manifest_path: Path | None = None,
    split_path: Path | None = None,
) -> dict:
    valid_models = {
        "averaged": {"xgboost", "mlp"},
        "sequence": {"gru", "transformer"},
    }
    if representation not in valid_models:
        raise ValueError(f"Unknown representation: {representation}")
    if model_name not in valid_models[representation]:
        raise ValueError(
            f"{representation} representation does not support {model_name}"
        )
    if class_count < 2:
        raise ValueError("class_count must be at least two")
    persisted_strategy = None
    if cache_manifest_path is not None:
        samples = load_cached_player_samples(cache_manifest_path, max_replays)
        if split_path is not None:
            persisted_split = SplitManifest.load(split_path)
            persisted_strategy = persisted_split.strategy
            if persisted_split.strategy == "focal-player-held-out":
                samples = samples[::2]
    elif json_path is not None:
        samples = load_player_samples(
            json_path,
            max_replays=max_replays,
            offsets_path=offsets_path,
            input_format=input_format,
            source_indices_path=source_indices_path,
        )
    else:
        raise ValueError("Task 3 requires a JSON source or cache manifest")
    data = build_skill_data(
        samples,
        include_race=include_race,
        mask_economy=mask_economy,
        mask_race=mask_race,
    )
    if split_path is not None:
        cache_fingerprint = (
            CacheManifest.load(cache_manifest_path).fingerprint()
            if cache_manifest_path is not None
            else None
        )
        split = indices_from_split_manifest(
            data.replay_ids, split_path, cache_fingerprint
        )
        if persisted_strategy in {
            "focal-player-held-out",
            "strict-player-disjoint",
        }:
            assert_disjoint_groups(split, data.player_ids)
        split_strategy = str(split_path)
    elif split_strategy == "player-held-out":
        split = player_held_out_split(data.player_ids, data.replay_ids, seed=seed)
        assert_disjoint_groups(split, data.player_ids)
    elif split_strategy == "replay-grouped":
        split = grouped_split(data.replay_ids, seed=seed)
    else:
        raise ValueError(f"Unknown Task 3 split strategy: {split_strategy}")
    assert_disjoint_groups(split, data.replay_ids)
    if not len(split.train) or not len(split.validation) or not len(split.test):
        raise ValueError("Player-held-out splitting produced an empty partition")

    if objective == "regression":
        train_targets = data.mmr[split.train]
        validation_targets = data.mmr[split.validation]
        test_targets = data.mmr[split.test]
    elif objective == "classification":
        all_classes, boundaries = quantile_classes(
            data.mmr[split.train], data.mmr, class_count=class_count
        )
        train_targets = all_classes[split.train]
        validation_targets = all_classes[split.validation]
        test_targets = all_classes[split.test]
        if np.unique(train_targets).size < 2:
            raise ValueError("Skill classification training requires two classes")
    else:
        raise ValueError(f"Unknown skill objective: {objective}")

    if representation == "averaged":
        if objective == "regression":
            model = make_regressor(model_name, seed=seed)
            model.fit(data.features[split.train], train_targets)
            validation_predictions = model.predict(data.features[split.validation])
            predictions = model.predict(data.features[split.test])
        else:
            model = make_multiclass_classifier(
                model_name,
                class_count=int(np.max(train_targets)) + 1,
                seed=seed,
            )
            model.fit(data.features[split.train], train_targets)
            validation_predictions = model.predict(data.features[split.validation])
            predictions = model.predict(data.features[split.test])
    elif representation == "sequence":
        trained = train_sequence_model(
            [data.sequences[index] for index in split.train],
            train_targets,
            name=model_name,
            task="multiclass" if objective == "classification" else objective,
            epochs=epochs,
            seed=seed,
            num_classes=int(np.max(train_targets)) + 1,
        )
        validation_predictions = predict_sequence_model(
            trained, [data.sequences[index] for index in split.validation]
        )
        predictions = predict_sequence_model(
            trained, [data.sequences[index] for index in split.test]
        )
    else:
        raise ValueError(f"Unknown representation: {representation}")

    if objective == "regression":
        validation_metrics = regression_metrics(
            validation_targets, validation_predictions
        )
        metrics = regression_metrics(test_targets, predictions)
        class_boundaries = None
    else:
        validation_metrics = _classification_metrics(
            validation_targets, validation_predictions
        )
        metrics = _classification_metrics(test_targets, predictions)
        class_boundaries = boundaries.tolist()

    return {
        "task": "3A" if objective == "regression" else "3B",
        "objective": objective,
        "representation": representation,
        "model": model_name,
        "split": split_strategy,
        "train_samples": int(len(split.train)),
        "validation_samples": int(len(split.validation)),
        "test_samples": int(len(split.test)),
        "validation_metrics": validation_metrics,
        "metrics_split": "test",
        "metrics": metrics,
        "quantile_boundaries": class_boundaries,
        "class_support": (
            {
                "train": np.bincount(train_targets.astype(int)).tolist(),
                "validation": np.bincount(validation_targets.astype(int)).tolist(),
                "test": np.bincount(test_targets.astype(int)).tolist(),
            }
            if objective == "classification"
            else None
        ),
        "feature_groups": {
            "economy": not mask_economy,
            "race": include_race and not mask_race,
        },
        "provenance": {
            "software": software_provenance(),
            "cache_fingerprint": CacheManifest.load(cache_manifest_path).fingerprint()
            if cache_manifest_path is not None
            else None,
        },
    }
