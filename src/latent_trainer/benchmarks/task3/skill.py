from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score

from latent_trainer.benchmarks.common.data import (
    PlayerReplaySample,
    load_player_samples,
)
from latent_trainer.benchmarks.common.metrics import regression_metrics
from latent_trainer.benchmarks.common.models import (
    make_multiclass_classifier,
    make_regressor,
    predict_sequence_model,
    train_sequence_model,
)
from latent_trainer.benchmarks.common.splits import (
    assert_disjoint_groups,
    player_held_out_split,
)

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
    boundaries = np.unique(
        np.quantile(train_mmr, np.linspace(0, 1, class_count + 1)[1:-1])
    )
    return np.digitize(values, boundaries).astype(int), boundaries


def _classification_metrics(
    targets: np.ndarray,
    predictions: np.ndarray,
) -> dict[str, float]:
    return {
        "accuracy": float(accuracy_score(targets, predictions)),
        "balanced_accuracy": float(balanced_accuracy_score(targets, predictions)),
        "macro_f1": float(f1_score(targets, predictions, average="macro")),
    }


def run_skill_benchmark(
    json_path: Path,
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
) -> dict:
    samples = load_player_samples(json_path, max_replays=max_replays)
    data = build_skill_data(
        samples,
        include_race=include_race,
        mask_economy=mask_economy,
        mask_race=mask_race,
    )
    split = player_held_out_split(data.player_ids, data.replay_ids, seed=seed)
    assert_disjoint_groups(split, data.player_ids)
    assert_disjoint_groups(split, data.replay_ids)

    if objective == "regression":
        train_targets = data.mmr[split.train]
        test_targets = data.mmr[split.test]
    elif objective == "classification":
        all_classes, boundaries = quantile_classes(
            data.mmr[split.train], data.mmr, class_count=class_count
        )
        train_targets = all_classes[split.train]
        test_targets = all_classes[split.test]
    else:
        raise ValueError(f"Unknown skill objective: {objective}")

    if representation == "averaged":
        if objective == "regression":
            model = make_regressor(model_name, seed=seed)
            model.fit(data.features[split.train], train_targets)
            predictions = model.predict(data.features[split.test])
        else:
            model = make_multiclass_classifier(
                model_name,
                class_count=int(np.max(train_targets)) + 1,
                seed=seed,
            )
            model.fit(data.features[split.train], train_targets)
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
        predictions = predict_sequence_model(
            trained, [data.sequences[index] for index in split.test]
        )
    else:
        raise ValueError(f"Unknown representation: {representation}")

    if objective == "regression":
        metrics = regression_metrics(test_targets, predictions)
        class_boundaries = None
    else:
        metrics = _classification_metrics(test_targets, predictions)
        class_boundaries = boundaries.tolist()

    return {
        "task": "3A" if objective == "regression" else "3B",
        "objective": objective,
        "representation": representation,
        "model": model_name,
        "split": "player-held-out",
        "train_samples": int(len(split.train)),
        "test_samples": int(len(split.test)),
        "metrics": metrics,
        "quantile_boundaries": class_boundaries,
        "feature_groups": {
            "economy": not mask_economy,
            "race": include_race and not mask_race,
        },
    }
