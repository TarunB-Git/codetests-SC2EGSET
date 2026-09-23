from pathlib import Path

import numpy as np

from latent_trainer.benchmarks.cache.dataset import (
    PlayerSequenceDataset,
    ShardReplayDataset,
)
from latent_trainer.benchmarks.common.data import load_player_samples
from latent_trainer.benchmarks.common.metrics import (
    classification_metrics,
    reliability_bins,
)
from latent_trainer.benchmarks.common.models import (
    predict_sequence_dataset,
    predict_sequence_model,
    train_sequence_dataset,
    train_sequence_model,
)
from latent_trainer.benchmarks.common.provenance import software_provenance
from latent_trainer.benchmarks.common.splits import (
    assert_disjoint_groups,
    grouped_split,
    indices_from_split_manifest,
)
from latent_trainer.benchmarks.data.source import InputFormat
from latent_trainer.benchmarks.splits.manifest import SplitManifest


def _run_cached_sequence_benchmark(
    cache_manifest_path: Path,
    split_path: Path,
    model_name: str,
    epochs: int,
    seed: int,
) -> dict:
    replay_dataset = ShardReplayDataset(cache_manifest_path)
    split = SplitManifest.load(split_path)
    cache_fingerprint = replay_dataset.manifest.fingerprint()
    if split.cache_fingerprint != cache_fingerprint:
        raise ValueError("Saved split does not belong to the supplied cache")
    datasets = {
        name: PlayerSequenceDataset(
            replay_dataset, split.splits[name]["dataset_indices"]
        )
        for name in ("train", "validation", "test")
    }
    train_targets = np.asarray(
        [datasets["train"][index][1] for index in range(len(datasets["train"]))]
    )
    if np.unique(train_targets).size != 2:
        raise ValueError("Task 1B training data requires both outcome classes")
    trained = train_sequence_dataset(
        datasets["train"],
        name=model_name,
        task="classification",
        epochs=epochs,
        seed=seed,
    )
    results = {}
    for name in ("validation", "test"):
        probabilities, targets = predict_sequence_dataset(trained, datasets[name])
        targets = targets.astype(int)
        results[name] = {
            "metrics": classification_metrics(targets, probabilities),
            "calibration": reliability_bins(targets, probabilities),
            "samples": len(targets),
            "class_counts": {
                str(value): int((targets == value).sum())
                for value in np.unique(targets)
            },
        }
    return {
        "task": "1B",
        "model": model_name,
        "split": str(split_path),
        "results": results,
        "provenance": {
            "software": software_provenance(),
            "cache_fingerprint": cache_fingerprint,
            "split_fingerprint": split.fingerprint(),
        },
    }


def run_sequence_benchmark(
    json_path: Path | None,
    model_name: str,
    epochs: int = 10,
    max_replays: int = 0,
    seed: int = 42,
    offsets_path: Path | None = None,
    input_format: InputFormat = "auto",
    source_indices_path: Path | None = None,
    cache_manifest_path: Path | None = None,
    split_path: Path | None = None,
) -> dict:
    if cache_manifest_path is not None:
        if split_path is None:
            raise ValueError("Cached Task 1B requires a saved split manifest")
        if max_replays > 0:
            raise ValueError("--max-replays is not supported with saved cache splits")
        return _run_cached_sequence_benchmark(
            cache_manifest_path, split_path, model_name, epochs, seed
        )
    elif json_path is not None:
        samples = load_player_samples(
            json_path,
            max_replays=max_replays,
            offsets_path=offsets_path,
            input_format=input_format,
            source_indices_path=source_indices_path,
        )
    else:
        raise ValueError("Task 1B requires a JSON source or cache manifest")
    replay_ids = [sample.replay_id for sample in samples]
    split = (
        indices_from_split_manifest(replay_ids, split_path)
        if split_path is not None
        else grouped_split(replay_ids, seed=seed)
    )
    assert_disjoint_groups(split, replay_ids)
    sequences = [sample.sequence for sample in samples]
    targets = np.asarray([sample.outcome for sample in samples])
    if np.unique(targets[split.train]).size != 2:
        raise ValueError("Task 1B training data requires both outcome classes")
    trained = train_sequence_model(
        [sequences[index] for index in split.train],
        targets[split.train],
        name=model_name,
        task="classification",
        epochs=epochs,
        seed=seed,
    )
    result = {}

    for split_name, indices in (
        ("validation", split.validation),
        ("test", split.test),
    ):
        probabilities = predict_sequence_model(
            trained, [sequences[index] for index in indices]
        )
        result[split_name] = {
            "metrics": classification_metrics(targets[indices], probabilities),
            "calibration": reliability_bins(targets[indices], probabilities),
            "samples": int(len(indices)),
            "class_counts": {
                str(value): int((targets[indices] == value).sum())
                for value in np.unique(targets[indices])
            },
        }

    return {
        "task": "1B",
        "model": model_name,
        "split": str(split_path) if split_path is not None else "replay-grouped",
        "results": result,
        "provenance": {
            "software": software_provenance(),
            "cache_fingerprint": None,
        },
    }
