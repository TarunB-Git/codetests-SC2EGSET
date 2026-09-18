from pathlib import Path

import numpy as np

from latent_trainer.benchmarks.common.data import load_player_samples
from latent_trainer.benchmarks.common.metrics import (
    classification_metrics,
    reliability_bins,
)
from latent_trainer.benchmarks.common.models import (
    predict_sequence_model,
    train_sequence_model,
)
from latent_trainer.benchmarks.common.splits import (
    assert_disjoint_groups,
    grouped_split,
)


def run_sequence_benchmark(
    json_path: Path,
    model_name: str,
    epochs: int = 10,
    max_replays: int = 0,
    seed: int = 42,
) -> dict:
    samples = load_player_samples(json_path, max_replays=max_replays)
    replay_ids = [sample.replay_id for sample in samples]
    split = grouped_split(replay_ids, seed=seed)
    assert_disjoint_groups(split, replay_ids)
    sequences = [sample.sequence for sample in samples]
    targets = np.asarray([sample.outcome for sample in samples])
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
        }

    return {
        "task": "1B",
        "model": model_name,
        "split": "replay-grouped",
        "results": result,
    }
