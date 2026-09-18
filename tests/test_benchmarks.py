from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
import torch
from sc2_datasets.torch.datasets.sc2_dataset_single_json import (
    SC2DatasetSingleJSON,
)

from latent_trainer.benchmarks.common.data import load_player_samples
from latent_trainer.benchmarks.common.metrics import (
    classification_metrics,
    regression_metrics,
)
from latent_trainer.benchmarks.common.models import (
    predict_sequence_model,
    train_sequence_model,
)
from latent_trainer.benchmarks.common.splits import (
    assert_disjoint_groups,
    grouped_split,
    mmr_shift_split,
    player_held_out_split,
    temporal_split,
)
from latent_trainer.benchmarks.task1.static import (
    run_static_benchmark,
    tabular_view,
)
from latent_trainer.benchmarks.task2.prefix import (
    build_prefix_data,
    turning_points,
)
from latent_trainer.benchmarks.task3.skill import (
    build_skill_data,
    quantile_classes,
)
from latent_trainer.benchmarks.transforms.aligned_economy import (
    historical_economy_average_players_vs_outcomes,
)

JSON_PATH = (
    Path(__file__).parents[1]
    / "data/synthetic/sc2egset_synthetic_merged/sc2egset_synthetic_merged.json"
)


def test_player_samples_and_prefixes_are_aligned_and_causal():
    samples = load_player_samples(JSON_PATH)
    assert len(samples) == 4

    for sample in samples:
        assert sample.average.shape == (39,)
        assert sample.sequence.shape[1] == 39
        assert sample.sequence_loops.max().item() <= sample.duration_loops
        assert sample.outcome in (0, 1)

    one_minute = build_prefix_data(samples, 1)
    ten_minutes = build_prefix_data(samples, 10)

    assert one_minute.gameplay_state.shape == (4, 78)
    assert ten_minutes.gameplay_state.shape == (4, 78)
    assert all(
        len(early) < len(late)
        for early, late in zip(one_minute.sequences, ten_minutes.sequences, strict=True)
    )
    assert one_minute.cutoff_loop == 1344


def test_historical_transform_applies_duration_filter():
    dataset = SC2DatasetSingleJSON.from_json_path(JSON_PATH)
    replay = dataset[0]
    features, labels = historical_economy_average_players_vs_outcomes(replay)
    short_replay = replace(
        replay, header=replace(replay.header, elapsedGameLoops=12000)
    )
    assert features.shape == (2, 39)
    assert labels.shape == (2,)
    assert historical_economy_average_players_vs_outcomes(short_replay) is None


def test_split_strategies_and_group_leakage_checks():
    groups = np.repeat([f"player-{index}" for index in range(10)], 2)
    split = grouped_split(groups, seed=7)
    assert_disjoint_groups(split, groups)
    assert len(split.train) + len(split.validation) + len(split.test) == len(groups)

    player_split = player_held_out_split(
        groups,
        [f"replay-{index // 2}" for index in range(len(groups))],
        seed=7,
    )
    assert_disjoint_groups(player_split, groups)
    assert_disjoint_groups(
        player_split, [f"replay-{index // 2}" for index in range(len(groups))]
    )

    temporal = temporal_split([f"2020-01-{index:02d}" for index in range(1, 21)])
    assert temporal.train.max() < temporal.validation.min() < temporal.test.min()

    shifted = mmr_shift_split(np.arange(1000, 3000, 100))
    assert shifted.train.max() < shifted.validation.min() < shifted.test.min()


def test_tabular_views_and_static_protocols(tmp_path: Path):
    generator = torch.Generator().manual_seed(3)
    features = torch.randn(30, 2, 39, generator=generator)
    labels = torch.tensor([[0, 1]] * 30)
    cache = {
        "train_features": features[:20],
        "train_labels": labels[:20],
        "val_features": features[20:25],
        "val_labels": labels[20:25],
        "test_features": features[25:],
        "test_labels": labels[25:],
        "transform": "averaged_economy",
    }
    cache_path = tmp_path / "cache.pt"
    torch.save(cache, cache_path)
    flattened, flattened_labels = tabular_view(
        features.numpy(), labels.numpy(), "one-player"
    )

    assert flattened.shape == (60, 39)
    assert flattened_labels.shape == (60,)

    result = run_static_benchmark(
        cache_path,
        models=("logistic",),
        view="one-player",
        protocol="corrected",
    )
    assert set(result["models"]["logistic"]) == {"validation", "test"}

    cache["transform"] = "historical_averaged_economy"
    torch.save(cache, cache_path)
    historical = run_static_benchmark(
        cache_path,
        models=("logistic",),
        view="one-player",
        protocol="historical",
    )
    assert historical["models"]["logistic"]["samples"] == 60


@pytest.mark.parametrize("name", ["gru", "transformer"])
def test_sequence_classification_models(name: str):
    sequences = [torch.randn(3 + index % 3, 6) for index in range(8)]
    targets = np.asarray([0, 1] * 4)
    trained = train_sequence_model(
        sequences,
        targets,
        name=name,
        task="classification",
        epochs=1,
        batch_size=4,
    )
    probabilities = predict_sequence_model(trained, sequences)
    assert probabilities.shape == (8,)
    assert np.all((probabilities >= 0) & (probabilities <= 1))


def test_sequence_regression_and_multiclass_models():
    sequences = [torch.randn(3 + index % 3, 6) for index in range(8)]
    regression = train_sequence_model(
        sequences,
        np.linspace(1000, 4000, 8),
        name="gru",
        task="regression",
        epochs=1,
        batch_size=4,
    )
    regression_predictions = predict_sequence_model(regression, sequences)
    assert regression_predictions.shape == (8,)

    multiclass = train_sequence_model(
        sequences,
        np.asarray([0, 1, 2, 3, 0, 1, 2, 3]),
        name="transformer",
        task="multiclass",
        num_classes=4,
        epochs=1,
        batch_size=4,
    )
    class_predictions = predict_sequence_model(multiclass, sequences)
    assert class_predictions.shape == (8,)
    assert set(class_predictions).issubset({0, 1, 2, 3})


def test_skill_data_is_metadata_blind_and_quantiles_fit_on_train():
    source = load_player_samples(JSON_PATH)[0]
    samples = [
        replace(
            source,
            player_toon=f"toon-{index}",
            replay_id=f"replay-{index}",
            mmr=float(1000 + index * 100),
            race=("Prot", "Terr", "Zerg")[index % 3],
        )
        for index in range(12)
    ]
    economy = build_skill_data(samples)
    race_only = build_skill_data(samples, include_race=True, mask_economy=True)
    classes, boundaries = quantile_classes(economy.mmr[:8], economy.mmr)

    assert economy.features.shape == (12, 39)
    assert race_only.features.shape == (12, 3)
    assert race_only.sequences[0].shape[1] == 3
    assert classes.shape == (12,)
    assert boundaries.shape == (3,)


def test_metrics_and_turning_points():
    classification = classification_metrics(
        np.asarray([0, 1, 0, 1]), np.asarray([0.1, 0.9, 0.2, 0.8])
    )
    regression = regression_metrics(
        np.asarray([1.0, 2.0, 3.0]), np.asarray([1.1, 1.9, 3.2])
    )
    points = turning_points({"sample": {1.0: 0.5, 2.0: 0.7, 3.0: 0.6}})

    assert classification["accuracy"] == 1.0
    assert regression["spearman"] == pytest.approx(1.0)
    assert points["sample"][1]["delta_probability"] == pytest.approx(0.2)
