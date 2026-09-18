from pathlib import Path

import torch
from sc2_datasets.torch.datasets.sc2_dataset_single_json import (
    SC2DatasetSingleJSON,
)
from sc2_datasets.transforms.utils import RESULT_DICT, average_player_stats

from latent_trainer.benchmarks.transforms.aligned_economy import (
    economy_average_players_vs_outcomes,
)
from latent_trainer.features.preprocess_dataset import (
    TransformEnumFunction,
    process_replay,
    stack_labels,
)

JSON_PATH = (
    Path(__file__).parents[1]
    / "data/synthetic/sc2egset_synthetic_merged/sc2egset_synthetic_merged.json"
)


def test_aligned_economy_on_every_synthetic_replay():
    dataset = SC2DatasetSingleJSON.from_json_path(JSON_PATH)

    assert len(dataset) == 2

    for replay_index in range(len(dataset)):
        replay = dataset[replay_index]
        features, targets = economy_average_players_vs_outcomes(replay)
        averaged = average_player_stats(replay)
        infos = sorted(
            (desc.toon_player_info for desc in replay.toonPlayerDescMap),
            key=lambda info: int(info.playerID),
        )

        assert features.shape == (2, 39)
        assert features.dtype == torch.float32
        assert targets.shape == (2,)
        assert targets.dtype == torch.long

        for row, info in enumerate(infos):
            player_id = str(info.playerID)
            expected_features = torch.tensor(averaged[player_id], dtype=torch.float32)
            assert torch.equal(features[row], expected_features)
            assert targets[row].item() == RESULT_DICT[info.result]


def test_averaged_economy_preprocessing_uses_aligned_transform():
    choice = TransformEnumFunction(["rich", "averaged_economy"])
    transform = choice.convert("averaged_economy", None, None)
    replay = SC2DatasetSingleJSON.from_json_path(JSON_PATH)[0]
    features, targets = process_replay(replay, transform)

    assert transform is economy_average_players_vs_outcomes
    assert features.shape == (2, 39)
    assert targets.shape == (2,)
    assert stack_labels([targets, targets]).shape == (2, 2)
    assert stack_labels([0, 1]).shape == (2,)
