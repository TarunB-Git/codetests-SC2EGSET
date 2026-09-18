from pathlib import Path

import torch
from sc2_datasets.torch.datasets.sc2_dataset_single_json import (
    SC2DatasetSingleJSON,
)
from sc2_datasets.transforms.utils import RESULT_DICT, average_player_stats

from latent_trainer.benchmarks.transforms.aligned_economy import (
    economy_average_players_vs_outcomes,
)

JSON_PATH = Path(
    "data/synthetic/"
    "sc2egset_synthetic_merged/"
    "sc2egset_synthetic_merged.json"
)


def main():
    dataset = SC2DatasetSingleJSON.from_json_path(JSON_PATH)
    assert len(dataset) == 2
    print(f"Replays found: {len(dataset)}")

    for replay_index in range(len(dataset)):
        replay = dataset[replay_index]
        features, targets = economy_average_players_vs_outcomes(replay)
        averaged = average_player_stats(replay)

        print("\n" + "=" * 70)
        print(f"REPLAY {replay_index}")
        print("=" * 70)

        print("features shape:", tuple(features.shape))
        print("targets:", targets.tolist())

        infos = sorted(
            (desc.toon_player_info for desc in replay.toonPlayerDescMap),
            key=lambda info: int(info.playerID),
        )

        assert tuple(features.shape) == (2, 39)
        assert tuple(targets.shape) == (2,)

        for row, info in enumerate(infos):
            player_id = str(info.playerID)
            expected = RESULT_DICT[info.result]

            print(
                f"row={row} "
                f"playerID={info.playerID} "
                f"race={info.race} "
                f"result={info.result} "
                f"expected_label={expected} "
                f"transform_label={targets[row].item()}"
            )

            assert targets[row].item() == expected, (
                f"Alignment failed for playerID={info.playerID}"
            )
            assert torch.equal(
                features[row], torch.tensor(averaged[player_id], dtype=torch.float32)
            ), f"Feature alignment failed for playerID={info.playerID}"

    print("\nAlignment check passed for every synthetic replay.")


if __name__ == "__main__":
    main()
