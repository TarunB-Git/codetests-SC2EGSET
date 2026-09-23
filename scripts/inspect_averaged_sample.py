from pathlib import Path

from sc2_datasets.lightning.sc2_egset_datamodule import (
    SC2EGSetDataModuleSingleJSON,
)
from sc2_datasets.replay_parser.tracker_events.events.player_stats.stats import Stats
from sc2_datasets.transforms.pytorch.economy_vs_outcome import (
    economy_average_vs_outcome,
)

JSON_PATH = Path(
    "data/synthetic/"
    "sc2egset_synthetic_merged/"
    "sc2egset_synthetic_merged.json"
)

#not hard-code 39 names in benchmark code.
feature_names = list(Stats.__dataclass_fields__.keys())

print(f"Number of PlayerStats features: {len(feature_names)}")
print("\nFeature order:")
for i, name in enumerate(feature_names):
    print(f"{i:2d}: {name}")


dm = SC2EGSetDataModuleSingleJSON(
    json_path=JSON_PATH,
    download=False,
)

dm.prepare_data()
dm.setup("fit")

# The synthetic fixture contains only two games
replay = dm.train_dataset[0]

features, label = economy_average_vs_outcome(replay)

print("\n" + "=" * 70)
print("TRANSFORM OUTPUT")
print("=" * 70)

print("Feature shape:", tuple(features.shape))
print("Label:", label)

print("\nReplay player information:")
for position, toon_desc in enumerate(replay.toonPlayerDescMap):
    info = toon_desc.toon_player_info

    print(
        f"position={position}, "
        f"playerID={info.playerID}, "
        f"race={info.race}, "
        f"result={info.result}, "
        f"MMR={info.MMR}"
    )

print("\nFirst five averaged features for each tensor row:")

for player_index in range(features.shape[0]):
    print(f"\nTensor row {player_index}:")

    for feature_index in range(5):
        print(
            f"  {feature_names[feature_index]}: "
            f"{features[player_index, feature_index].item():.3f}"
        )
