from pathlib import Path

import torch

DATASETS = [
    Path("data/cached_dataset_rich_sc2egset.pt"),
    Path("data/cached_dataset_rich_sc2ggset_2178.pt"),
    Path("data/cached_dataset_averaged_economy_test.pt"),
]

def describe(value):
    if isinstance(value, torch.Tensor):
        return {
            "type": "Tensor",
            "shape": tuple(value.shape),
            "dtype": str(value.dtype),
        }

    if isinstance(value, (list, tuple)):
        return {
            "type": type(value).__name__,
            "length": len(value),
            "first_type": type(value[0]).__name__ if value else None,
        }

    if isinstance(value, dict):
        return {
            "type": "dict",
            "keys": list(value.keys()),
        }

    return {
        "type": type(value).__name__,
        "value": str(value)[:100],
    }


for path in DATASETS:
    print("\n" + "=" * 70)
    print(path)
    print("=" * 70)

    if not path.exists():
        print("NOT FOUND")
        continue

    data = torch.load(path, map_location="cpu", weights_only=False)

    print("Top-level type:", type(data).__name__)

    if isinstance(data, dict):
        for key, value in data.items():
            print(f"{key}: {describe(value)}")
    else:
        print(describe(data))
