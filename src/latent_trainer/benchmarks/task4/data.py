from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import Dataset

from latent_trainer.benchmarks.cache.dataset import ShardReplayDataset
from latent_trainer.benchmarks.data.schema import (
    canonical_feature_names,
    validate_encoder_fields,
)
from latent_trainer.benchmarks.splits.manifest import SplitManifest


@dataclass(frozen=True)
class StreamingNormalization:
    count: int
    mean: torch.Tensor
    std: torch.Tensor
    minimum: torch.Tensor
    maximum: torch.Tensor

    def save(self, path: Path) -> None:
        value = {
            "count": self.count,
            "feature_names": list(canonical_feature_names()),
            "mean": self.mean.tolist(),
            "std": self.std.tolist(),
            "minimum": self.minimum.tolist(),
            "maximum": self.maximum.tolist(),
        }
        path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")

    def fingerprint(self) -> str:
        value = {
            "count": self.count,
            "feature_names": list(canonical_feature_names()),
            "mean": self.mean.tolist(),
            "std": self.std.tolist(),
            "minimum": self.minimum.tolist(),
            "maximum": self.maximum.tolist(),
        }
        payload = json.dumps(value, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(payload.encode()).hexdigest()

    @classmethod
    def load(cls, path: Path) -> StreamingNormalization:
        value = json.loads(path.read_text(encoding="utf-8"))
        if tuple(value["feature_names"]) != canonical_feature_names():
            raise ValueError("Normalization feature order is incompatible")
        return cls(
            count=int(value["count"]),
            mean=torch.tensor(value["mean"], dtype=torch.float32),
            std=torch.tensor(value["std"], dtype=torch.float32),
            minimum=torch.tensor(value["minimum"], dtype=torch.float32),
            maximum=torch.tensor(value["maximum"], dtype=torch.float32),
        )


def fit_streaming_normalization(
    dataset: ShardReplayDataset, train_indices: list[int]
) -> StreamingNormalization:
    validate_encoder_fields(canonical_feature_names())
    count = 0
    mean = torch.zeros(len(canonical_feature_names()), dtype=torch.float64)
    second = torch.zeros_like(mean)
    minimum = torch.full_like(mean, float("inf"))
    maximum = torch.full_like(mean, float("-inf"))
    for index in train_indices:
        for row in dataset[index]["average"].double():
            count += 1
            delta = row - mean
            mean += delta / count
            second += delta * (row - mean)
            minimum = torch.minimum(minimum, row)
            maximum = torch.maximum(maximum, row)
    if count < 2:
        raise ValueError("At least two training player rows are required")
    std = torch.sqrt(second / (count - 1)).clamp_min(1e-6)
    return StreamingNormalization(
        count=count,
        mean=mean.float(),
        std=std.float(),
        minimum=minimum.float(),
        maximum=maximum.float(),
    )


class GuidedVAEShardAdapter(Dataset[tuple[torch.Tensor, torch.Tensor, dict[str, Any]]]):
    def __init__(
        self,
        dataset: ShardReplayDataset,
        indices: list[int],
        normalization: StreamingNormalization,
        guide_label: str = "outcome",
    ) -> None:
        if guide_label not in {"outcome", "mmr", "skill_class"}:
            raise ValueError("Unsupported guide label")
        self.dataset = dataset
        self.indices = indices
        self.normalization = normalization
        self.guide_label = guide_label

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(
        self, index: int
    ) -> tuple[torch.Tensor, torch.Tensor, dict[str, Any]]:
        sample = self.dataset[self.indices[index]]
        encoder_input = (
            sample["average"] - self.normalization.mean
        ) / self.normalization.std
        if self.guide_label == "outcome":
            label = sample["outcomes"][0].float()
        elif self.guide_label == "mmr":
            if not bool(sample["mmr_valid"][0]):
                raise ValueError("MMR guide requested for an invalid rating")
            label = sample["raw_mmr"][0].float()
        else:
            raise ValueError("Skill-class guides require training-derived boundaries")
        metadata = {
            "replay_id": sample["replay_id"],
            "source_index": sample["source_index"],
            "toon_ids": sample["toon_ids"],
            "raw_mmr": sample["raw_mmr"],
            "mmr_valid": sample["mmr_valid"],
            "outcomes": sample["outcomes"],
            "game_version": sample["game_version"],
        }
        return encoder_input, label, metadata


def load_compatible_split(
    dataset: ShardReplayDataset, split_path: Path
) -> SplitManifest:
    split = SplitManifest.load(split_path)
    if split.cache_fingerprint != dataset.manifest.fingerprint():
        raise ValueError("Split does not belong to this cache")
    return split
