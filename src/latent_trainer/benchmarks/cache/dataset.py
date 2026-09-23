from __future__ import annotations

import bisect
from collections import OrderedDict
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import Dataset

from latent_trainer.benchmarks.cache.manifest import CacheManifest
from latent_trainer.benchmarks.cache.schema import CACHE_SCHEMA_VERSION
from latent_trainer.benchmarks.data.schema import canonical_feature_names


class ShardReplayDataset(Dataset[dict[str, Any]]):
    def __init__(
        self,
        manifest_path: Path,
        lru_shards: int = 2,
        prefix_loop: int | None = None,
    ) -> None:
        if lru_shards <= 0:
            raise ValueError("LRU shard count must be positive")
        self.manifest_path = manifest_path.resolve()
        self.manifest = CacheManifest.load(self.manifest_path)
        if self.manifest.cache_schema_version != CACHE_SCHEMA_VERSION:
            raise ValueError("Unsupported cache schema")
        if tuple(self.manifest.feature_names) != canonical_feature_names():
            raise ValueError("Cache feature order differs from canonical Stats order")
        self.lru_shards = lru_shards
        self.prefix_loop = prefix_loop
        self._cache: OrderedDict[int, dict[str, Any]] = OrderedDict()
        self._ends: list[int] = []
        total = 0
        for shard in self.manifest.shards:
            total += shard.replay_count
            self._ends.append(total)

    def __len__(self) -> int:
        return self._ends[-1] if self._ends else 0

    def _load_shard(self, shard_index: int) -> dict[str, Any]:
        if shard_index in self._cache:
            value = self._cache.pop(shard_index)
            self._cache[shard_index] = value
            return value
        entry = self.manifest.shards[shard_index]
        value = torch.load(
            self.manifest_path.parent / entry.path,
            map_location="cpu",
            weights_only=True,
        )
        self._cache[shard_index] = value
        while len(self._cache) > self.lru_shards:
            self._cache.popitem(last=False)
        return value

    def _location(self, index: int) -> tuple[int, int]:
        if index < 0:
            index += len(self)
        if index < 0 or index >= len(self):
            raise IndexError(index)
        shard_index = bisect.bisect_right(self._ends, index)
        start = 0 if shard_index == 0 else self._ends[shard_index - 1]
        return shard_index, index - start

    def __getitem__(self, index: int) -> dict[str, Any]:
        shard_index, row = self._location(index)
        shard = self._load_shard(shard_index)
        sequences = []
        loops = []
        for slot in range(2):
            player_row = row * 2 + slot
            start = int(shard["sequence_offsets"][player_row])
            stop = int(shard["sequence_offsets"][player_row + 1])
            player_values = shard["sequence_values"][start:stop]
            player_loops = shard["sequence_loops"][start:stop]
            if self.prefix_loop is not None:
                keep = player_loops <= self.prefix_loop
                player_values = player_values[keep]
                player_loops = player_loops[keep]
            sequences.append(player_values)
            loops.append(player_loops)
        return {
            "average": shard["averaged_gameplay"][row],
            "sequences": sequences,
            "sequence_loops": loops,
            "outcomes": shard["outcomes"][row],
            "raw_mmr": shard["raw_mmr"][row],
            "mmr_valid": shard["mmr_valid"][row],
            "local_record_index": int(shard["local_record_indices"][row]),
            "source_index": int(shard["source_indices"][row]),
            "replay_id": shard["replay_ids"][row],
            "player_ids": tuple(shard["player_ids"][row]),
            "toon_ids": tuple(shard["toon_ids"][row]),
            "races": tuple(shard["races"][row]),
            "regions": tuple(shard["regions"][row]),
            "timestamp": shard["timestamps"][row],
            "game_version": shard["game_versions"][row],
            "map_name": shard["map_names"][row],
            "duration_loops": int(shard["duration_loops"][row]),
            "parser_status": shard["parser_status"][row],
        }

    def shard_batches(self, batch_size: int) -> Iterator[list[int]]:
        if batch_size <= 0:
            raise ValueError("Batch size must be positive")
        start = 0
        for end in self._ends:
            for batch_start in range(start, end, batch_size):
                yield list(range(batch_start, min(batch_start + batch_size, end)))
            start = end

    def iter_metadata(self) -> Iterator[dict[str, Any]]:
        for shard_index, entry in enumerate(self.manifest.shards):
            shard = self._load_shard(shard_index)
            for row in range(entry.replay_count):
                yield {
                    "local_record_index": int(shard["local_record_indices"][row]),
                    "source_index": int(shard["source_indices"][row]),
                    "replay_id": shard["replay_ids"][row],
                    "toon_ids": tuple(shard["toon_ids"][row]),
                    "timestamp": shard["timestamps"][row],
                    "game_version": shard["game_versions"][row],
                    "raw_mmr": shard["raw_mmr"][row].tolist(),
                    "mmr_valid": shard["mmr_valid"][row].tolist(),
                }

    def tabular_batches(
        self,
        batch_size: int,
        player_view: bool = True,
    ) -> Iterator[tuple[np.ndarray, np.ndarray]]:
        for indices in self.shard_batches(batch_size):
            samples = [self[index] for index in indices]
            features = torch.stack([item["average"] for item in samples])
            targets = torch.stack([item["outcomes"] for item in samples])
            if player_view:
                yield (
                    features.reshape(-1, features.shape[-1]).numpy(),
                    targets.reshape(-1).numpy(),
                )
            else:
                yield features.numpy(), targets.numpy()


class PlayerSequenceDataset(Dataset[tuple[torch.Tensor, float]]):
    def __init__(
        self,
        replay_dataset: ShardReplayDataset,
        replay_indices: list[int],
        target: str = "outcome",
    ) -> None:
        if target not in {"outcome", "mmr"}:
            raise ValueError("Sequence target must be outcome or MMR")
        self.replay_dataset = replay_dataset
        self.rows = [
            (replay_index, slot)
            for replay_index in replay_indices
            for slot in range(2)
            if target != "mmr" or bool(replay_dataset[replay_index]["mmr_valid"][slot])
        ]
        self.target = target

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, float]:
        replay_index, slot = self.rows[index]
        sample = self.replay_dataset[replay_index]
        target = (
            float(sample["outcomes"][slot])
            if self.target == "outcome"
            else float(sample["raw_mmr"][slot])
        )
        return sample["sequences"][slot], target
