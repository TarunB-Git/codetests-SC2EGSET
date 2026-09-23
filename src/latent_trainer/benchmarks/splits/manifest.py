from __future__ import annotations

import hashlib
import json
import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

from latent_trainer.benchmarks.cache.dataset import ShardReplayDataset
from latent_trainer.benchmarks.common.splits import (
    SplitIndices,
    grouped_split,
    mmr_shift_split,
    random_split,
    temporal_split,
    version_split,
)

SPLIT_SCHEMA_VERSION = 1


@dataclass
class SplitManifest:
    schema_version: int
    strategy: str
    strategy_version: int
    seed: int
    cache_fingerprint: str
    source_fingerprint: str
    splits: dict[str, dict[str, Any]]
    sizes: dict[str, int]
    exclusion_counts: dict[str, int]
    assertions: dict[str, bool]
    parameters: dict[str, Any]
    component_sizes: list[int]
    data_loss: dict[str, int]

    @classmethod
    def load(cls, path: Path) -> SplitManifest:
        with path.open("r", encoding="utf-8") as handle:
            return cls(**json.load(handle))

    def save_atomic(self, path: Path) -> None:
        if path.exists():
            raise FileExistsError(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
        with temporary.open("w", encoding="utf-8") as handle:
            json.dump(asdict(self), handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)

    def fingerprint(self) -> str:
        payload = json.dumps(asdict(self), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(payload.encode()).hexdigest()


class _UnionFind:
    def __init__(self) -> None:
        self.parent: dict[str, str] = {}

    def find(self, value: str) -> str:
        self.parent.setdefault(value, value)
        while self.parent[value] != value:
            self.parent[value] = self.parent[self.parent[value]]
            value = self.parent[value]
        return value

    def union(self, left: str, right: str) -> None:
        left_root = self.find(left)
        right_root = self.find(right)
        if left_root != right_root:
            self.parent[right_root] = left_root


def _component_sizes(players: list[tuple[str, str]]) -> list[int]:
    union = _UnionFind()
    for left, right in players:
        union.union(left, right)
    counts: dict[str, int] = {}
    for left, _ in players:
        root = union.find(left)
        counts[root] = counts.get(root, 0) + 1
    return sorted(counts.values(), reverse=True)


def _strict_components(
    players: list[tuple[str, str]],
    validation_fraction: float,
    test_fraction: float,
    seed: int,
) -> tuple[SplitIndices, list[int]]:
    union = _UnionFind()
    for left, right in players:
        union.union(left, right)
    groups: dict[str, list[int]] = {}
    for index, pair in enumerate(players):
        root = union.find(pair[0])
        groups.setdefault(root, []).append(index)
    component_sizes = sorted((len(value) for value in groups.values()), reverse=True)
    train_fraction = 1.0 - validation_fraction - test_fraction
    if len(groups) < 3 or (
        component_sizes and component_sizes[0] > round(len(players) * train_fraction)
    ):
        raise ValueError(
            "Strict player-disjoint target ratios are infeasible for graph components"
        )
    rng = np.random.default_rng(seed)
    components = list(groups.values())
    rng.shuffle(components)
    components.sort(key=len, reverse=True)
    targets = np.array(
        [train_fraction, validation_fraction, test_fraction], dtype=float
    ) * len(players)
    assigned: list[list[int]] = [[], [], []]
    for component in components:
        deficits = targets - np.array([len(value) for value in assigned])
        slot = int(np.argmax(deficits))
        assigned[slot].extend(component)
    return (
        SplitIndices(
            train=np.asarray(sorted(assigned[0]), dtype=int),
            validation=np.asarray(sorted(assigned[1]), dtype=int),
            test=np.asarray(sorted(assigned[2]), dtype=int),
        ),
        component_sizes,
    )


def _strict_with_edge_drops(
    players: list[tuple[str, str]],
    validation_fraction: float,
    test_fraction: float,
    seed: int,
) -> tuple[SplitIndices, int]:
    unique = sorted({player for pair in players for player in pair})
    player_split = random_split(len(unique), validation_fraction, test_fraction, seed)
    labels: dict[str, int] = {}
    for slot, indices in enumerate(
        [player_split.train, player_split.validation, player_split.test]
    ):
        labels.update({unique[index]: slot for index in indices})
    assigned: list[list[int]] = [[], [], []]
    dropped = 0
    for index, pair in enumerate(players):
        if labels[pair[0]] != labels[pair[1]]:
            dropped += 1
        else:
            assigned[labels[pair[0]]].append(index)
    return (
        SplitIndices(*(np.asarray(value, dtype=int) for value in assigned)),
        dropped,
    )


def _focal_player_split(
    players: list[tuple[str, str]],
    validation_fraction: float,
    test_fraction: float,
    seed: int,
) -> SplitIndices:
    focal = [pair[0] for pair in players]
    return grouped_split(focal, validation_fraction, test_fraction, seed)


def _assert_replay_disjoint(split: SplitIndices) -> bool:
    values = [set(split.train), set(split.validation), set(split.test)]
    return not (values[0] & values[1] or values[0] & values[2] or values[1] & values[2])


def _assert_player_disjoint(
    split: SplitIndices, players: list[tuple[str, str]], focal_only: bool
) -> bool:
    values = []
    for indices in [split.train, split.validation, split.test]:
        if focal_only:
            values.append({players[index][0] for index in indices})
        else:
            values.append({player for index in indices for player in players[index]})
    return not (values[0] & values[1] or values[0] & values[2] or values[1] & values[2])


def _serialize_split(
    indices: np.ndarray,
    samples: list[dict[str, Any]],
) -> dict[str, Any]:
    selected = [samples[int(index)] for index in indices]
    return {
        "dataset_indices": [int(value) for value in indices],
        "local_record_indices": [item["local_record_index"] for item in selected],
        "source_indices": [item["source_index"] for item in selected],
        "replay_ids": [item["replay_id"] for item in selected],
        "player_identifiers": [list(item["toon_ids"]) for item in selected],
    }


def generate_split_manifest(
    cache_manifest_path: Path,
    output_path: Path,
    strategy: str = "replay-grouped",
    seed: int = 42,
    validation_fraction: float = 0.1,
    test_fraction: float = 0.2,
    test_version: str | None = None,
    validation_version: str | None = None,
    allow_cross_edge_drops: bool = False,
) -> SplitManifest:
    dataset = ShardReplayDataset(cache_manifest_path)
    samples = list(dataset.iter_metadata())
    players = [item["toon_ids"] for item in samples]
    component_sizes: list[int] = []
    dropped = 0
    parameters: dict[str, Any] = {
        "validation_fraction": validation_fraction,
        "test_fraction": test_fraction,
    }
    if strategy in {"random", "replay-grouped"}:
        split = random_split(len(samples), validation_fraction, test_fraction, seed)
    elif strategy == "focal-player-held-out":
        split = _focal_player_split(players, validation_fraction, test_fraction, seed)
    elif strategy == "strict-player-disjoint":
        if allow_cross_edge_drops:
            component_sizes = _component_sizes(players)
            split, dropped = _strict_with_edge_drops(
                players, validation_fraction, test_fraction, seed
            )
        else:
            split, component_sizes = _strict_components(
                players, validation_fraction, test_fraction, seed
            )
    elif strategy == "temporal":
        split = temporal_split(
            [item["timestamp"] for item in samples],
            validation_fraction,
            test_fraction,
        )
    elif strategy == "game-version":
        if test_version is None:
            raise ValueError("Game-version splitting requires a test version")
        split = version_split(
            [item["game_version"] for item in samples],
            test_version,
            validation_version,
        )
        parameters.update(
            {
                "test_version": test_version,
                "validation_version": validation_version,
            }
        )
    elif strategy == "mmr-ood":
        ratings = [
            float(np.mean(np.asarray(item["raw_mmr"])[item["mmr_valid"]]))
            if any(item["mmr_valid"])
            else float("nan")
            for item in samples
        ]
        valid_indices = np.flatnonzero(np.isfinite(ratings))
        local = mmr_shift_split(np.asarray(ratings)[valid_indices])
        split = SplitIndices(
            train=valid_indices[local.train],
            validation=valid_indices[local.validation],
            test=valid_indices[local.test],
        )
        dropped = len(samples) - len(valid_indices)
    else:
        raise ValueError(f"Unsupported split strategy: {strategy}")
    replay_disjoint = _assert_replay_disjoint(split)
    focal_only = strategy == "focal-player-held-out"
    player_disjoint = _assert_player_disjoint(split, players, focal_only)
    if not replay_disjoint:
        raise ValueError("Replay groups overlap")
    if (
        strategy in {"focal-player-held-out", "strict-player-disjoint"}
        and not player_disjoint
    ):
        raise ValueError("Requested player-disjoint split overlaps")
    manifest = SplitManifest(
        schema_version=SPLIT_SCHEMA_VERSION,
        strategy=strategy,
        strategy_version=1,
        seed=seed,
        cache_fingerprint=dataset.manifest.fingerprint(),
        source_fingerprint=str(dataset.manifest.source_identity["fingerprint"]),
        splits={
            "train": _serialize_split(split.train, samples),
            "validation": _serialize_split(split.validation, samples),
            "test": _serialize_split(split.test, samples),
        },
        sizes={
            "train": len(split.train),
            "validation": len(split.validation),
            "test": len(split.test),
        },
        exclusion_counts={
            "unassigned_replays": len(samples)
            - sum(map(len, [split.train, split.validation, split.test]))
        },
        assertions={
            "replay_disjoint": replay_disjoint,
            "focal_players_disjoint": player_disjoint if focal_only else False,
            "all_players_disjoint": player_disjoint
            if strategy == "strict-player-disjoint"
            else False,
            "temporal_ordered": strategy != "temporal"
            or (
                max(samples[index]["timestamp"] for index in split.train)
                <= min(samples[index]["timestamp"] for index in split.validation)
                and max(samples[index]["timestamp"] for index in split.validation)
                <= min(samples[index]["timestamp"] for index in split.test)
            ),
        },
        parameters=parameters,
        component_sizes=component_sizes,
        data_loss={"dropped_cross_split_edges": dropped},
    )
    manifest.save_atomic(output_path)
    return manifest
