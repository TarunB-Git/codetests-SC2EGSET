from __future__ import annotations

import hashlib
import json
import os
from dataclasses import asdict, dataclass, field
from pathlib import Path

from latent_trainer.benchmarks.cache.schema import CACHE_SCHEMA_VERSION


@dataclass(frozen=True)
class ShardEntry:
    path: str
    start_index: int
    stop_index: int
    replay_count: int
    player_row_count: int
    rejection_count: int
    byte_size: int
    sha256: str


@dataclass
class CacheManifest:
    cache_schema_version: int
    extractor_version: int
    feature_names: list[str]
    source_identity: dict[str, object]
    source_format: str
    source_path: str
    offsets_path: str
    source_indices_path: str | None
    source_indices_fingerprint: str | None
    sc2_datasets_version: str
    sc2_datasets_commit: str | None
    package_version: str
    package_commit: str | None
    seed: int
    shard_size: int
    requested_start_index: int
    requested_stop_index: int
    completed_ranges: list[list[int]] = field(default_factory=list)
    shards: list[ShardEntry] = field(default_factory=list)
    valid_replays: int = 0
    skipped_replays: int = 0
    error_replays: int = 0
    rejection_counts: dict[str, int] = field(default_factory=dict)
    creation_state: str = "in_progress"
    completion_state: str = "incomplete"

    @classmethod
    def load(cls, path: Path) -> CacheManifest:
        with path.open("r", encoding="utf-8") as handle:
            value = json.load(handle)
        value["shards"] = [ShardEntry(**item) for item in value["shards"]]
        return cls(**value)

    def save_atomic(self, path: Path) -> None:
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

    def validate_compatibility(
        self, source_fingerprint: str, shard_size: int, seed: int
    ) -> None:
        if self.cache_schema_version != CACHE_SCHEMA_VERSION:
            raise ValueError("Cache schema version differs")
        if self.source_identity["fingerprint"] != source_fingerprint:
            raise ValueError("Cache source fingerprint differs")
        if self.shard_size != shard_size or self.seed != seed:
            raise ValueError("Resume settings differ from the existing cache")
