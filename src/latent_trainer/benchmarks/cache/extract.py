from __future__ import annotations

import hashlib
import importlib.metadata
import multiprocessing
import os
import subprocess
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict
from pathlib import Path
from typing import Any, cast

import torch

from latent_trainer.benchmarks.cache.manifest import CacheManifest, ShardEntry
from latent_trainer.benchmarks.cache.schema import (
    CACHE_SCHEMA_VERSION,
    EXTRACTOR_VERSION,
    RejectionCode,
)
from latent_trainer.benchmarks.common.data import player_samples_from_replay
from latent_trainer.benchmarks.data.schema import canonical_feature_names
from latent_trainer.benchmarks.data.source import (
    InputFormat,
    ReplayReadError,
    ReplaySource,
)


def _git_commit(path: Path) -> str | None:
    try:
        return subprocess.run(
            ["git", "-C", str(path), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _rejection_for_replay(replay: Any) -> RejectionCode:
    if replay.trackerEventsErr:
        return RejectionCode.TRACKER_ERROR
    if len(replay.toonPlayerDescMap) != 2:
        return RejectionCode.NOT_TWO_PLAYERS
    results = {
        description.toon_player_info.result for description in replay.toonPlayerDescMap
    }
    if results != {"Win", "Loss"}:
        return RejectionCode.INVALID_OUTCOME
    return RejectionCode.MISSING_PLAYER_STATS


def _empty_shard(start: int, stop: int) -> dict[str, Any]:
    feature_count = len(canonical_feature_names())
    return {
        "schema_version": CACHE_SCHEMA_VERSION,
        "source_range": [start, stop],
        "local_record_indices": torch.empty(0, dtype=torch.long),
        "source_indices": torch.empty(0, dtype=torch.long),
        "replay_ids": [],
        "player_ids": [],
        "toon_ids": [],
        "averaged_gameplay": torch.empty((0, 2, feature_count)),
        "outcomes": torch.empty((0, 2), dtype=torch.long),
        "raw_mmr": torch.empty((0, 2)),
        "mmr_valid": torch.empty((0, 2), dtype=torch.bool),
        "races": [],
        "regions": [],
        "timestamps": [],
        "game_versions": [],
        "map_names": [],
        "duration_loops": torch.empty(0, dtype=torch.long),
        "sequence_values": torch.empty((0, feature_count)),
        "sequence_loops": torch.empty(0, dtype=torch.long),
        "sequence_offsets": torch.tensor([0], dtype=torch.long),
        "sequence_replay_rows": torch.empty(0, dtype=torch.long),
        "sequence_player_slots": torch.empty(0, dtype=torch.long),
        "parser_status": [],
        "rejections": [],
    }


def _process_range(source: ReplaySource, start: int, stop: int) -> dict[str, Any]:
    shard = _empty_shard(start, stop)
    local_indices: list[int] = []
    source_indices: list[int] = []
    averaged: list[torch.Tensor] = []
    outcomes: list[list[int]] = []
    raw_mmr: list[list[float]] = []
    mmr_valid: list[list[bool]] = []
    durations: list[int] = []
    sequence_values: list[torch.Tensor] = []
    sequence_loops: list[torch.Tensor] = []
    sequence_offsets = [0]
    sequence_replay_rows: list[int] = []
    sequence_player_slots: list[int] = []

    for local_index in range(start, stop):
        try:
            record = source[local_index]
        except ReplayReadError as error:
            code = (
                RejectionCode.MALFORMED_JSON
                if error.code == "malformed_json"
                else RejectionCode.PARSER_FAILURE
                if error.code == "parser_failure"
                else RejectionCode.SCHEMA_FAILURE
            )
            shard["rejections"].append(
                {
                    "local_record_index": local_index,
                    "source_index": error.source_index,
                    "code": str(code),
                }
            )
            continue
        try:
            samples = player_samples_from_replay(record.replay)
        except (KeyError, TypeError, ValueError, RuntimeError):
            samples = []
            rejection = RejectionCode.TRANSFORM_FAILURE
        else:
            rejection = _rejection_for_replay(record.replay)
        if len(samples) != 2:
            shard["rejections"].append(
                {
                    "local_record_index": record.local_index,
                    "source_index": record.source_index,
                    "code": str(rejection),
                }
            )
            continue
        samples = sorted(samples, key=lambda item: int(item.player_id))
        descriptions = {
            str(item.toon_player_info.playerID): item
            for item in record.replay.toonPlayerDescMap
        }
        replay_row = len(local_indices)
        local_indices.append(record.local_index)
        source_indices.append(record.source_index)
        shard["replay_ids"].append(record.replay_id)
        shard["player_ids"].append([item.player_id for item in samples])
        shard["toon_ids"].append([item.player_toon for item in samples])
        averaged.append(torch.stack([item.average for item in samples]))
        outcomes.append([item.outcome for item in samples])
        ratings = [
            float(descriptions[item.player_id].toon_player_info.MMR) for item in samples
        ]
        raw_mmr.append(ratings)
        mmr_valid.append([value > 0 for value in ratings])
        shard["races"].append([item.race for item in samples])
        shard["regions"].append(
            [descriptions[item.player_id].toon_player_info.region for item in samples]
        )
        shard["timestamps"].append(samples[0].timestamp)
        shard["game_versions"].append(samples[0].game_version)
        shard["map_names"].append(samples[0].map_name)
        durations.append(samples[0].duration_loops)
        shard["parser_status"].append(
            {
                "game_events_error": bool(record.replay.gameEventsErr),
                "message_events_error": bool(record.replay.messageEventsErr),
                "tracker_events_error": bool(record.replay.trackerEventsErr),
            }
        )
        for slot, sample in enumerate(samples):
            sequence_values.append(sample.sequence)
            sequence_loops.append(sample.sequence_loops)
            sequence_offsets.append(sequence_offsets[-1] + len(sample.sequence))
            sequence_replay_rows.append(replay_row)
            sequence_player_slots.append(slot)

    shard["local_record_indices"] = torch.tensor(local_indices, dtype=torch.long)
    shard["source_indices"] = torch.tensor(source_indices, dtype=torch.long)
    if averaged:
        shard["averaged_gameplay"] = torch.stack(averaged)
        shard["outcomes"] = torch.tensor(outcomes, dtype=torch.long)
        shard["raw_mmr"] = torch.tensor(raw_mmr, dtype=torch.float32)
        shard["mmr_valid"] = torch.tensor(mmr_valid, dtype=torch.bool)
        shard["duration_loops"] = torch.tensor(durations, dtype=torch.long)
    if sequence_values:
        shard["sequence_values"] = torch.cat(sequence_values)
        shard["sequence_loops"] = torch.cat(sequence_loops)
        shard["sequence_offsets"] = torch.tensor(sequence_offsets, dtype=torch.long)
        shard["sequence_replay_rows"] = torch.tensor(
            sequence_replay_rows, dtype=torch.long
        )
        shard["sequence_player_slots"] = torch.tensor(
            sequence_player_slots, dtype=torch.long
        )
    return shard


def _worker_process_range(
    arguments: tuple[str, str, str, int, int, str],
) -> tuple[ShardEntry, dict[str, int]]:
    json_path, offsets_path, input_format, start, stop, output_path = arguments
    with ReplaySource(
        Path(json_path),
        Path(offsets_path),
        input_format=cast(InputFormat, input_format),
    ) as source:
        return _process_and_write(source, start, stop, Path(output_path))


def _process_and_write(
    source: ReplaySource, start: int, stop: int, output_path: Path
) -> tuple[ShardEntry, dict[str, int]]:
    shard = _process_range(source, start, stop)
    entry = _write_shard(output_path, shard)
    counts = Counter(rejection["code"] for rejection in shard["rejections"])
    return entry, dict(counts)


def _write_shard(path: Path, shard: dict[str, Any]) -> ShardEntry:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as handle:
        torch.save(shard, handle)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)
    validate_shard(path)
    return ShardEntry(
        path=path.name,
        start_index=int(shard["source_range"][0]),
        stop_index=int(shard["source_range"][1]),
        replay_count=len(shard["replay_ids"]),
        player_row_count=int(shard["sequence_replay_rows"].numel()),
        rejection_count=len(shard["rejections"]),
        byte_size=path.stat().st_size,
        sha256=_sha256(path),
    )


def validate_shard(path: Path, expected: ShardEntry | None = None) -> None:
    shard = torch.load(path, map_location="cpu", weights_only=True)
    if shard.get("schema_version") != CACHE_SCHEMA_VERSION:
        raise ValueError(f"Invalid shard schema: {path}")
    replay_count = len(shard["replay_ids"])
    if shard["averaged_gameplay"].shape != (
        replay_count,
        2,
        len(canonical_feature_names()),
    ):
        raise ValueError(f"Invalid averaged gameplay shape: {path}")
    offsets = shard["sequence_offsets"]
    if offsets.numel() != replay_count * 2 + 1:
        raise ValueError(f"Invalid ragged offsets: {path}")
    if offsets[-1].item() != len(shard["sequence_values"]):
        raise ValueError(f"Ragged offsets do not cover sequence values: {path}")
    if expected is not None:
        if (
            path.stat().st_size != expected.byte_size
            or _sha256(path) != expected.sha256
        ):
            raise ValueError(f"Shard fingerprint mismatch: {path}")


def _initial_manifest(
    source: ReplaySource,
    output_dir: Path,
    source_indices_path: Path | None,
    shard_size: int,
    seed: int,
    start: int,
    stop: int,
) -> CacheManifest:
    source_indices_fingerprint = None
    if source_indices_path is not None:
        source_indices_fingerprint = _sha256(source_indices_path)
    root = Path(__file__).resolve().parents[4]
    return CacheManifest(
        cache_schema_version=CACHE_SCHEMA_VERSION,
        extractor_version=EXTRACTOR_VERSION,
        feature_names=list(canonical_feature_names()),
        source_identity=asdict(source.identity),
        source_format=source.identity.input_format,
        source_path=str(source.json_path),
        offsets_path=str(source.offsets_path),
        source_indices_path=str(source_indices_path.resolve())
        if source_indices_path
        else None,
        source_indices_fingerprint=source_indices_fingerprint,
        sc2_datasets_version=importlib.metadata.version("sc2-datasets"),
        sc2_datasets_commit="3f3ee3a48a48f2d47c0d62930d79ca465113816a",
        package_version=importlib.metadata.version("latent-trainer"),
        package_commit=_git_commit(root),
        seed=seed,
        shard_size=shard_size,
        requested_start_index=start,
        requested_stop_index=stop,
    )


def extract_cache(
    source: ReplaySource,
    output_dir: Path,
    source_indices_path: Path | None = None,
    start_index: int = 0,
    stop_index: int | None = None,
    max_replays: int = 0,
    shard_size: int = 500,
    workers: int = 1,
    seed: int = 42,
    resume: bool = False,
    overwrite: bool = False,
    dry_run: bool = False,
) -> dict[str, Any]:
    if shard_size <= 0 or workers <= 0:
        raise ValueError("Shard size and workers must be positive")
    stop = len(source) if stop_index is None else min(stop_index, len(source))
    if max_replays > 0:
        stop = min(stop, start_index + max_replays)
    if start_index < 0 or stop < start_index:
        raise ValueError("Invalid extraction range")
    plan = {
        "start_index": start_index,
        "stop_index": stop,
        "shard_size": shard_size,
        "shard_count": (stop - start_index + shard_size - 1) // shard_size,
        "workers": workers,
        "source_fingerprint": source.identity.fingerprint,
    }
    if dry_run:
        return plan
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = output_dir / "manifest.json"
    existing_entries: dict[tuple[int, int], ShardEntry] = {}
    orphaned_shards = list(output_dir.glob("*.shard.pt"))
    if orphaned_shards and not manifest_path.exists() and not overwrite:
        raise FileExistsError(
            "Cache shards exist without a manifest; choose --overwrite explicitly"
        )
    if overwrite and not manifest_path.exists():
        for item in orphaned_shards:
            item.unlink()
    if manifest_path.exists():
        if overwrite:
            for item in output_dir.glob("*.shard.pt"):
                item.unlink()
            manifest_path.unlink()
        elif not resume:
            raise FileExistsError("Cache exists; choose --resume or --overwrite")
    for temporary in output_dir.glob("*.shard.pt.tmp"):
        temporary.unlink()
    if manifest_path.exists():
        manifest = CacheManifest.load(manifest_path)
        manifest.validate_compatibility(source.identity.fingerprint, shard_size, seed)
        if (
            manifest.requested_start_index != start_index
            or manifest.requested_stop_index != stop
        ):
            raise ValueError("Resume range differs from the existing cache")
        existing_entries = {
            (entry.start_index, entry.stop_index): entry for entry in manifest.shards
        }
        for entry in manifest.shards:
            validate_shard(output_dir / entry.path, entry)
    else:
        manifest = _initial_manifest(
            source,
            output_dir,
            source_indices_path,
            shard_size,
            seed,
            start_index,
            stop,
        )
        manifest.save_atomic(manifest_path)
    ranges = [
        (start, min(start + shard_size, stop))
        for start in range(start_index, stop, shard_size)
        if (start, min(start + shard_size, stop)) not in existing_entries
    ]
    arguments = [
        (
            str(source.json_path),
            str(source.offsets_path),
            source.identity.input_format,
            start,
            end,
            str(output_dir / f"shard-{start:09d}-{end:09d}.shard.pt"),
        )
        for start, end in ranges
    ]
    executor = (
        ProcessPoolExecutor(
            max_workers=workers, mp_context=multiprocessing.get_context("spawn")
        )
        if workers > 1
        else None
    )
    if executor is not None:
        processed = executor.map(_worker_process_range, arguments)
    else:
        processed = (
            _process_and_write(
                source,
                start,
                end,
                output_dir / f"shard-{start:09d}-{end:09d}.shard.pt",
            )
            for start, end in ranges
        )
    counts = Counter(manifest.rejection_counts)
    try:
        for (start, end), (entry, shard_counts) in zip(ranges, processed, strict=True):
            manifest.shards.append(entry)
            manifest.shards.sort(key=lambda item: item.start_index)
            manifest.completed_ranges = [
                [item.start_index, item.stop_index] for item in manifest.shards
            ]
            counts.update(shard_counts)
            manifest.rejection_counts = dict(sorted(counts.items()))
            manifest.valid_replays += entry.replay_count
            manifest.skipped_replays += entry.rejection_count
            manifest.error_replays = sum(
                count
                for code, count in counts.items()
                if code
                in {
                    str(RejectionCode.MALFORMED_JSON),
                    str(RejectionCode.SCHEMA_FAILURE),
                    str(RejectionCode.PARSER_FAILURE),
                    str(RejectionCode.TRANSFORM_FAILURE),
                }
            )
            manifest.save_atomic(manifest_path)
    finally:
        if executor is not None:
            executor.shutdown(wait=True, cancel_futures=True)
    manifest.creation_state = "finished"
    manifest.completion_state = "complete"
    manifest.save_atomic(manifest_path)
    return {"manifest": str(manifest_path), **plan}


def validate_cache(manifest_path: Path) -> dict[str, Any]:
    manifest = CacheManifest.load(manifest_path)
    if manifest.completion_state != "complete":
        raise ValueError("Cache manifest is incomplete")
    previous_stop = manifest.requested_start_index
    for entry in manifest.shards:
        if entry.start_index != previous_stop:
            raise ValueError("Cache shard ranges are not contiguous")
        validate_shard(manifest_path.parent / entry.path, entry)
        previous_stop = entry.stop_index
    if previous_stop != manifest.requested_stop_index:
        raise ValueError("Cache does not cover its requested source range")
    return {
        "cache_schema_version": manifest.cache_schema_version,
        "cache_fingerprint": manifest.fingerprint(),
        "shards": len(manifest.shards),
        "valid_replays": manifest.valid_replays,
        "skipped_replays": manifest.skipped_replays,
        "error_replays": manifest.error_replays,
        "status": "valid",
    }
