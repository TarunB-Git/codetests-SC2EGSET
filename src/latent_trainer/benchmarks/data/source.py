from __future__ import annotations

import hashlib
import json
import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import BinaryIO, Iterator, Literal

from sc2_datasets.replay_data.sc2_replay_data import SC2ReplayData
from sc2_datasets.utils.json_utils import get_object_at_index

from latent_trainer.benchmarks.data.jsonl import scan_line_offsets

InputFormat = Literal["auto", "jsonl", "single-json"]
SOURCE_SCHEMA_VERSION = 1
OFFSET_SCHEMA_VERSION = 1


class ReplayReadError(RuntimeError):
    def __init__(
        self,
        code: str,
        local_index: int,
        source_index: int,
        message: str,
    ) -> None:
        super().__init__(message)
        self.code = code
        self.local_index = local_index
        self.source_index = source_index


@dataclass(frozen=True)
class SourceIdentity:
    resolved_path: str
    byte_size: int
    modification_time_ns: int
    record_count: int
    input_format: str
    schema_version: int
    fingerprint: str
    checksum: str | None = None


@dataclass(frozen=True)
class ReplayRecord:
    local_index: int
    byte_offset: int
    source_index: int
    replay_id: str
    replay: SC2ReplayData


def detect_input_format(path: Path) -> Literal["jsonl", "single-json"]:
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(4096)
            if not chunk:
                raise ValueError("Replay source is empty")
            stripped = chunk.lstrip()
            if not stripped:
                continue
            if stripped.startswith(b"["):
                return "single-json"
            if stripped.startswith(b"{"):
                return "jsonl"
            raise ValueError("Replay source must begin with '[' or '{'")


def _sampled_fingerprint(path: Path) -> str:
    size = path.stat().st_size
    digest = hashlib.blake2b(digest_size=32)
    digest.update(str(size).encode())
    positions = sorted({0, max(0, size // 2 - 32768), max(0, size - 65536)})
    with path.open("rb") as handle:
        for position in positions:
            handle.seek(position)
            digest.update(position.to_bytes(8, "little"))
            digest.update(handle.read(65536))
    return digest.hexdigest()


def _full_checksum(path: Path, expected: str) -> str:
    algorithm, separator, value = expected.partition(":")
    if not separator:
        algorithm, value = "sha256", algorithm
    if algorithm not in hashlib.algorithms_available:
        raise ValueError(f"Unsupported checksum algorithm: {algorithm}")
    digest = hashlib.new(algorithm)
    with path.open("rb") as handle:
        while chunk := handle.read(8 * 1024 * 1024):
            digest.update(chunk)
    actual = f"{algorithm}:{digest.hexdigest()}"
    if digest.hexdigest().lower() != value.lower():
        raise ValueError("Source checksum does not match")
    return actual


def _indices_fingerprint(path: Path | None) -> str | None:
    if path is None:
        return None
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _load_source_indices(path: Path | None, count: int) -> list[int]:
    if path is None:
        return list(range(count))
    with path.open("r", encoding="utf-8") as handle:
        values = json.load(handle)
    if not isinstance(values, list) or any(
        isinstance(value, bool) or not isinstance(value, int) for value in values
    ):
        raise ValueError("Source-index file must contain a JSON list of integers")
    if len(values) != count:
        raise ValueError(
            f"Source-index count {len(values)} does not match record count {count}"
        )
    return values


def _atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, sort_keys=True, separators=(",", ":"))
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def build_source_index(
    json_path: Path,
    offsets_path: Path,
    input_format: InputFormat = "auto",
    source_indices_path: Path | None = None,
    checksum: str | None = None,
    overwrite: bool = False,
) -> SourceIdentity:
    source_path = json_path.resolve()
    if offsets_path.resolve().parent == source_path.parent:
        raise ValueError("Offsets must be stored outside the raw-data directory")
    detected = detect_input_format(source_path)
    selected = detected if input_format == "auto" else input_format
    if selected != detected:
        raise ValueError(
            f"Requested format {selected} does not match detected {detected}"
        )
    if offsets_path.exists() and not overwrite:
        return ReplaySource(
            source_path,
            offsets_path,
            input_format=selected,
            source_indices_path=source_indices_path,
            checksum=checksum,
        ).identity
    offsets = scan_line_offsets(source_path, selected)
    source_indices = _load_source_indices(source_indices_path, len(offsets))
    stat = source_path.stat()
    identity = SourceIdentity(
        resolved_path=str(source_path),
        byte_size=stat.st_size,
        modification_time_ns=stat.st_mtime_ns,
        record_count=len(offsets),
        input_format=selected,
        schema_version=SOURCE_SCHEMA_VERSION,
        fingerprint=_sampled_fingerprint(source_path),
        checksum=_full_checksum(source_path, checksum) if checksum else None,
    )
    payload = {
        "schema_version": OFFSET_SCHEMA_VERSION,
        "source": asdict(identity),
        "source_indices_path": str(source_indices_path.resolve())
        if source_indices_path
        else None,
        "source_indices_fingerprint": _indices_fingerprint(source_indices_path),
        "offsets": offsets,
        "source_indices": source_indices,
    }
    _atomic_json(offsets_path, payload)
    return identity


class ReplaySource:
    def __init__(
        self,
        json_path: Path,
        offsets_path: Path,
        input_format: InputFormat = "auto",
        source_indices_path: Path | None = None,
        checksum: str | None = None,
    ) -> None:
        self._handle: BinaryIO | None = None
        self._pid: int | None = None
        self.json_path = json_path.resolve()
        self.offsets_path = offsets_path.resolve()
        with self.offsets_path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
        if payload.get("schema_version") != OFFSET_SCHEMA_VERSION:
            raise ValueError("Unsupported offset schema version")
        self.offsets = [int(value) for value in payload["offsets"]]
        self.source_indices = [int(value) for value in payload["source_indices"]]
        self.identity = SourceIdentity(**payload["source"])
        detected = detect_input_format(self.json_path)
        selected = detected if input_format == "auto" else input_format
        stat = self.json_path.stat()
        current_fingerprint = _sampled_fingerprint(self.json_path)
        if (
            self.identity.resolved_path != str(self.json_path)
            or self.identity.byte_size != stat.st_size
            or self.identity.modification_time_ns != stat.st_mtime_ns
            or self.identity.record_count != len(self.offsets)
            or self.identity.input_format != selected
            or self.identity.fingerprint != current_fingerprint
        ):
            raise ValueError("Offsets are stale or belong to another replay source")
        if len(self.source_indices) != len(self.offsets):
            raise ValueError("Offset and original-source-index counts differ")
        if source_indices_path is not None:
            supplied = _load_source_indices(source_indices_path, len(self.offsets))
            if supplied != self.source_indices:
                raise ValueError("Original-source-index mapping does not match offsets")
            if payload.get("source_indices_fingerprint") != _indices_fingerprint(
                source_indices_path
            ):
                raise ValueError("Original-source-index fingerprint does not match")
        if checksum is not None:
            actual = _full_checksum(self.json_path, checksum)
            if self.identity.checksum not in {None, actual}:
                raise ValueError("Stored checksum does not match source")

    def __len__(self) -> int:
        return len(self.offsets)

    def _file_handle(self) -> BinaryIO:
        pid = os.getpid()
        if self._handle is None or self._handle.closed or self._pid != pid:
            self.close()
            self._handle = self.json_path.open("rb")
            self._pid = pid
        return self._handle

    def close(self) -> None:
        if self._handle is not None and not self._handle.closed:
            self._handle.close()
        self._handle = None
        self._pid = None

    def __enter__(self) -> ReplaySource:
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    def __del__(self) -> None:
        self.close()

    def __getitem__(self, index: int) -> ReplayRecord:
        if index < 0:
            index += len(self)
        if index < 0 or index >= len(self):
            raise IndexError(index)
        source_index = self.source_indices[index]
        try:
            loaded = get_object_at_index(self._file_handle(), self.offsets, index)
        except (UnicodeDecodeError, json.JSONDecodeError) as error:
            raise ReplayReadError(
                "malformed_json", index, source_index, str(error)
            ) from error
        if not isinstance(loaded, dict):
            raise ReplayReadError(
                "schema_not_object", index, source_index, "Replay JSON is not an object"
            )
        required = {"header", "initData", "details", "metadata"}
        missing = sorted(required - loaded.keys())
        if missing:
            raise ReplayReadError(
                "schema_missing_fields",
                index,
                source_index,
                f"Missing fields: {missing}",
            )
        replay_id = f"record-{source_index}"
        try:
            replay = SC2ReplayData.from_dict(loaded, replay_filepath=replay_id)
        except (KeyError, TypeError, ValueError) as error:
            raise ReplayReadError(
                "parser_failure", index, source_index, str(error)
            ) from error
        return ReplayRecord(
            local_index=index,
            byte_offset=self.offsets[index],
            source_index=source_index,
            replay_id=replay_id,
            replay=replay,
        )

    def __iter__(self) -> Iterator[ReplayRecord]:
        for index in range(len(self)):
            yield self[index]
