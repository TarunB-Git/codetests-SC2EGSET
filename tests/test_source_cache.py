import json
from copy import deepcopy
from pathlib import Path

import pytest
import torch
from click.testing import CliRunner

import latent_trainer.benchmarks.cache.extract as extract_module
from latent_trainer.benchmarks.cache.collate import collate_replays, packed_sequences
from latent_trainer.benchmarks.cache.dataset import ShardReplayDataset
from latent_trainer.benchmarks.cache.extract import extract_cache, validate_cache
from latent_trainer.benchmarks.cli import main
from latent_trainer.benchmarks.common.data import (
    load_player_samples,
    player_samples_from_replay,
)
from latent_trainer.benchmarks.data.schema import (
    FeatureRole,
    canonical_feature_names,
    fields_for_role,
    metadata_blind_feature_names,
)
from latent_trainer.benchmarks.data.source import (
    ReplayReadError,
    ReplaySource,
    build_source_index,
    detect_input_format,
)
from latent_trainer.benchmarks.splits import generate_split_manifest
from latent_trainer.benchmarks.task0 import audit_source
from latent_trainer.benchmarks.task1.sequence import run_sequence_benchmark
from latent_trainer.benchmarks.task1.static import run_static_benchmark
from latent_trainer.benchmarks.task4.counterfactual import (
    constrain_to_observed_range,
)
from latent_trainer.benchmarks.task4.data import (
    GuidedVAEShardAdapter,
    fit_streaming_normalization,
)
from latent_trainer.models.guided_vae import suGuidedVAE
from latent_trainer.models.losses import loss_supervised

SYNTHETIC_PATH = (
    Path(__file__).parents[1]
    / "data/synthetic/sc2egset_synthetic_merged/sc2egset_synthetic_merged.json"
)


def _records(count: int = 6, connected: bool = False) -> list[dict]:
    originals = json.loads(SYNTHETIC_PATH.read_text(encoding="utf-8"))
    records = []
    for index in range(count):
        record = deepcopy(originals[index % len(originals)])
        record["messageEvents"] = []
        record["gameEvents"] = []
        record["trackerEvents"] = [
            event
            for event in record["trackerEvents"]
            if event.get("evtTypeName") == "PlayerStats"
        ]
        record["details"]["timeUTC"] = f"2020-01-{index + 1:02d}T00:00:00Z"
        record["metadata"]["gameVersion"] = f"5.0.{index % 2}"
        record["ToonPlayerDescMap"] = {
            (f"toon-{index + slot}" if connected else f"toon-{index}-{slot}"): value
            for slot, value in enumerate(record["ToonPlayerDescMap"].values())
        }
        records.append(record)
    return records


def _write_source(
    tmp_path: Path, records: list[dict], input_format: str = "jsonl"
) -> tuple[Path, Path, Path]:
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()
    suffix = "jsonl" if input_format == "jsonl" else "json"
    source = raw_dir / f"replays.{suffix}"
    if input_format == "jsonl":
        source.write_text(
            "".join(json.dumps(record) + "\n" for record in records),
            encoding="utf-8",
        )
    else:
        source.write_text(
            "[\n" + ",\n".join(json.dumps(record) for record in records) + "\n]\n",
            encoding="utf-8",
        )
    indices = raw_dir / "indices.json"
    indices.write_text(
        json.dumps([100 + index * 7 for index in range(len(records))]),
        encoding="utf-8",
    )
    offsets = tmp_path / "indices" / "replays.offsets.json"
    return source, indices, offsets


@pytest.mark.parametrize("input_format", ["jsonl", "single-json"])
def test_source_detection_indexing_and_mapping(tmp_path: Path, input_format: str):
    records = _records(3)
    source_path, indices_path, offsets_path = _write_source(
        tmp_path, records, input_format
    )
    assert detect_input_format(source_path) == input_format
    identity = build_source_index(
        source_path,
        offsets_path,
        source_indices_path=indices_path,
    )
    assert identity.record_count == 3
    assert offsets_path.parent != source_path.parent
    source_path.chmod(0o444)
    with ReplaySource(
        source_path, offsets_path, source_indices_path=indices_path
    ) as source:
        assert source[0].source_index == 100
        assert source[-1].source_index == 114
        assert source[0].byte_offset < source[-1].byte_offset
        assert [record.source_index for record in source] == [100, 107, 114]
        assert source[0].replay_id == "record-100"


def test_shared_loader_reads_first_and_last_jsonl_records(tmp_path: Path):
    source_path, _, _ = _write_source(tmp_path, _records(3))

    samples = load_player_samples(source_path)

    assert len(samples) == 6
    assert list(dict.fromkeys(sample.replay_id for sample in samples)) == [
        "record-0",
        "record-1",
        "record-2",
    ]


def test_cli_help_and_fatal_errors_are_nonzero():
    runner = CliRunner()
    assert runner.invoke(main, ["source-index", "--help"]).exit_code == 0
    result = runner.invoke(main, ["task1-sequence", "--model", "gru"])
    assert result.exit_code != 0
    assert "exactly one" in result.output


def test_source_rejects_malformed_stale_and_mismatched_indices(tmp_path: Path):
    source_path, indices_path, offsets_path = _write_source(tmp_path, _records(2))
    build_source_index(source_path, offsets_path, source_indices_path=indices_path)
    mismatched = tmp_path / "bad-indices.json"
    mismatched.write_text("[1]", encoding="utf-8")
    with pytest.raises(ValueError, match="count"):
        ReplaySource(source_path, offsets_path, source_indices_path=mismatched)
    source_path.write_text(
        source_path.read_text(encoding="utf-8") + "{}\n", encoding="utf-8"
    )
    with pytest.raises(ValueError, match="stale"):
        ReplaySource(source_path, offsets_path)

    malformed_root = tmp_path / "malformed"
    malformed_root.mkdir()
    malformed = malformed_root / "source.jsonl"
    malformed.write_text(
        json.dumps(_records(1)[0]) + "\n{bad json}\n", encoding="utf-8"
    )
    malformed_offsets = tmp_path / "malformed-index" / "offsets.json"
    build_source_index(malformed, malformed_offsets)
    with ReplaySource(malformed, malformed_offsets) as source:
        assert source[0].local_index == 0
        with pytest.raises(ReplayReadError, match="Expecting") as error:
            source[1]
        assert error.value.code == "malformed_json"


def test_cache_shards_loading_audit_and_equivalence(tmp_path: Path):
    source_path, indices_path, offsets_path = _write_source(tmp_path, _records(6))
    build_source_index(source_path, offsets_path, source_indices_path=indices_path)
    cache_dir = tmp_path / "cache"
    with ReplaySource(
        source_path, offsets_path, source_indices_path=indices_path
    ) as source:
        plan = extract_cache(source, cache_dir, dry_run=True, shard_size=2)
        assert plan["shard_count"] == 3
        extraction = extract_cache(
            source,
            cache_dir,
            source_indices_path=indices_path,
            shard_size=2,
            workers=1,
        )
        audit = audit_source(source)
        direct = player_samples_from_replay(source[0].replay)
    manifest_path = Path(extraction["manifest"])
    validated = validate_cache(manifest_path)
    dataset = ShardReplayDataset(manifest_path, lru_shards=1)
    assert validated["shards"] == 3
    assert validated["valid_replays"] == 6
    assert audit["replay_count"] == 6
    assert len(dataset) == 6
    assert torch.equal(dataset[0]["average"][0], direct[0].average)
    assert dataset[0]["source_index"] == 100
    shard = torch.load(
        cache_dir / dataset.manifest.shards[0].path,
        map_location="cpu",
        weights_only=True,
    )
    assert len(shard["sequence_offsets"]) == 2 * len(shard["replay_ids"]) + 1
    assert shard["sequence_replay_rows"].numel() == 2 * len(shard["replay_ids"])
    batch = collate_replays([dataset[0], dataset[1]])
    assert batch["sequence"].shape[0] == 4
    assert batch["padding_mask"].shape[:2] == batch["sequence"].shape[:2]
    assert packed_sequences(batch).batch_sizes[0] == 4
    prefix = ShardReplayDataset(manifest_path, prefix_loop=1)
    assert all(int(loops.max()) <= 1 for loops in prefix[0]["sequence_loops"])


def test_cache_accepts_an_empty_rejected_shard(tmp_path: Path):
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()
    source_path = raw_dir / "rejected.jsonl"
    source_path.write_text("{}\n", encoding="utf-8")
    offsets_path = tmp_path / "indices" / "offsets.json"
    build_source_index(source_path, offsets_path)
    with ReplaySource(source_path, offsets_path) as source:
        extraction = extract_cache(source, tmp_path / "cache", shard_size=1)
    result = validate_cache(Path(extraction["manifest"]))
    assert result["valid_replays"] == 0
    assert result["skipped_replays"] == 1


def test_cache_resume_rejects_incompatible_settings_and_corruption(tmp_path: Path):
    source_path, indices_path, offsets_path = _write_source(tmp_path, _records(4))
    build_source_index(source_path, offsets_path, source_indices_path=indices_path)
    cache_dir = tmp_path / "cache"
    with ReplaySource(source_path, offsets_path) as source:
        extraction = extract_cache(source, cache_dir, shard_size=2)
        resumed = extract_cache(source, cache_dir, shard_size=2, resume=True)
        assert resumed["manifest"] == extraction["manifest"]
        with pytest.raises(ValueError, match="settings"):
            extract_cache(source, cache_dir, shard_size=3, resume=True)
    dataset = ShardReplayDataset(Path(extraction["manifest"]))
    shard_path = cache_dir / dataset.manifest.shards[0].path
    with shard_path.open("ab") as handle:
        handle.write(b"corruption")
    with pytest.raises(ValueError, match="fingerprint"):
        validate_cache(Path(extraction["manifest"]))


def test_interrupted_extraction_resumes_equivalently(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    source_path, indices_path, offsets_path = _write_source(tmp_path, _records(6))
    build_source_index(source_path, offsets_path, source_indices_path=indices_path)
    interrupted_dir = tmp_path / "interrupted"
    complete_dir = tmp_path / "complete"
    original = extract_module._process_range
    calls = 0

    def interrupt(source: ReplaySource, start: int, stop: int):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise RuntimeError("simulated interruption")
        return original(source, start, stop)

    with ReplaySource(source_path, offsets_path) as source:
        monkeypatch.setattr(extract_module, "_process_range", interrupt)
        with pytest.raises(RuntimeError, match="simulated"):
            extract_cache(source, interrupted_dir, shard_size=2)
        monkeypatch.setattr(extract_module, "_process_range", original)
        interrupted = extract_cache(source, interrupted_dir, shard_size=2, resume=True)
        complete = extract_cache(source, complete_dir, shard_size=2, workers=2)
    interrupted_dataset = ShardReplayDataset(Path(interrupted["manifest"]))
    complete_dataset = ShardReplayDataset(Path(complete["manifest"]))
    assert len(interrupted_dataset) == len(complete_dataset)
    for index in range(len(complete_dataset)):
        assert torch.equal(
            interrupted_dataset[index]["average"],
            complete_dataset[index]["average"],
        )


def test_splits_feature_contract_and_guided_vae_adapter(tmp_path: Path):
    source_path, indices_path, offsets_path = _write_source(tmp_path, _records(12))
    build_source_index(source_path, offsets_path, source_indices_path=indices_path)
    cache_dir = tmp_path / "cache"
    with ReplaySource(source_path, offsets_path) as source:
        extraction = extract_cache(source, cache_dir, shard_size=3)
    manifest_path = Path(extraction["manifest"])
    split_path = tmp_path / "splits" / "replay.json"
    split = generate_split_manifest(
        manifest_path,
        split_path,
        strategy="replay-grouped",
        seed=7,
    )
    assert split.assertions["replay_disjoint"]
    all_indices = [
        index
        for name in ("train", "validation", "test")
        for index in split.splits[name]["dataset_indices"]
    ]
    assert sorted(all_indices) == list(range(12))
    strict_path = tmp_path / "splits" / "strict.json"
    strict = generate_split_manifest(
        manifest_path,
        strict_path,
        strategy="strict-player-disjoint",
        seed=7,
    )
    assert strict.assertions["all_players_disjoint"]
    task1 = run_static_benchmark(
        manifest_path,
        models=("logistic",),
        split_path=split_path,
    )
    assert task1["models"]["logistic"]["test"]["samples"] > 0
    task1_sequence = run_sequence_benchmark(
        None,
        model_name="gru",
        epochs=1,
        cache_manifest_path=manifest_path,
        split_path=split_path,
    )
    assert task1_sequence["results"]["test"]["samples"] > 0

    assert len(canonical_feature_names()) == 39
    assert "mmr" not in metadata_blind_feature_names()
    assert "player_toon" not in metadata_blind_feature_names()
    assert "mmr" in fields_for_role(FeatureRole.TARGET)
    dataset = ShardReplayDataset(manifest_path)
    train_indices = split.splits["train"]["dataset_indices"]
    normalization = fit_streaming_normalization(dataset, train_indices)
    adapter = GuidedVAEShardAdapter(dataset, train_indices, normalization)
    encoder_input, guide, metadata = adapter[0]
    assert encoder_input.shape == (2, 39)
    assert guide.ndim == 0
    assert "raw_mmr" in metadata
    assert encoder_input.numel() == 2 * len(canonical_feature_names())
    model = suGuidedVAE([16], latent_dim=6, input_dim=39, supervised_dim=2)
    reconstruction, mean, logvar, prediction = model(encoder_input.unsqueeze(0))
    loss, reconstruction_loss = loss_supervised(
        reconstruction, encoder_input.unsqueeze(0), mean, logvar
    )
    assert reconstruction.shape == (1, 2, 39)
    assert prediction.shape == (1, 1)
    assert torch.isfinite(loss)
    assert torch.isfinite(reconstruction_loss)
    constrained, invalid = constrain_to_observed_range(
        torch.tensor([[-1.0, float("nan"), 3.0]]),
        torch.tensor([0.0, 0.0, 0.0]),
        torch.tensor([2.0, 2.0, 2.0]),
    )
    assert invalid.tolist() == [[True, True, True]]
    assert constrained.tolist() == [[0.0, 0.0, 2.0]]


def test_strict_split_reports_graph_infeasibility_and_edge_drops(tmp_path: Path):
    source_path, indices_path, offsets_path = _write_source(
        tmp_path, _records(8, connected=True)
    )
    build_source_index(source_path, offsets_path, source_indices_path=indices_path)
    cache_dir = tmp_path / "cache"
    with ReplaySource(source_path, offsets_path) as source:
        extraction = extract_cache(source, cache_dir, shard_size=4)
    manifest_path = Path(extraction["manifest"])
    with pytest.raises(ValueError, match="infeasible"):
        generate_split_manifest(
            manifest_path,
            tmp_path / "infeasible.json",
            strategy="strict-player-disjoint",
        )
    split = generate_split_manifest(
        manifest_path,
        tmp_path / "dropped.json",
        strategy="strict-player-disjoint",
        allow_cross_edge_drops=True,
    )
    assert split.assertions["all_players_disjoint"]
    assert split.component_sizes == [8]
    assert split.data_loss["dropped_cross_split_edges"] > 0
