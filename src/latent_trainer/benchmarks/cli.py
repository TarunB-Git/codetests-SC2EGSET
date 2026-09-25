import json
import math
from pathlib import Path
from typing import cast

import click

from latent_trainer.benchmarks.cache import extract_cache, validate_cache
from latent_trainer.benchmarks.data import ReplaySource, build_source_index
from latent_trainer.benchmarks.data.source import InputFormat
from latent_trainer.benchmarks.splits import generate_split_manifest
from latent_trainer.benchmarks.task0 import audit_source, benchmark_loading
from latent_trainer.benchmarks.task1 import (
    run_sequence_benchmark,
    run_static_benchmark,
)
from latent_trainer.benchmarks.task2 import run_prefix_benchmark
from latent_trainer.benchmarks.task3 import run_skill_benchmark
from latent_trainer.benchmarks.task4 import (
    evaluate_guided_vae,
    generate_counterfactual,
    train_guided_vae,
)


def _json_safe(value):
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, dict):
        return {key: _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    return value


def _emit(result: dict, output: Path | None) -> None:
    rendered = json.dumps(
        _json_safe(result),
        indent=2,
        sort_keys=True,
        allow_nan=False,
    )
    if output is None:
        click.echo(rendered)
    else:
        if output.exists():
            raise click.ClickException(f"Output already exists: {output}")
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(rendered + "\n", encoding="utf-8")
        click.echo(str(output))


def _path_option(function):
    return click.option(
        "--output",
        type=click.Path(path_type=Path, dir_okay=False),
        default=None,
    )(function)


@click.group()
def main() -> None:
    pass


def _source(
    json_path: Path,
    offsets_path: Path,
    input_format: str,
    source_indices: Path | None,
) -> ReplaySource:
    return ReplaySource(
        json_path,
        offsets_path,
        input_format=cast(InputFormat, input_format),
        source_indices_path=source_indices,
    )


@main.command("source-index")
@click.option(
    "--json-path",
    required=True,
    type=click.Path(path_type=Path, exists=True, dir_okay=False),
)
@click.option(
    "--input-format",
    type=click.Choice(["auto", "jsonl", "single-json"]),
    default="auto",
)
@click.option(
    "--source-indices",
    type=click.Path(path_type=Path, exists=True, dir_okay=False),
    default=None,
)
@click.option(
    "--offsets-path",
    required=True,
    type=click.Path(path_type=Path, dir_okay=False),
)
@click.option("--checksum", default=None)
@click.option("--overwrite", is_flag=True, default=False)
def source_index(
    json_path: Path,
    input_format: str,
    source_indices: Path | None,
    offsets_path: Path,
    checksum: str | None,
    overwrite: bool,
) -> None:
    identity = build_source_index(
        json_path,
        offsets_path,
        input_format=cast(InputFormat, input_format),
        source_indices_path=source_indices,
        checksum=checksum,
        overwrite=overwrite,
    )
    _emit({"source": identity.__dict__, "offsets_path": str(offsets_path)}, None)


def _source_options(function):
    options = [
        click.option(
            "--json-path",
            required=True,
            type=click.Path(path_type=Path, exists=True, dir_okay=False),
        ),
        click.option(
            "--input-format",
            type=click.Choice(["auto", "jsonl", "single-json"]),
            default="auto",
        ),
        click.option(
            "--source-indices",
            type=click.Path(path_type=Path, exists=True, dir_okay=False),
            default=None,
        ),
        click.option(
            "--offsets-path",
            required=True,
            type=click.Path(path_type=Path, exists=True, dir_okay=False),
        ),
    ]
    for option in reversed(options):
        function = option(function)
    return function


@main.command("task0-audit")
@_source_options
@click.option(
    "--output-dir", required=True, type=click.Path(path_type=Path, file_okay=False)
)
@click.option("--invalid-mmr-sentinel", multiple=True, type=float, default=(-36400.0,))
@click.option("--nonpositive-mmr-invalid/--allow-nonpositive-mmr", default=True)
@click.option("--max-replays", type=int, default=0)
def task0_audit(
    json_path: Path,
    input_format: str,
    source_indices: Path | None,
    offsets_path: Path,
    output_dir: Path,
    invalid_mmr_sentinel: tuple[float, ...],
    nonpositive_mmr_invalid: bool,
    max_replays: int,
) -> None:
    output = output_dir / "audit.json"
    with _source(
        json_path, offsets_path, input_format, source_indices
    ) as replay_source:
        result = audit_source(
            replay_source,
            invalid_mmr_sentinels=invalid_mmr_sentinel,
            nonpositive_mmr_invalid=nonpositive_mmr_invalid,
            max_replays=max_replays,
        )
    _emit(result, output)


@main.command("loading-benchmark")
@_source_options
@click.option("--count", type=int, default=100)
@click.option("--seed", type=int, default=42)
@click.option(
    "--cache-manifest",
    type=click.Path(path_type=Path, exists=True, dir_okay=False),
    default=None,
)
@_path_option
def loading_benchmark(
    json_path: Path,
    input_format: str,
    source_indices: Path | None,
    offsets_path: Path,
    count: int,
    seed: int,
    cache_manifest: Path | None,
    output: Path | None,
) -> None:
    with _source(
        json_path, offsets_path, input_format, source_indices
    ) as replay_source:
        result = benchmark_loading(replay_source, count, seed, cache_manifest)
    _emit(result, output)


@main.command("cache-extract")
@_source_options
@click.option(
    "--output-dir", required=True, type=click.Path(path_type=Path, file_okay=False)
)
@click.option("--dry-run", is_flag=True, default=False)
@click.option("--max-replays", type=int, default=0)
@click.option("--start-index", type=int, default=0)
@click.option("--stop-index", type=int, default=None)
@click.option("--shard-size", type=int, default=500)
@click.option("--workers", type=int, default=1)
@click.option("--seed", type=int, default=42)
@click.option("--resume", is_flag=True, default=False)
@click.option("--overwrite", is_flag=True, default=False)
def cache_extract(
    json_path: Path,
    input_format: str,
    source_indices: Path | None,
    offsets_path: Path,
    output_dir: Path,
    dry_run: bool,
    max_replays: int,
    start_index: int,
    stop_index: int | None,
    shard_size: int,
    workers: int,
    seed: int,
    resume: bool,
    overwrite: bool,
) -> None:
    with _source(
        json_path, offsets_path, input_format, source_indices
    ) as replay_source:
        result = extract_cache(
            replay_source,
            output_dir,
            source_indices,
            start_index,
            stop_index,
            max_replays,
            shard_size,
            workers,
            seed,
            resume,
            overwrite,
            dry_run,
        )
    _emit(result, None)


@main.command("cache-validate")
@click.option(
    "--manifest",
    required=True,
    type=click.Path(path_type=Path, exists=True, dir_okay=False),
)
@_path_option
def cache_validate(manifest: Path, output: Path | None) -> None:
    _emit(validate_cache(manifest), output)


@main.command("split-generate")
@click.option(
    "--cache-manifest",
    required=True,
    type=click.Path(path_type=Path, exists=True, dir_okay=False),
)
@click.option(
    "--strategy",
    type=click.Choice(
        [
            "random",
            "replay-grouped",
            "focal-player-held-out",
            "strict-player-disjoint",
            "temporal",
            "game-version",
            "mmr-ood",
        ]
    ),
    default="replay-grouped",
)
@click.option("--seed", type=int, default=42)
@click.option("--validation-fraction", type=float, default=0.1)
@click.option("--test-fraction", type=float, default=0.2)
@click.option("--test-version", default=None)
@click.option("--validation-version", default=None)
@click.option("--allow-cross-edge-drops", is_flag=True, default=False)
@click.option(
    "--output", required=True, type=click.Path(path_type=Path, dir_okay=False)
)
def split_generate(
    cache_manifest: Path,
    strategy: str,
    seed: int,
    validation_fraction: float,
    test_fraction: float,
    test_version: str | None,
    validation_version: str | None,
    allow_cross_edge_drops: bool,
    output: Path,
) -> None:
    manifest = generate_split_manifest(
        cache_manifest,
        output,
        strategy,
        seed,
        validation_fraction,
        test_fraction,
        test_version,
        validation_version,
        allow_cross_edge_drops,
    )
    _emit(
        {
            "split_path": str(output),
            "split_fingerprint": manifest.fingerprint(),
            "sizes": manifest.sizes,
            "assertions": manifest.assertions,
        },
        None,
    )


@main.command("task1-static")
@click.option(
    "--cache",
    "cache_path",
    required=True,
    type=click.Path(path_type=Path, exists=True, dir_okay=False),
)
@click.option(
    "--model",
    "models",
    multiple=True,
    type=click.Choice(["logistic", "svm", "xgboost", "mlp"]),
    default=("logistic", "svm", "xgboost", "mlp"),
)
@click.option(
    "--view", type=click.Choice(["one-player", "two-player"]), default="one-player"
)
@click.option(
    "--protocol", type=click.Choice(["historical", "corrected"]), default="corrected"
)
@click.option("--calibrated/--uncalibrated", default=False)
@click.option(
    "--evaluation-cache",
    type=click.Path(path_type=Path, exists=True, dir_okay=False),
    default=None,
)
@click.option(
    "--evaluation-split",
    type=click.Path(path_type=Path, exists=True, dir_okay=False),
    default=None,
)
@click.option(
    "--split",
    "split_path",
    type=click.Path(path_type=Path, exists=True, dir_okay=False),
    default=None,
)
@click.option("--seed", type=int, default=42)
@_path_option
def task1_static(
    cache_path: Path,
    models: tuple[str, ...],
    view: str,
    protocol: str,
    calibrated: bool,
    evaluation_cache: Path | None,
    evaluation_split: Path | None,
    split_path: Path | None,
    seed: int,
    output: Path | None,
) -> None:
    _emit(
        run_static_benchmark(
            cache_path,
            models=models,
            view=view,
            protocol=protocol,
            calibrated=calibrated,
            seed=seed,
            evaluation_cache_path=evaluation_cache,
            split_path=split_path,
            evaluation_split_path=evaluation_split,
        ),
        output,
    )


@main.command("task1-sequence")
@click.option(
    "--json-path",
    required=False,
    type=click.Path(path_type=Path, exists=True, dir_okay=False),
)
@click.option(
    "--input-format",
    type=click.Choice(["auto", "jsonl", "single-json"]),
    default="auto",
)
@click.option(
    "--source-indices",
    type=click.Path(path_type=Path, exists=True, dir_okay=False),
    default=None,
)
@click.option(
    "--offsets-path",
    type=click.Path(path_type=Path, exists=True, dir_okay=False),
    default=None,
)
@click.option(
    "--cache-manifest",
    type=click.Path(path_type=Path, exists=True, dir_okay=False),
    default=None,
)
@click.option(
    "--split",
    "split_path",
    type=click.Path(path_type=Path, exists=True, dir_okay=False),
    default=None,
)
@click.option(
    "--model", "model_name", type=click.Choice(["gru", "transformer"]), required=True
)
@click.option("--epochs", type=int, default=10)
@click.option("--max-replays", type=int, default=0)
@click.option("--seed", type=int, default=42)
@_path_option
def task1_sequence(
    json_path: Path | None,
    input_format: str,
    source_indices: Path | None,
    offsets_path: Path | None,
    cache_manifest: Path | None,
    split_path: Path | None,
    model_name: str,
    epochs: int,
    max_replays: int,
    seed: int,
    output: Path | None,
) -> None:
    if (json_path is None) == (cache_manifest is None):
        raise click.UsageError("Provide exactly one of --json-path or --cache-manifest")
    _emit(
        run_sequence_benchmark(
            json_path,
            model_name=model_name,
            epochs=epochs,
            max_replays=max_replays,
            seed=seed,
            offsets_path=offsets_path,
            input_format=cast(InputFormat, input_format),
            source_indices_path=source_indices,
            cache_manifest_path=cache_manifest,
            split_path=split_path,
        ),
        output,
    )


@main.command("task2")
@click.option(
    "--json-path",
    required=False,
    type=click.Path(path_type=Path, exists=True, dir_okay=False),
)
@click.option(
    "--input-format",
    type=click.Choice(["auto", "jsonl", "single-json"]),
    default="auto",
)
@click.option(
    "--source-indices",
    type=click.Path(path_type=Path, exists=True, dir_okay=False),
    default=None,
)
@click.option(
    "--offsets-path",
    type=click.Path(path_type=Path, exists=True, dir_okay=False),
    default=None,
)
@click.option(
    "--cache-manifest",
    type=click.Path(path_type=Path, exists=True, dir_okay=False),
    default=None,
)
@click.option(
    "--split",
    "split_path",
    type=click.Path(path_type=Path, exists=True, dir_okay=False),
    default=None,
)
@click.option(
    "--model",
    "model_name",
    type=click.Choice(["logistic", "xgboost", "gru", "transformer"]),
    required=True,
)
@click.option(
    "--information",
    type=click.Choice(["prior-only", "gameplay-only", "combined"]),
    required=True,
)
@click.option(
    "--minute", "minutes", type=float, multiple=True, default=(1, 2, 3, 5, 7, 10)
)
@click.option("--epochs", type=int, default=10)
@click.option("--max-replays", type=int, default=0)
@click.option("--calibrated/--uncalibrated", default=True)
@click.option(
    "--source", type=click.Choice(["sc2ggset", "sc2egset"]), default="sc2ggset"
)
@click.option("--seed", type=int, default=42)
@_path_option
def task2(
    json_path: Path | None,
    input_format: str,
    source_indices: Path | None,
    offsets_path: Path | None,
    cache_manifest: Path | None,
    split_path: Path | None,
    model_name: str,
    information: str,
    minutes: tuple[float, ...],
    epochs: int,
    max_replays: int,
    calibrated: bool,
    source: str,
    seed: int,
    output: Path | None,
) -> None:
    if (json_path is None) == (cache_manifest is None):
        raise click.UsageError("Provide exactly one of --json-path or --cache-manifest")
    if information == "prior-only" and model_name in {"gru", "transformer"}:
        raise click.BadParameter(
            "Prior-only requires logistic or xgboost",
            param_hint="--model",
        )
    _emit(
        run_prefix_benchmark(
            json_path,
            model_name=model_name,
            information=information,
            minutes=minutes,
            epochs=epochs,
            max_replays=max_replays,
            calibrated=calibrated,
            source=source,
            seed=seed,
            offsets_path=offsets_path,
            input_format=cast(InputFormat, input_format),
            source_indices_path=source_indices,
            cache_manifest_path=cache_manifest,
            split_path=split_path,
        ),
        output,
    )


@main.command("task3")
@click.option(
    "--json-path",
    required=False,
    type=click.Path(path_type=Path, exists=True, dir_okay=False),
)
@click.option(
    "--input-format",
    type=click.Choice(["auto", "jsonl", "single-json"]),
    default="auto",
)
@click.option(
    "--source-indices",
    type=click.Path(path_type=Path, exists=True, dir_okay=False),
    default=None,
)
@click.option(
    "--offsets-path",
    type=click.Path(path_type=Path, exists=True, dir_okay=False),
    default=None,
)
@click.option(
    "--cache-manifest",
    type=click.Path(path_type=Path, exists=True, dir_okay=False),
    default=None,
)
@click.option(
    "--split",
    "split_path",
    type=click.Path(path_type=Path, exists=True, dir_okay=False),
    default=None,
)
@click.option(
    "--objective", type=click.Choice(["regression", "classification"]), required=True
)
@click.option(
    "--representation", type=click.Choice(["averaged", "sequence"]), required=True
)
@click.option(
    "--model",
    "model_name",
    type=click.Choice(["xgboost", "mlp", "gru", "transformer"]),
    required=True,
)
@click.option("--include-race/--exclude-race", default=False)
@click.option("--mask-economy", is_flag=True, default=False)
@click.option("--mask-race", is_flag=True, default=False)
@click.option("--class-count", type=int, default=4)
@click.option("--epochs", type=int, default=10)
@click.option("--max-replays", type=int, default=0)
@click.option("--seed", type=int, default=42)
@click.option(
    "--split-strategy",
    type=click.Choice(["replay-grouped", "player-held-out"]),
    default="player-held-out",
)
@_path_option
def task3(
    json_path: Path | None,
    input_format: str,
    source_indices: Path | None,
    offsets_path: Path | None,
    cache_manifest: Path | None,
    split_path: Path | None,
    objective: str,
    representation: str,
    model_name: str,
    include_race: bool,
    mask_economy: bool,
    mask_race: bool,
    class_count: int,
    epochs: int,
    max_replays: int,
    seed: int,
    split_strategy: str,
    output: Path | None,
) -> None:
    if (json_path is None) == (cache_manifest is None):
        raise click.UsageError("Provide exactly one of --json-path or --cache-manifest")
    if representation == "averaged" and model_name not in {"xgboost", "mlp"}:
        raise click.BadParameter("Averaged representation requires xgboost or mlp")
    if representation == "sequence" and model_name not in {"gru", "transformer"}:
        raise click.BadParameter("Sequence representation requires gru or transformer")
    _emit(
        run_skill_benchmark(
            json_path,
            objective=objective,
            representation=representation,
            model_name=model_name,
            include_race=include_race,
            mask_economy=mask_economy,
            mask_race=mask_race,
            class_count=class_count,
            epochs=epochs,
            max_replays=max_replays,
            seed=seed,
            split_strategy=split_strategy,
            offsets_path=offsets_path,
            input_format=cast(InputFormat, input_format),
            source_indices_path=source_indices,
            cache_manifest_path=cache_manifest,
            split_path=split_path,
        ),
        output,
    )


@main.command("task4-train")
@click.option(
    "--cache-manifest",
    required=True,
    type=click.Path(path_type=Path, exists=True, dir_okay=False),
)
@click.option(
    "--split",
    "split_path",
    required=True,
    type=click.Path(path_type=Path, exists=True, dir_okay=False),
)
@click.option(
    "--output-dir", required=True, type=click.Path(path_type=Path, file_okay=False)
)
@click.option("--epochs", type=int, default=20)
@click.option("--batch-size", type=int, default=128)
@click.option("--latent-dim", type=int, default=16)
@click.option("--supervised-dim", type=int, default=4)
@click.option("--learning-rate", type=float, default=1e-4)
@click.option("--seed", type=int, default=42)
@click.option("--workers", type=int, default=0)
@click.option(
    "--accelerator",
    type=click.Choice(["auto", "cpu", "gpu"]),
    default="auto",
)
def task4_train(
    cache_manifest: Path,
    split_path: Path,
    output_dir: Path,
    epochs: int,
    batch_size: int,
    latent_dim: int,
    supervised_dim: int,
    learning_rate: float,
    seed: int,
    workers: int,
    accelerator: str,
) -> None:
    _emit(
        train_guided_vae(
            cache_manifest,
            split_path,
            output_dir,
            epochs,
            batch_size,
            latent_dim,
            supervised_dim,
            learning_rate=learning_rate,
            seed=seed,
            workers=workers,
            accelerator=accelerator,
        ),
        None,
    )


@main.command("task4-evaluate")
@click.option(
    "--cache-manifest",
    required=True,
    type=click.Path(path_type=Path, exists=True, dir_okay=False),
)
@click.option(
    "--split",
    "split_path",
    required=True,
    type=click.Path(path_type=Path, exists=True, dir_okay=False),
)
@click.option(
    "--checkpoint",
    "checkpoint_path",
    required=True,
    type=click.Path(path_type=Path, exists=True, dir_okay=False),
)
@click.option(
    "--normalization",
    "normalization_path",
    required=True,
    type=click.Path(path_type=Path, exists=True, dir_okay=False),
)
@click.option("--class-count", type=int, default=4)
@click.option(
    "--secondary-cache-manifest",
    type=click.Path(path_type=Path, exists=True, dir_okay=False),
    default=None,
)
@click.option(
    "--secondary-split",
    type=click.Path(path_type=Path, exists=True, dir_okay=False),
    default=None,
)
@_path_option
def task4_evaluate(
    cache_manifest: Path,
    split_path: Path,
    checkpoint_path: Path,
    normalization_path: Path,
    class_count: int,
    secondary_cache_manifest: Path | None,
    secondary_split: Path | None,
    output: Path | None,
) -> None:
    _emit(
        evaluate_guided_vae(
            cache_manifest,
            split_path,
            checkpoint_path,
            normalization_path,
            class_count,
            secondary_cache_manifest,
            secondary_split,
        ),
        output,
    )


@main.command("task4-counterfactual")
@click.option(
    "--cache-manifest",
    required=True,
    type=click.Path(path_type=Path, exists=True, dir_okay=False),
)
@click.option(
    "--split",
    "split_path",
    required=True,
    type=click.Path(path_type=Path, exists=True, dir_okay=False),
)
@click.option(
    "--checkpoint",
    "checkpoint_path",
    required=True,
    type=click.Path(path_type=Path, exists=True, dir_okay=False),
)
@click.option(
    "--normalization",
    "normalization_path",
    required=True,
    type=click.Path(path_type=Path, exists=True, dir_okay=False),
)
@click.option("--dataset-index", type=int, required=True)
@click.option("--waypoints", type=int, default=20)
@click.option("--top-k", type=int, default=10)
@_path_option
def task4_counterfactual(
    cache_manifest: Path,
    split_path: Path,
    checkpoint_path: Path,
    normalization_path: Path,
    dataset_index: int,
    waypoints: int,
    top_k: int,
    output: Path | None,
) -> None:
    _emit(
        generate_counterfactual(
            cache_manifest,
            split_path,
            checkpoint_path,
            normalization_path,
            dataset_index,
            waypoints,
            top_k,
        ),
        output,
    )
