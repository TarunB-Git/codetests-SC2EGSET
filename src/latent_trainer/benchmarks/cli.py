import json
from pathlib import Path

import click

from latent_trainer.benchmarks.task1 import (
    run_sequence_benchmark,
    run_static_benchmark,
)
from latent_trainer.benchmarks.task2 import run_prefix_benchmark
from latent_trainer.benchmarks.task3 import run_skill_benchmark


def _emit(result: dict, output: Path | None) -> None:
    rendered = json.dumps(result, indent=2, sort_keys=True, allow_nan=True)
    if output is None:
        click.echo(rendered)
    else:
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
@click.option("--seed", type=int, default=42)
@_path_option
def task1_static(
    cache_path: Path,
    models: tuple[str, ...],
    view: str,
    protocol: str,
    calibrated: bool,
    evaluation_cache: Path | None,
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
        ),
        output,
    )


@main.command("task1-sequence")
@click.option(
    "--json-path",
    required=True,
    type=click.Path(path_type=Path, exists=True, dir_okay=False),
)
@click.option(
    "--model", "model_name", type=click.Choice(["gru", "transformer"]), required=True
)
@click.option("--epochs", type=int, default=10)
@click.option("--max-replays", type=int, default=0)
@click.option("--seed", type=int, default=42)
@_path_option
def task1_sequence(
    json_path: Path,
    model_name: str,
    epochs: int,
    max_replays: int,
    seed: int,
    output: Path | None,
) -> None:
    _emit(
        run_sequence_benchmark(
            json_path,
            model_name=model_name,
            epochs=epochs,
            max_replays=max_replays,
            seed=seed,
        ),
        output,
    )


@main.command("task2")
@click.option(
    "--json-path",
    required=True,
    type=click.Path(path_type=Path, exists=True, dir_okay=False),
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
    json_path: Path,
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
        ),
        output,
    )


@main.command("task3")
@click.option(
    "--json-path",
    required=True,
    type=click.Path(path_type=Path, exists=True, dir_okay=False),
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
@_path_option
def task3(
    json_path: Path,
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
    output: Path | None,
) -> None:
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
        ),
        output,
    )
