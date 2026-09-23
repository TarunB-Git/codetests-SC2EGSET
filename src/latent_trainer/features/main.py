import logging
from pathlib import Path
from typing import Callable

import click
import lightning as pl

from latent_trainer.features.preprocess_dataset import (
    TransformEnumFunction,
    preprocess_dataset_chunked_profile,
    preprocess_dataset_test_only,
)
from latent_trainer.settings import DATA_DIR, LOGGING_FORMAT


@click.command(
    help="Pre-process the SC2_Dataset in a JSON format and cache transformed tensors to disk."
)
@click.option(
    "--transform",
    type=TransformEnumFunction(
        ["rich", "averaged_economy", "historical_averaged_economy"]
    ),
    default="rich",
    show_default=True,
    help="Transform to use: rich, averaged_economy, or historical_averaged_economy",
)
@click.option(
    "--single_json_dataset_path",
    type=click.Path(exists=True, dir_okay=False, path_type=Path, resolve_path=True),
    default=Path("H:/sc2egset_merged/sc2egset_merged.json").resolve(),
    help="Path to the single JSON dataset file.",
    show_default=True,
)
@click.option(
    "--n_workers",
    type=int,
    default=24,
    show_default=True,
    help="Number of parallel workers for processing replays.",
)
@click.option(
    "--n_samples",
    type=int,
    default=0,
    show_default=True,
    help="Total games to randomly sample before processing. 0 = use all.",
)
@click.option(
    "--seed",
    type=int,
    default=42,
    show_default=True,
    help="Random seed for reproducible sampling.",
)
@click.option(
    "--test_only",
    is_flag=True,
    default=False,
    help=(
        "Extract a test-only dataset. Pools all split indices, samples n_samples from "
        "the full pool, and stores everything in test_features/test_labels with empty "
        "train/val. Output: cached_dataset_<transform>_test_<n>.pt. "
        "Use this for path-charting evaluation on an independent dataset."
    ),
)
def main(
    transform: Callable,
    single_json_dataset_path: Path,
    n_workers: int,
    n_samples: int,
    seed: int,
    test_only: bool,
) -> None:
    """Pre-process Single JSON SC2_Dataset and cache the transformed tensors to drive."""
    transform_name = TransformEnumFunction._TRANSFORM_NAMES[transform]

    logging.basicConfig(
        level=logging.INFO,
        format=LOGGING_FORMAT,
    )

    pl.seed_everything(seed)

    try:
        if test_only:
            preprocess_dataset_test_only(
                output_directory=DATA_DIR,
                single_json_dataset_path=single_json_dataset_path,
                transform_fn=transform,
                transform_name=transform_name,
                n_workers=n_workers,
                n_samples=n_samples,
                seed=seed,
            )
            return
        preprocess_dataset_chunked_profile(
            output_directory=DATA_DIR,
            single_json_dataset_path=single_json_dataset_path,
            transform_fn=transform,
            transform_name=transform_name,
            n_workers=n_workers,
            n_samples=n_samples,
            seed=seed,
        )
    except Exception as error:
        logging.exception("Dataset preprocessing failed")
        raise click.ClickException(str(error)) from error


if __name__ == "__main__":
    main()
