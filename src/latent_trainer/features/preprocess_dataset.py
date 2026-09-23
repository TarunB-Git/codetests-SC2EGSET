"""Pre-process the SC2EGSet dataset and cache transformed tensors to disk.

This script iterates through all replays once, applies the transform,
catches broken replays, and saves valid (features, label) pairs to a .pt file.
Subsequent training runs can load from this cache instantly.

Usage:
    # Default (rich transform):
    uv run python src/latent_trainer/features/preprocess_dataset.py

    # Legacy economy-average transform:
    uv run python src/latent_trainer/features/preprocess_dataset.py --transform economy
"""

import logging
from concurrent.futures import (
    FIRST_COMPLETED,
    ProcessPoolExecutor,
    ThreadPoolExecutor,
    as_completed,
    wait,
)
from dataclasses import asdict
from pathlib import Path
from typing import Callable

import click
import torch
from sc2_datasets.lightning.sc2_egset_datamodule import (
    SC2EGSetDataModuleSingleJSON,
)
from sc2_datasets.replay_data.sc2_replay_data import SC2ReplayData
from torch.utils.data import Dataset
from torch.utils.data.dataloader import DataLoader
from tqdm import tqdm

from latent_trainer.benchmarks.transforms.aligned_economy import (
    economy_average_players_vs_outcomes,
    historical_economy_average_players_vs_outcomes,
)
from latent_trainer.features.rich_transform import rich_transform
from latent_trainer.features.type import CachedDatasetFileSpec
from latent_trainer.settings import DATA_DIR


def _transform_single_object(
    dataset_object: DataLoader,
    index: int,
    transform_fn: Callable,
) -> tuple[torch.Tensor, torch.Tensor] | None | Exception:

    try:
        replay = dataset_object[index]

        result = process_replay(
            replay=replay,
            transform_fn=transform_fn,
        )

        if result is None:
            return None
    except Exception as e:
        return e

    return result


def process_batch(
    executor: ProcessPoolExecutor | ThreadPoolExecutor,
    n_workers: int,
    dataset_object: Dataset,
    transform_fn: Callable,
    start_index: int,
    end_index: int,
) -> tuple[list[torch.Tensor], list[torch.Tensor], int, int]:
    set_features = []
    set_labels = []
    skipped = 0
    errors = 0

    with executor(max_workers=n_workers) as executor:
        futures = {
            executor.submit(
                _transform_single_object,
                dataset_object=dataset_object,
                index=i,
                transform_fn=transform_fn,
            ): i
            for i in range(start_index, end_index)
        }
        results = {}

        for future in tqdm(
            as_completed(futures),
            desc=f"  Batch {start_index}:{end_index} (parallel)",
            total=len(futures),
        ):
            results[futures[future]] = future.result()

        for index in range(start_index, end_index):
            result = results[index]
            if isinstance(result, Exception):
                errors += 1
            elif result is None:
                skipped += 1
            else:
                features, label = result
                set_features.append(
                    features
                    if isinstance(features, torch.Tensor)
                    else torch.tensor(features, dtype=torch.float32)
                )
                set_labels.append(label)

    return set_features, set_labels, skipped, errors


def process_set(
    dataset_object: DataLoader,
    executor: ProcessPoolExecutor | ThreadPoolExecutor,
    transform_fn: Callable[[SC2ReplayData], tuple[torch.Tensor, torch.Tensor]] = None,
    n_workers: int = 24,
) -> tuple[list[torch.Tensor], list[torch.Tensor], int, int]:
    """
    Process a dataset split (train/test/val) in parallel,
    apply the transform, and return lists of features and labels along with counts of skipped and errored replays.

    Parameters
    ----------
    dataset_object : DataLoader
        Dataloader for the dataset split to process (train/test/val)
    transform_fn : Callable[[SC2ReplayData], tuple[torch.Tensor, torch.Tensor]], optional
        Function to apply as the transform, by default None
    n_workers : int, optional
        Number of worker processes to use for parallel processing, by default 24

    Returns
    -------
    tuple[list[torch.Tensor], list[torch.Tensor], int, int]
        A tuple containing:
        - List of feature tensors
        - List of label tensors
        - Count of skipped replays (where transform returned None)
        - Count of errors encountered during processing
    """

    skipped = 0
    errors = 0

    set_features = []
    set_labels = []

    batch_size = 20_000
    total_size = len(dataset_object)

    for start_index in range(0, total_size, batch_size):
        end_index = min(start_index + batch_size, total_size)
        batch_features, batch_labels, batch_skipped, batch_errors = process_batch(
            executor=executor,
            n_workers=n_workers,
            dataset_object=dataset_object,
            transform_fn=transform_fn,
            start_index=start_index,
            end_index=end_index,
        )

        set_features.extend(batch_features)
        set_labels.extend(batch_labels)
        skipped += batch_skipped
        errors += batch_errors

    return set_features, set_labels, skipped, errors


def process_replay(
    replay: SC2ReplayData,
    transform_fn: Callable[[SC2ReplayData], tuple[torch.Tensor, int | torch.Tensor]],
) -> tuple[torch.Tensor, int | torch.Tensor] | None:
    """Apply transform and return (features, label) or None."""
    # Rich transform takes the raw replay directly
    result = transform_fn(replay)

    if result is None:
        return None

    features, label = result

    if features is None or label is None:
        return None

    label_tensor = torch.as_tensor(label)
    if torch.any(label_tensor == -1).item():
        return None

    return features, label


def stack_labels(labels: list) -> torch.Tensor:
    return torch.stack([torch.as_tensor(label, dtype=torch.long) for label in labels])


def check_split(
    train_dataset: DataLoader,
    test_dataset: DataLoader,
    val_dataset: DataLoader,
) -> None:

    total = len(train_dataset) + len(test_dataset) + len(val_dataset)
    logging.info(
        f"  Total replays: {total} (train: {len(train_dataset)}, test: {len(test_dataset)}, val: {len(val_dataset)})"
    )
    if total == 0:
        logging.warning("Dataset is empty! No replays found.")
        return

    train_frac = len(train_dataset) / total
    test_frac = len(test_dataset) / total
    val_frac = len(val_dataset) / total

    # Check if within 1.5% of target
    if (
        abs(train_frac - 0.8) > 0.015
        or abs(test_frac - 0.1) > 0.015
        or abs(val_frac - 0.1) > 0.015
    ):
        logging.warning(
            f"Dataset split deviates from 80/10/10! "
            f"Got: Train={train_frac:.2%}, Test={test_frac:.2%}, Val={val_frac:.2%}"
        )
        raise ValueError(
            f"Dataset split deviates from 80/10/10! "
            f"Got: Train={train_frac:.2%}, Test={test_frac:.2%}, Val={val_frac:.2%}"
        )

    return total


def preprocess_dataset(
    transform_name: str,
    transform_fn: Callable[[SC2ReplayData], tuple[torch.Tensor, torch.Tensor]],
    single_json_dataset_path: Path | str,
    output_directory: Path | str = DATA_DIR,
    n_workers: int = 24,
) -> None:
    """
    Preprocess a single JSON dataset and cache the transformed tensors to disk.

    Parameters
    ----------
    transform_name : str
        The name of the transformation (e.g. ``"rich"`` or ``"averaged_economy"``).
        Used to label the output file as ``cached_dataset_<transform_name>.pt``.
    transform_fn : Callable[[SC2ReplayData], tuple[torch.Tensor, torch.Tensor]]
        Function that maps a raw SC2ReplayData replay to a (features, label) pair.
    dataset_name : str, optional
        Name of the dataset, matching the JSON file stem, by default ``"sc2egset_merged"``.
    single_json_dataset_path : Path | str, optional
        Path to the single-JSON index file for the dataset,
        by default ``H:/sc2egset_merged/sc2egset_merged.json``.
    output_directory : Path | str, optional
        Directory where the cached ``.pt`` file will be written,
        by default ``./data``.
    n_workers : int, optional
        Number of worker processes for parallel replay processing, by default 24.
    """

    logging.info(f"SC2EGSet Dataset Pre-processing ({transform_name} transform)")

    # Initialize datamodule (this downloads + extracts if needed)
    logging.info("[1/3] Loading SC2EGSet datamodule (downloading if needed)...")

    datamodule = SC2EGSetDataModuleSingleJSON(
        json_path=single_json_dataset_path,
        download=False,
    )

    # Initialize the datamodule to get train/test/val splits (but skip any transforms for now)
    datamodule.prepare_data()
    datamodule.setup("fit")

    # Get train, test, and val datasets
    train_dataset = datamodule.train_dataset
    test_dataset = datamodule.test_dataset
    val_dataset = datamodule.val_dataset

    train_dataset.indices = sorted(train_dataset.indices)
    test_dataset.indices = sorted(test_dataset.indices)
    val_dataset.indices = sorted(val_dataset.indices)

    total = check_split(
        train_dataset=train_dataset,
        test_dataset=test_dataset,
        val_dataset=val_dataset,
    )

    # Process all replays
    logging.info("[2/3] Processing replays and applying transform...")

    logging.info("  Processing training set...")
    train_features, train_labels, skipped_train, errors_train = process_set(
        dataset_object=train_dataset,
        executor=ProcessPoolExecutor if n_workers > 1 else ThreadPoolExecutor,
        transform_fn=transform_fn,
        n_workers=n_workers,
    )
    logging.info("  Processing test set...")
    test_features, test_labels, skipped_test, errors_test = process_set(
        dataset_object=test_dataset,
        executor=ProcessPoolExecutor if n_workers > 1 else ThreadPoolExecutor,
        transform_fn=transform_fn,
        n_workers=n_workers,
    )
    logging.info("  Processing validation set...")
    val_features, val_labels, skipped_val, errors_val = process_set(
        dataset_object=val_dataset,
        executor=ProcessPoolExecutor if n_workers > 1 else ThreadPoolExecutor,
        transform_fn=transform_fn,
        n_workers=n_workers,
    )

    # Stack into tensors
    logging.info("\n[3/3] Saving cached dataset...")

    train_features_tensor = torch.stack(train_features)
    train_labels_tensor = stack_labels(train_labels)
    test_features_tensor = torch.stack(test_features)
    test_labels_tensor = stack_labels(test_labels)
    val_features_tensor = torch.stack(val_features)
    val_labels_tensor = stack_labels(val_labels)

    file_spec = CachedDatasetFileSpec(
        train_features=train_features_tensor,
        train_labels=train_labels_tensor,
        test_features=test_features_tensor,
        test_labels=test_labels_tensor,
        val_features=val_features_tensor,
        val_labels=val_labels_tensor,
        transform=transform_name,
    )

    # Save
    output_path = Path(output_directory)
    output_path.mkdir(parents=True, exist_ok=True)
    path_to_save = output_path / f"cached_dataset_{transform_name}.pt"
    torch.save(
        asdict(file_spec),
        path_to_save,
    )

    file_size_mb = path_to_save.stat().st_size / (1024 * 1024)

    logging.info(f"\n{'=' * 60}")
    logging.info("Pre-processing complete!")
    logging.info(f"  Transform:        {transform_name}")
    logging.info(f"  Total replays:    {total}")
    logging.info(f"  Valid train:      {len(train_features_tensor)}")
    logging.info(f"  Valid val:        {len(val_features_tensor)}")
    logging.info(f"  Valid test:       {len(test_features_tensor)}")
    logging.info(f"  Skipped (None):   {skipped_train + skipped_val + skipped_test}")
    logging.info(f"  Errors:           {errors_train + errors_val + errors_test}")
    logging.info(f"  Feature shape:    {train_features_tensor.shape}")
    logging.info(f"  Cache file:       {path_to_save} ({file_size_mb:.1f} MB)")
    logging.info(f"{'=' * 60}")


class TransformEnumFunction(click.Choice):
    """Custom Click Choice type that returns the actual transform function instead of the string name."""

    _TRANSFORM_NAMES: dict[Callable, str] = {
        rich_transform: "rich",
        economy_average_players_vs_outcomes: "averaged_economy",
        historical_economy_average_players_vs_outcomes: "historical_averaged_economy",
    }

    def convert(self, value, param, ctx):
        match value:
            case "rich":
                return rich_transform
            case "averaged_economy":
                return economy_average_players_vs_outcomes
            case "historical_averaged_economy":
                return historical_economy_average_players_vs_outcomes
            case _:
                raise click.BadParameter(f"Invalid transform choice: {value}")


def _transform_index_chunk(
    dataset_object: DataLoader,
    indices: list[int],
    transform_fn: Callable,
) -> tuple[list[torch.Tensor], list[torch.Tensor], int, int]:
    """Process a contiguous chunk of replay indices in one worker call."""

    set_features: list[torch.Tensor] = []
    set_labels: list[torch.Tensor] = []
    skipped = 0
    errors = 0

    for index in indices:
        result = _transform_single_object(
            dataset_object=dataset_object,
            index=index,
            transform_fn=transform_fn,
        )

        if isinstance(result, Exception):
            errors += 1
            logging.error(f"Error processing replay at index {index}: {result}")
        elif result is None:
            skipped += 1
        else:
            features, label = result
            set_features.append(
                features
                if isinstance(features, torch.Tensor)
                else torch.tensor(features, dtype=torch.float32)
            )
            set_labels.append(label)

    return set_features, set_labels, skipped, errors


def process_set_chunked_single_pool(
    dataset_object: DataLoader,
    executor: ProcessPoolExecutor | ThreadPoolExecutor,
    transform_fn: Callable[[SC2ReplayData], tuple[torch.Tensor, torch.Tensor]] = None,
    n_workers: int = 8,
    chunk_size: int = 256,
    max_inflight_tasks: int | None = None,
    set_name: str = "train",
) -> tuple[list[torch.Tensor], list[torch.Tensor], int, int]:
    """Alternative fast path: chunked index tasks with one persistent process pool.

    This function is meant for profiling/benchmarking against ``process_set``.
    It avoids recreating process pools and reduces scheduling overhead by
    submitting chunks of indices as single tasks.
    """
    if max_inflight_tasks is None:
        max_inflight_tasks = max(2, int(n_workers * 1.1))

    total_size = len(dataset_object)
    chunks: list[list[int]] = [
        list(range(start, min(start + chunk_size, total_size)))
        for start in range(0, total_size, chunk_size)
    ]

    set_features: list[torch.Tensor] = []
    set_labels: list[torch.Tensor] = []
    skipped = 0
    errors = 0

    pending_futures = {}
    next_chunk = 0
    results = {}

    with executor(max_workers=n_workers) as executor:
        with tqdm(total=total_size, desc=f"{set_name} (chunked-single-pool)") as pbar:
            while (
                next_chunk < len(chunks) and len(pending_futures) < max_inflight_tasks
            ):
                future = executor.submit(
                    _transform_index_chunk,
                    dataset_object=dataset_object,
                    indices=chunks[next_chunk],
                    transform_fn=transform_fn,
                )
                pending_futures[future] = next_chunk
                next_chunk += 1

            while pending_futures:
                done, _ = wait(pending_futures, return_when=FIRST_COMPLETED)

                for future in done:
                    chunk_index = pending_futures.pop(future)

                    chunk_features, chunk_labels, chunk_skipped, chunk_errors = (
                        future.result()
                    )
                    results[chunk_index] = (chunk_features, chunk_labels)
                    skipped += chunk_skipped
                    errors += chunk_errors

                    pbar.update(len(chunk_features) + chunk_skipped + chunk_errors)

                    if next_chunk < len(chunks):
                        new_future = executor.submit(
                            _transform_index_chunk,
                            dataset_object=dataset_object,
                            indices=chunks[next_chunk],
                            transform_fn=transform_fn,
                        )
                        pending_futures[new_future] = next_chunk
                        next_chunk += 1

    for chunk_index in range(len(chunks)):
        chunk_features, chunk_labels = results[chunk_index]
        set_features.extend(chunk_features)
        set_labels.extend(chunk_labels)

    return set_features, set_labels, skipped, errors


def preprocess_dataset_test_only(
    transform_name: str,
    transform_fn: Callable[[SC2ReplayData], tuple[torch.Tensor, torch.Tensor]],
    single_json_dataset_path: Path | str,
    output_directory: Path | str = DATA_DIR,
    n_workers: int = 8,
    chunk_size: int = 64,
    max_inflight_tasks: int | None = None,
    n_samples: int = 0,
    seed: int = 42,
) -> None:
    """Extract a test-only cached dataset from any SC2EGSet JSON file.

    Pools all available indices (train + val + test splits), optionally
    subsamples *n_samples* of them, processes every selected replay through
    the transform, and stores the result exclusively in the ``test_features``
    / ``test_labels`` fields.  ``train_*`` and ``val_*`` tensors are left
    empty.

    Use this when you already have a trained model and want to evaluate
    path-charting on an independent dataset without re-splitting it 80/10/10.
    """
    logging.info(
        "SC2EGSet Dataset Pre-processing (test-only): "
        f"transform={transform_name}, n_workers={n_workers}, n_samples={n_samples}"
    )

    datamodule = SC2EGSetDataModuleSingleJSON(
        json_path=single_json_dataset_path,
        download=False,
    )
    datamodule.prepare_data()
    datamodule.setup("fit")

    train_dataset = datamodule.train_dataset
    test_dataset = datamodule.test_dataset
    val_dataset = datamodule.val_dataset

    all_indices: list[int] = sorted(
        list(train_dataset.indices)
        + list(test_dataset.indices)
        + list(val_dataset.indices)
    )
    total = len(all_indices)
    logging.info(f"  Total available replays: {total}")

    # Shuffle once with the seeded generator so each batch draws new indices
    # without repeats and without rebuilding a "used" set.
    generator = torch.Generator()
    generator.manual_seed(seed)
    perm = torch.randperm(total, generator=generator).tolist()
    shuffled: list[int] = [all_indices[i] for i in perm]

    # Iterative batched processing: keep pulling batches from the shuffled
    # list until we have n_samples valid results (or the pool is exhausted).
    # The initial batch size is 1.5× the target to account for skips; subsequent
    # batches are sized from the observed yield rate.
    target = n_samples if n_samples > 0 else total
    batch_size = min(int(target * 1.5) + 1, total)

    all_features: list[torch.Tensor] = []
    all_labels: list = []
    total_skipped = 0
    total_errors = 0
    offset = 0

    while len(all_features) < target and offset < total:
        batch = shuffled[offset : offset + batch_size]
        offset += len(batch)

        test_dataset.indices = sorted(batch)
        batch_features, batch_labels, batch_skipped, batch_errors = (
            process_set_chunked_single_pool(
                dataset_object=test_dataset,
                executor=ProcessPoolExecutor if n_workers > 1 else ThreadPoolExecutor,
                transform_fn=transform_fn,
                n_workers=n_workers,
                chunk_size=chunk_size,
                max_inflight_tasks=max_inflight_tasks,
                set_name="test",
            )
        )
        all_features.extend(batch_features)
        all_labels.extend(batch_labels)
        total_skipped += batch_skipped
        total_errors += batch_errors

        logging.info(
            f"  Progress: {len(all_features)}/{target} valid "
            f"({total_skipped} skipped, {total_errors} errors, {offset}/{total} processed)"
        )

        if len(all_features) < target and offset < total:
            remaining_needed = target - len(all_features)
            yield_rate = len(all_features) / max(offset, 1)
            batch_size = min(
                int(remaining_needed / max(yield_rate, 0.01) * 1.2) + 1,
                total - offset,
            )

    # Truncate to exactly target if we overshot
    all_features = all_features[:target]
    all_labels = all_labels[:target]

    logging.info(
        f"  Done: {len(all_features)} valid, {total_skipped} skipped, {total_errors} errors."
    )

    test_features_tensor = torch.stack(all_features)
    test_labels_tensor = stack_labels(all_labels)
    empty_features = torch.empty(0, *test_features_tensor.shape[1:])
    empty_labels = torch.empty(0, *test_labels_tensor.shape[1:], dtype=torch.long)

    file_spec = CachedDatasetFileSpec(
        train_features=empty_features,
        train_labels=empty_labels,
        test_features=test_features_tensor,
        test_labels=test_labels_tensor,
        val_features=empty_features,
        val_labels=empty_labels,
        transform=transform_name,
    )

    output_path = Path(output_directory)
    output_path.mkdir(parents=True, exist_ok=True)
    n_tag = f"_{len(all_features)}" if n_samples > 0 else ""
    path_to_save = output_path / f"cached_dataset_{transform_name}_test{n_tag}.pt"
    torch.save(asdict(file_spec), path_to_save)

    file_size_mb = path_to_save.stat().st_size / (1024 * 1024)
    logging.info(f"\n{'=' * 60}")
    logging.info("Test-only pre-processing complete!")
    logging.info(f"  Transform:     {transform_name}")
    logging.info(f"  Valid test:    {len(test_features_tensor)}")
    logging.info(f"  Feature shape: {test_features_tensor.shape}")
    logging.info(f"  Cache file:    {path_to_save} ({file_size_mb:.1f} MB)")
    logging.info(f"{'=' * 60}")


def _subsample_indices(
    indices: list[int],
    n_target: int,
    generator: torch.Generator,
) -> list[int]:
    """Return a sorted random subset of *indices* of size *n_target* using *generator*."""
    if n_target <= 0 or n_target >= len(indices):
        return indices
    perm = torch.randperm(len(indices), generator=generator)[:n_target]
    return sorted(torch.tensor(indices)[perm].tolist())


def preprocess_dataset_chunked_profile(
    transform_name: str,
    transform_fn: Callable[[SC2ReplayData], tuple[torch.Tensor, torch.Tensor]],
    single_json_dataset_path: Path | str,
    output_directory: Path | str = DATA_DIR,
    n_workers: int = 8,
    chunk_size: int = 64,
    max_inflight_tasks: int | None = None,
    n_samples: int = 0,
    seed: int = 42,
) -> None:
    """Alternative preprocess entrypoint for performance profiling.

    This leaves ``preprocess_dataset`` unchanged and writes a separate cache file
    with ``_chunked`` suffix for side-by-side comparison.
    """

    logging.info(
        "SC2EGSet Dataset Pre-processing (chunked profile path): "
        f"transform={transform_name}, n_workers={n_workers}, "
        f"chunk_size={chunk_size}, max_inflight_tasks={max_inflight_tasks}"
    )

    datamodule = SC2EGSetDataModuleSingleJSON(
        json_path=single_json_dataset_path,
        download=False,
    )
    datamodule.prepare_data()
    datamodule.setup("fit")

    train_dataset = datamodule.train_dataset
    test_dataset = datamodule.test_dataset
    val_dataset = datamodule.val_dataset

    train_dataset.indices = sorted(train_dataset.indices)
    test_dataset.indices = sorted(test_dataset.indices)
    val_dataset.indices = sorted(val_dataset.indices)

    total = check_split(
        train_dataset=train_dataset,
        test_dataset=test_dataset,
        val_dataset=val_dataset,
    )

    if n_samples > 0:
        generator = torch.Generator()
        generator.manual_seed(seed)
        n_train = round(n_samples * len(train_dataset.indices) / total)
        n_test = round(n_samples * len(test_dataset.indices) / total)
        n_val = n_samples - n_train - n_test
        train_dataset.indices = _subsample_indices(
            train_dataset.indices, n_train, generator
        )
        test_dataset.indices = _subsample_indices(
            test_dataset.indices, n_test, generator
        )
        val_dataset.indices = _subsample_indices(val_dataset.indices, n_val, generator)
        logging.info(
            f"  Sampled {len(train_dataset.indices)} train / "
            f"{len(test_dataset.indices)} test / "
            f"{len(val_dataset.indices)} val from {total} total (seed={seed})"
        )

    val_features, val_labels, skipped_val, errors_val = process_set_chunked_single_pool(
        dataset_object=val_dataset,
        executor=ProcessPoolExecutor if n_workers > 1 else ThreadPoolExecutor,
        transform_fn=transform_fn,
        n_workers=n_workers,
        chunk_size=chunk_size,
        max_inflight_tasks=max_inflight_tasks,
        set_name="val",
    )

    logging.info(
        f"Validation set complete: {len(val_features)} valid, {skipped_val} skipped, {errors_val} errors."
    )

    val_features_tensor = torch.stack(val_features)
    val_labels_tensor = stack_labels(val_labels)

    test_features, test_labels, skipped_test, errors_test = (
        process_set_chunked_single_pool(
            dataset_object=test_dataset,
            executor=ProcessPoolExecutor if n_workers > 1 else ThreadPoolExecutor,
            transform_fn=transform_fn,
            n_workers=n_workers,
            chunk_size=chunk_size,
            max_inflight_tasks=max_inflight_tasks,
            set_name="test",
        )
    )

    test_features_tensor = torch.stack(test_features)
    test_labels_tensor = stack_labels(test_labels)

    train_features, train_labels, skipped_train, errors_train = (
        process_set_chunked_single_pool(
            dataset_object=train_dataset,
            executor=ProcessPoolExecutor if n_workers > 1 else ThreadPoolExecutor,
            transform_fn=transform_fn,
            n_workers=n_workers,
            chunk_size=chunk_size,
            max_inflight_tasks=max_inflight_tasks,
            set_name="train",
        )
    )

    train_features_tensor = torch.stack(train_features)
    train_labels_tensor = stack_labels(train_labels)

    file_spec = CachedDatasetFileSpec(
        train_features=train_features_tensor,
        train_labels=train_labels_tensor,
        test_features=test_features_tensor,
        test_labels=test_labels_tensor,
        val_features=val_features_tensor,
        val_labels=val_labels_tensor,
        transform=transform_name,
    )

    output_path = Path(output_directory)
    output_path.mkdir(parents=True, exist_ok=True)
    suffix = f"_{n_samples}" if n_samples > 0 else ""
    path_to_save = output_path / f"cached_dataset_{transform_name}{suffix}.pt"
    torch.save(asdict(file_spec), path_to_save)

    logging.info(f"\n{'=' * 60}")
    logging.info("Chunked profile pre-processing complete!")
    logging.info(f"  Transform:        {transform_name}")
    logging.info(f"  Total replays:    {total}")
    logging.info(f"  Valid train:      {len(train_features_tensor)}")
    logging.info(f"  Valid val:        {len(val_features_tensor)}")
    logging.info(f"  Valid test:       {len(test_features_tensor)}")
    logging.info(f"  Skipped (None):   {skipped_train + skipped_val + skipped_test}")
    logging.info(f"  Errors:           {errors_train + errors_val + errors_test}")
    logging.info(f"  Feature shape:    {train_features_tensor.shape}")
    logging.info(f"  Cache file:       {path_to_save}")
    logging.info(f"{'=' * 60}")
