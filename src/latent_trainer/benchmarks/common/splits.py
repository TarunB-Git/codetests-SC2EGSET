from dataclasses import dataclass
from pathlib import Path

import numpy as np
from sklearn.model_selection import (
    GroupShuffleSplit,
    StratifiedGroupKFold,
    train_test_split,
)


@dataclass(frozen=True)
class SplitIndices:
    train: np.ndarray
    validation: np.ndarray
    test: np.ndarray


def _validate_fractions(validation_fraction: float, test_fraction: float) -> None:
    if validation_fraction < 0 or test_fraction <= 0:
        raise ValueError(
            "Split fractions must be non-negative and test must be positive"
        )
    if validation_fraction + test_fraction >= 1:
        raise ValueError("Validation and test fractions must sum to less than one")


def grouped_calibration_folds(
    targets: list[int] | np.ndarray,
    groups: list[str] | np.ndarray,
    max_splits: int = 3,
    seed: int = 42,
) -> list[tuple[np.ndarray, np.ndarray]]:
    target_values = np.asarray(targets, dtype=int)
    group_values = np.asarray(groups)
    if len(target_values) != len(group_values):
        raise ValueError("Targets and groups must have the same length")
    classes = np.unique(target_values)
    if classes.size != 2:
        raise ValueError("Calibration requires two outcome classes")
    groups_per_class = [
        np.unique(group_values[target_values == value]).size for value in classes
    ]
    n_splits = min(max_splits, np.unique(group_values).size, *groups_per_class)
    if n_splits < 2:
        raise ValueError("Calibration requires at least two groups per class")
    splitter = StratifiedGroupKFold(
        n_splits=n_splits,
        shuffle=True,
        random_state=seed,
    )
    return list(splitter.split(target_values, target_values, group_values))


def _validation_split(
    train_validation: np.ndarray,
    groups: np.ndarray,
    relative_fraction: float,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    if relative_fraction <= 0:
        return train_validation, np.array([], dtype=int)
    splitter = GroupShuffleSplit(
        n_splits=1, test_size=relative_fraction, random_state=seed
    )
    train_local, validation_local = next(
        splitter.split(train_validation, groups=groups[train_validation])
    )
    return train_validation[train_local], train_validation[validation_local]


def grouped_split(
    groups: list[str] | np.ndarray,
    validation_fraction: float = 0.1,
    test_fraction: float = 0.2,
    seed: int = 42,
) -> SplitIndices:
    _validate_fractions(validation_fraction, test_fraction)
    groups_array = np.asarray(groups)
    indices = np.arange(len(groups_array))
    unique_groups = np.unique(groups_array)
    if unique_groups.size < 3:
        raise ValueError("At least three groups are required for a three-way split")
    splitter = GroupShuffleSplit(n_splits=1, test_size=test_fraction, random_state=seed)
    train_validation_local, test_local = next(
        splitter.split(indices, groups=groups_array)
    )
    train, validation = _validation_split(
        indices[train_validation_local],
        groups_array,
        validation_fraction / (1.0 - test_fraction),
        seed + 1,
    )
    return SplitIndices(train=train, validation=validation, test=indices[test_local])


def player_held_out_split(
    player_ids: list[str] | np.ndarray,
    replay_ids: list[str] | np.ndarray,
    validation_fraction: float = 0.1,
    test_fraction: float = 0.2,
    seed: int = 42,
) -> SplitIndices:
    players = np.asarray(player_ids)
    replays = np.asarray(replay_ids)
    split = grouped_split(
        players,
        validation_fraction=validation_fraction,
        test_fraction=test_fraction,
        seed=seed,
    )
    test_replays = set(replays[split.test])
    validation = split.validation[
        np.asarray([replay not in test_replays for replay in replays[split.validation]])
    ]
    validation_replays = set(replays[validation])
    excluded_replays = test_replays | validation_replays
    train = split.train[
        np.asarray([replay not in excluded_replays for replay in replays[split.train]])
    ]
    result = SplitIndices(train=train, validation=validation, test=split.test)
    assert_disjoint_groups(result, players)
    assert_disjoint_groups(result, replays)
    return result


def random_split(
    size: int,
    validation_fraction: float = 0.1,
    test_fraction: float = 0.2,
    seed: int = 42,
) -> SplitIndices:
    _validate_fractions(validation_fraction, test_fraction)
    indices = np.arange(size)
    train_validation, test = train_test_split(
        indices, test_size=test_fraction, random_state=seed, shuffle=True
    )
    relative = validation_fraction / (1.0 - test_fraction)
    train, validation = train_test_split(
        train_validation, test_size=relative, random_state=seed + 1, shuffle=True
    )
    return SplitIndices(train=train, validation=validation, test=test)


def temporal_split(
    timestamps: list[str] | np.ndarray,
    validation_fraction: float = 0.1,
    test_fraction: float = 0.2,
) -> SplitIndices:
    _validate_fractions(validation_fraction, test_fraction)
    ordered = np.argsort(np.asarray(timestamps))
    test_start = int(round(len(ordered) * (1.0 - test_fraction)))
    validation_start = int(
        round(len(ordered) * (1.0 - test_fraction - validation_fraction))
    )
    return SplitIndices(
        train=ordered[:validation_start],
        validation=ordered[validation_start:test_start],
        test=ordered[test_start:],
    )


def mmr_shift_split(
    mmr: list[float] | np.ndarray,
    validation_quantile: float = 0.7,
    test_quantile: float = 0.85,
) -> SplitIndices:
    if not 0 < validation_quantile < test_quantile < 1:
        raise ValueError("MMR quantiles must satisfy 0 < validation < test < 1")
    values = np.asarray(mmr, dtype=float)
    validation_boundary, test_boundary = np.quantile(
        values, [validation_quantile, test_quantile]
    )
    return SplitIndices(
        train=np.flatnonzero(values < validation_boundary),
        validation=np.flatnonzero(
            (values >= validation_boundary) & (values < test_boundary)
        ),
        test=np.flatnonzero(values >= test_boundary),
    )


def version_split(
    versions: list[str] | np.ndarray,
    test_version: str,
    validation_version: str | None = None,
) -> SplitIndices:
    values = np.asarray(versions)
    test = np.flatnonzero(values == test_version)
    validation = (
        np.flatnonzero(values == validation_version)
        if validation_version is not None
        else np.array([], dtype=int)
    )
    excluded = np.isin(np.arange(len(values)), np.concatenate([test, validation]))
    return SplitIndices(
        train=np.flatnonzero(~excluded), validation=validation, test=test
    )


def assert_disjoint_groups(
    split: SplitIndices,
    groups: list[str] | np.ndarray,
) -> None:
    values = np.asarray(groups)
    train = set(values[split.train])
    validation = set(values[split.validation])
    test = set(values[split.test])
    if train & validation or train & test or validation & test:
        raise ValueError("Split groups overlap")


def indices_from_split_manifest(
    replay_ids: list[str] | np.ndarray,
    split_path: Path,
    cache_fingerprint: str | None = None,
) -> SplitIndices:
    from latent_trainer.benchmarks.splits.manifest import SplitManifest

    manifest = SplitManifest.load(split_path)
    if (
        cache_fingerprint is not None
        and manifest.cache_fingerprint != cache_fingerprint
    ):
        raise ValueError("Saved split does not belong to the supplied cache")
    values = np.asarray(replay_ids)
    groups = {
        name: set(manifest.splits[name]["replay_ids"])
        for name in ("train", "validation", "test")
    }
    split = SplitIndices(
        train=np.flatnonzero(np.isin(values, list(groups["train"]))),
        validation=np.flatnonzero(np.isin(values, list(groups["validation"]))),
        test=np.flatnonzero(np.isin(values, list(groups["test"]))),
    )
    assert_disjoint_groups(split, values)
    if not len(split.train) or not len(split.validation) or not len(split.test):
        raise ValueError("Saved split produces an empty sample partition")
    return split
