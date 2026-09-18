from dataclasses import dataclass

import numpy as np
from sklearn.model_selection import GroupShuffleSplit, train_test_split


@dataclass(frozen=True)
class SplitIndices:
    train: np.ndarray
    validation: np.ndarray
    test: np.ndarray


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
