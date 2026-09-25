from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression

from latent_trainer.benchmarks.cache.manifest import CacheManifest
from latent_trainer.benchmarks.common.data import (
    GAME_LOOPS_PER_SECOND,
    PlayerReplaySample,
    load_cached_player_samples,
    load_player_samples,
)
from latent_trainer.benchmarks.common.metrics import (
    classification_metrics,
    reliability_bins,
)
from latent_trainer.benchmarks.common.models import (
    make_classifier,
    positive_probabilities,
    predict_sequence_model,
    train_sequence_model,
)
from latent_trainer.benchmarks.common.provenance import software_provenance
from latent_trainer.benchmarks.common.splits import (
    grouped_split,
    indices_from_split_manifest,
)
from latent_trainer.benchmarks.data.source import InputFormat
from latent_trainer.benchmarks.splits.manifest import SplitManifest


@dataclass(frozen=True)
class PrefixData:
    sample_keys: list[str]
    replay_ids: list[str]
    gameplay_state: np.ndarray
    prior: np.ndarray
    sequences: list[torch.Tensor]
    labels: np.ndarray
    mmr: np.ndarray
    last_observed_loops: np.ndarray
    cutoff_loop: int


def _paired_prefix(
    sample: PlayerReplaySample,
    cutoff_loop: int,
) -> tuple[torch.Tensor, int] | None:
    own_positions = {
        int(loop): sample.sequence[index]
        for index, loop in enumerate(sample.sequence_loops)
        if int(loop) <= cutoff_loop
    }
    opponent_positions = {
        int(loop): sample.opponent_sequence[index]
        for index, loop in enumerate(sample.opponent_sequence_loops)
        if int(loop) <= cutoff_loop
    }
    loops = sorted(set(own_positions) | set(opponent_positions))
    own = None
    opponent = None
    paired = []
    paired_loops = []

    for loop in loops:
        if loop in own_positions:
            own = own_positions[loop]
        if loop in opponent_positions:
            opponent = opponent_positions[loop]
        if own is not None and opponent is not None:
            paired.append(torch.cat([own, opponent]))
            paired_loops.append(loop)

    if not paired:
        return None
    return torch.stack(paired), paired_loops[-1]


def build_prefix_data(
    samples: list[PlayerReplaySample],
    minute: float,
) -> PrefixData:
    cutoff_loop = int(round(minute * 60.0 * GAME_LOOPS_PER_SECOND))
    keys = []
    replay_ids = []
    states = []
    priors = []
    sequences = []
    labels = []
    mmr = []
    last_observed_loops = []

    for sample in samples:
        if sample.duration_loops < cutoff_loop:
            continue
        prefix = _paired_prefix(sample, cutoff_loop)
        if prefix is None:
            continue
        sequence, last_observed_loop = prefix
        keys.append(f"{sample.replay_id}:{sample.player_id}")
        replay_ids.append(sample.replay_id)
        states.append(sequence[-1].numpy())
        priors.append([sample.mmr, sample.opponent_mmr])
        sequences.append(sequence)
        labels.append(sample.outcome)
        mmr.append(sample.mmr)
        last_observed_loops.append(last_observed_loop)

    if not sequences:
        raise ValueError(f"No samples remain at {minute} minutes")

    return PrefixData(
        sample_keys=keys,
        replay_ids=replay_ids,
        gameplay_state=np.asarray(states, dtype=np.float32),
        prior=np.asarray(priors, dtype=np.float32),
        sequences=sequences,
        labels=np.asarray(labels, dtype=int),
        mmr=np.asarray(mmr, dtype=float),
        last_observed_loops=np.asarray(last_observed_loops, dtype=int),
        cutoff_loop=cutoff_loop,
    )


def _information_arrays(
    data: PrefixData,
    information: str,
) -> np.ndarray:
    if information == "prior-only":
        return data.prior
    if information == "gameplay-only":
        return data.gameplay_state
    if information == "combined":
        return np.concatenate([data.prior, data.gameplay_state], axis=1)
    raise ValueError(f"Unknown information setting: {information}")


def _information_sequences(
    data: PrefixData,
    information: str,
) -> list[torch.Tensor]:
    if information == "gameplay-only":
        return data.sequences
    if information == "combined":
        return [
            torch.cat(
                [
                    torch.tensor(data.prior[index], dtype=torch.float32)
                    .unsqueeze(0)
                    .expand(len(sequence), -1),
                    sequence,
                ],
                dim=1,
            )
            for index, sequence in enumerate(data.sequences)
        ]
    raise ValueError("Sequence models require gameplay-only or combined information")


def _skill_subgroups(mmr: np.ndarray, reference: np.ndarray) -> dict[str, np.ndarray]:
    valid_reference = reference[reference > 0]
    result = {"all": np.arange(len(mmr))}
    if len(valid_reference) < 3:
        return result
    low, high = np.quantile(valid_reference, [1 / 3, 2 / 3])
    result.update(
        {
            "low_mmr": np.flatnonzero((mmr > 0) & (mmr <= low)),
            "middle_mmr": np.flatnonzero((mmr > low) & (mmr <= high)),
            "high_mmr": np.flatnonzero(mmr > high),
        }
    )
    return result


def _evaluate_groups(
    labels: np.ndarray,
    probabilities: np.ndarray,
    mmr: np.ndarray,
    reference_mmr: np.ndarray,
    source: str,
) -> dict:
    result = {}
    for name, indices in _skill_subgroups(mmr, reference_mmr).items():
        if len(indices) == 0:
            continue
        result[name] = {
            "metrics": classification_metrics(labels[indices], probabilities[indices]),
            "calibration": reliability_bins(labels[indices], probabilities[indices]),
            "samples": int(len(indices)),
        }
    if source == "sc2egset":
        result["tournament_pro"] = result["all"]
    return result


def _calibrate_sequence_probabilities(
    validation_labels: np.ndarray,
    validation_probabilities: np.ndarray,
    test_probabilities: np.ndarray,
) -> np.ndarray:
    if np.unique(validation_labels).size < 2:
        raise ValueError("Calibration requires both outcome classes in validation")
    epsilon = 1e-6
    validation_logits = np.log(
        np.clip(validation_probabilities, epsilon, 1 - epsilon)
        / np.clip(1 - validation_probabilities, epsilon, 1 - epsilon)
    ).reshape(-1, 1)
    test_logits = np.log(
        np.clip(test_probabilities, epsilon, 1 - epsilon)
        / np.clip(1 - test_probabilities, epsilon, 1 - epsilon)
    ).reshape(-1, 1)
    calibrator = LogisticRegression().fit(validation_logits, validation_labels)
    return calibrator.predict_proba(test_logits)[:, 1]


def turning_points(
    predictions: dict[str, dict[float, float]],
    event_loops: dict[str, dict[float, int]] | None = None,
) -> dict[str, list[dict[str, object]]]:
    result: dict[str, list[dict[str, object]]] = {}
    for key, values in predictions.items():
        ordered = sorted(values.items())
        result[key] = [
            {
                "minute": float(minute),
                "probability": float(probability),
                "delta_probability": float(probability - ordered[index - 1][1])
                if index > 0
                else 0.0,
                "candidate_type": "model_derived_turning_point",
                "causal_claim": False,
                "mapped_replay_event": {
                    "event_type": "PlayerStats",
                    "game_loop": event_loops[key][minute],
                    "mapping": "latest observed PlayerStats at the prefix cutoff",
                }
                if event_loops is not None
                else None,
            }
            for index, (minute, probability) in enumerate(ordered)
        ]
    return result


def run_prefix_benchmark(
    json_path: Path | None,
    model_name: str,
    information: str,
    minutes: tuple[float, ...] = (1, 2, 3, 5, 7, 10),
    epochs: int = 10,
    max_replays: int = 0,
    calibrated: bool = True,
    source: str = "sc2ggset",
    seed: int = 42,
    offsets_path: Path | None = None,
    input_format: InputFormat = "auto",
    source_indices_path: Path | None = None,
    cache_manifest_path: Path | None = None,
    split_path: Path | None = None,
) -> dict:
    if model_name in {"gru", "transformer"} and information == "prior-only":
        raise ValueError("Prior-only uses a tabular model; choose logistic or xgboost")
    if cache_manifest_path is not None:
        samples = load_cached_player_samples(cache_manifest_path, max_replays)
    elif json_path is not None:
        samples = load_player_samples(
            json_path,
            max_replays=max_replays,
            offsets_path=offsets_path,
            input_format=input_format,
            source_indices_path=source_indices_path,
        )
    else:
        raise ValueError("Task 2 requires a JSON source or cache manifest")
    replay_ids = sorted({sample.replay_id for sample in samples})
    cache_fingerprint = (
        CacheManifest.load(cache_manifest_path).fingerprint()
        if cache_manifest_path is not None
        else None
    )
    replay_split = (
        indices_from_split_manifest(replay_ids, split_path, cache_fingerprint)
        if split_path is not None
        else grouped_split(replay_ids, seed=seed)
    )
    train_replays = {replay_ids[index] for index in replay_split.train}
    validation_replays = {replay_ids[index] for index in replay_split.validation}
    test_replays = {replay_ids[index] for index in replay_split.test}
    output = {}
    predictions: dict[str, dict[float, float]] = {}
    event_loops: dict[str, dict[float, int]] = {}

    for minute in minutes:
        data = build_prefix_data(samples, minute)
        train = np.asarray(
            [
                index
                for index, replay in enumerate(data.replay_ids)
                if replay in train_replays
            ]
        )
        test = np.asarray(
            [
                index
                for index, replay in enumerate(data.replay_ids)
                if replay in test_replays
            ]
        )
        validation = np.asarray(
            [
                index
                for index, replay in enumerate(data.replay_ids)
                if replay in validation_replays
            ]
        )

        invalid_prior_excluded = 0
        if information in {"prior-only", "combined"}:
            valid = np.all(data.prior > 0, axis=1)
            invalid_prior_excluded = int((~valid).sum())
            train = train[valid[train]]
            validation = validation[valid[validation]]
            test = test[valid[test]]

        if not len(train) or not len(validation) or not len(test):
            raise ValueError(f"Window {minute} has an empty data partition")
        if np.unique(data.labels[train]).size != 2:
            raise ValueError(
                f"Window {minute} training data requires both outcome classes"
            )

        if model_name in {"gru", "transformer"}:
            sequences = _information_sequences(data, information)
            trained = train_sequence_model(
                [sequences[index] for index in train],
                data.labels[train],
                name=model_name,
                task="classification",
                epochs=epochs,
                seed=seed,
            )
            probabilities = predict_sequence_model(
                trained, [sequences[index] for index in test]
            )
            if calibrated:
                validation_probabilities = predict_sequence_model(
                    trained, [sequences[index] for index in validation]
                )
                probabilities = _calibrate_sequence_probabilities(
                    data.labels[validation],
                    validation_probabilities,
                    probabilities,
                )
        else:
            features = _information_arrays(data, information)
            model = make_classifier(
                model_name,
                seed=seed,
                calibrated=False,
            )
            model.fit(features[train], data.labels[train])
            probabilities = positive_probabilities(model, features[test])
            if calibrated:
                validation_probabilities = positive_probabilities(
                    model, features[validation]
                )
                probabilities = _calibrate_sequence_probabilities(
                    data.labels[validation],
                    validation_probabilities,
                    probabilities,
                )

        for index, probability in zip(test, probabilities, strict=True):
            predictions.setdefault(data.sample_keys[index], {})[minute] = float(
                probability
            )
            event_loops.setdefault(data.sample_keys[index], {})[minute] = int(
                data.last_observed_loops[index]
            )

        output[str(minute)] = {
            "number_at_risk": int(len(data.labels)),
            "games_at_risk": int(len(set(data.replay_ids))),
            "train_samples": int(len(train)),
            "test_samples": int(len(test)),
            "invalid_prior_samples_excluded": invalid_prior_excluded,
            "train_class_counts": {
                str(value): int((data.labels[train] == value).sum())
                for value in np.unique(data.labels[train])
            },
            "test_class_counts": {
                str(value): int((data.labels[test] == value).sum())
                for value in np.unique(data.labels[test])
            },
            "saved_probabilities": [
                {
                    "sample_key": data.sample_keys[index],
                    "replay_id": data.replay_ids[index],
                    "probability": float(probability),
                }
                for index, probability in zip(test, probabilities, strict=True)
            ],
            "groups": _evaluate_groups(
                data.labels[test],
                probabilities,
                data.mmr[test],
                data.mmr[train],
                source,
            ),
        }

    return {
        "task": "2",
        "model": model_name,
        "information": information,
        "source": source,
        "split": str(split_path) if split_path is not None else "replay-grouped",
        "windows": output,
        "turning_points": turning_points(predictions, event_loops),
        "provenance": {
            "software": software_provenance(),
            "cache_fingerprint": cache_fingerprint,
            "split_fingerprint": SplitManifest.load(split_path).fingerprint()
            if split_path is not None
            else None,
        },
    }
