from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression

from latent_trainer.benchmarks.common.data import (
    GAME_LOOPS_PER_SECOND,
    PlayerReplaySample,
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
from latent_trainer.benchmarks.common.splits import grouped_split


@dataclass(frozen=True)
class PrefixData:
    sample_keys: list[str]
    replay_ids: list[str]
    gameplay_state: np.ndarray
    prior: np.ndarray
    sequences: list[torch.Tensor]
    labels: np.ndarray
    mmr: np.ndarray
    cutoff_loop: int


def _paired_prefix(
    sample: PlayerReplaySample,
    cutoff_loop: int,
) -> torch.Tensor | None:
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

    for loop in loops:
        if loop in own_positions:
            own = own_positions[loop]
        if loop in opponent_positions:
            opponent = opponent_positions[loop]
        if own is not None and opponent is not None:
            paired.append(torch.cat([own, opponent]))

    if not paired:
        return None
    return torch.stack(paired)


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

    for sample in samples:
        if sample.duration_loops < cutoff_loop:
            continue
        sequence = _paired_prefix(sample, cutoff_loop)
        if sequence is None:
            continue
        keys.append(f"{sample.replay_id}:{sample.player_id}")
        replay_ids.append(sample.replay_id)
        states.append(sequence[-1].numpy())
        priors.append([sample.mmr, sample.opponent_mmr])
        sequences.append(sequence)
        labels.append(sample.outcome)
        mmr.append(sample.mmr)

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
) -> dict[str, list[dict[str, float]]]:
    result = {}
    for key, values in predictions.items():
        ordered = sorted(values.items())
        result[key] = [
            {
                "minute": float(minute),
                "probability": float(probability),
                "delta_probability": float(probability - ordered[index - 1][1])
                if index > 0
                else 0.0,
            }
            for index, (minute, probability) in enumerate(ordered)
        ]
    return result


def run_prefix_benchmark(
    json_path: Path,
    model_name: str,
    information: str,
    minutes: tuple[float, ...] = (1, 2, 3, 5, 7, 10),
    epochs: int = 10,
    max_replays: int = 0,
    calibrated: bool = True,
    source: str = "sc2ggset",
    seed: int = 42,
) -> dict:
    samples = load_player_samples(json_path, max_replays=max_replays)
    replay_ids = sorted({sample.replay_id for sample in samples})
    replay_split = grouped_split(replay_ids, seed=seed)
    train_replays = {replay_ids[index] for index in replay_split.train}
    validation_replays = {replay_ids[index] for index in replay_split.validation}
    test_replays = {replay_ids[index] for index in replay_split.test}
    output = {}
    predictions: dict[str, dict[float, float]] = {}

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

        if information in {"prior-only", "combined"}:
            valid = np.all(data.prior > 0, axis=1)
            train = train[valid[train]]
            validation = validation[valid[validation]]
            test = test[valid[test]]

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
            model = make_classifier(model_name, seed=seed, calibrated=calibrated)
            model.fit(features[train], data.labels[train])
            probabilities = positive_probabilities(model, features[test])

        for index, probability in zip(test, probabilities, strict=True):
            predictions.setdefault(data.sample_keys[index], {})[minute] = float(
                probability
            )

        output[str(minute)] = {
            "number_at_risk": int(len(data.labels)),
            "games_at_risk": int(len(set(data.replay_ids))),
            "train_samples": int(len(train)),
            "test_samples": int(len(test)),
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
        "split": "replay-grouped",
        "windows": output,
        "turning_points": turning_points(predictions),
    }
