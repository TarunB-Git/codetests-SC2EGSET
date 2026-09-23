from dataclasses import dataclass
from typing import Protocol

import numpy as np
import torch
from sklearn.base import BaseEstimator
from sklearn.calibration import CalibratedClassifierCV
from sklearn.compose import TransformedTargetRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_sequence


class SequenceDataset(Protocol):
    def __len__(self) -> int: ...

    def __getitem__(self, index: int) -> tuple[torch.Tensor, float]: ...


def make_classifier(
    name: str,
    seed: int = 42,
    calibrated: bool = False,
    historical: bool = False,
    calibration_cv: int | list[tuple[np.ndarray, np.ndarray]] = 3,
) -> BaseEstimator:
    if name == "logistic":
        estimator = LogisticRegression(C=10, max_iter=2000, random_state=seed)
    elif name == "svm":
        estimator = SVC(
            kernel="rbf",
            C=10,
            gamma="auto",
            probability=not calibrated,
            random_state=seed,
        )
    elif name == "xgboost":
        from xgboost import XGBClassifier

        estimator = XGBClassifier(
            objective="binary:logistic",
            booster="gbtree",
            learning_rate=0.2,
            max_depth=5,
            n_estimators=100,
            eval_metric="logloss",
            n_jobs=1,
            random_state=seed,
        )
    elif name == "mlp":
        estimator = MLPClassifier(
            hidden_layer_sizes=(64, 32),
            early_stopping=True,
            max_iter=300,
            random_state=seed,
        )
    else:
        raise ValueError(f"Unknown classifier: {name}")

    if not historical and name in {"logistic", "svm", "mlp"}:
        estimator = make_pipeline(StandardScaler(), estimator)
    if calibrated:
        estimator = CalibratedClassifierCV(
            estimator,
            method="sigmoid",
            cv=calibration_cv,
        )
    return estimator


def make_regressor(name: str, seed: int = 42) -> BaseEstimator:
    if name == "xgboost":
        from xgboost import XGBRegressor

        return XGBRegressor(
            objective="reg:squarederror",
            learning_rate=0.05,
            max_depth=5,
            n_estimators=200,
            n_jobs=1,
            random_state=seed,
        )
    if name == "mlp":
        return TransformedTargetRegressor(
            regressor=make_pipeline(
                StandardScaler(),
                MLPRegressor(
                    hidden_layer_sizes=(64, 32),
                    early_stopping=True,
                    max_iter=400,
                    random_state=seed,
                ),
            ),
            transformer=StandardScaler(),
        )
    raise ValueError(f"Unknown regressor: {name}")


def make_multiclass_classifier(
    name: str,
    class_count: int,
    seed: int = 42,
) -> BaseEstimator:
    if name == "xgboost":
        from xgboost import XGBClassifier

        parameters: dict[str, object] = {
            "objective": "binary:logistic" if class_count == 2 else "multi:softprob",
            "eval_metric": "logloss" if class_count == 2 else "mlogloss",
        }
        if class_count > 2:
            parameters["num_class"] = class_count
        return XGBClassifier(
            learning_rate=0.2,
            max_depth=5,
            n_estimators=100,
            n_jobs=1,
            random_state=seed,
            **parameters,
        )
    if name == "mlp":
        return make_pipeline(
            StandardScaler(),
            MLPClassifier(
                hidden_layer_sizes=(64, 32),
                early_stopping=True,
                max_iter=300,
                random_state=seed,
            ),
        )
    raise ValueError(f"Unknown multiclass classifier: {name}")


def positive_probabilities(
    estimator: BaseEstimator, features: np.ndarray
) -> np.ndarray:
    if hasattr(estimator, "predict_proba"):
        return np.asarray(estimator.predict_proba(features))[:, 1]
    scores = np.asarray(estimator.decision_function(features), dtype=float)
    return 1.0 / (1.0 + np.exp(-np.clip(scores, -709, 709)))


def padded_batch(
    sequences: list[torch.Tensor],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    lengths = torch.tensor([len(sequence) for sequence in sequences], dtype=torch.long)
    padded = pad_sequence(sequences, batch_first=True)
    positions = torch.arange(padded.shape[1]).unsqueeze(0)
    mask = positions >= lengths.unsqueeze(1)
    return padded, lengths, mask


class GRUSequenceModel(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 64,
        num_layers: int = 1,
        output_dim: int = 1,
    ):
        super().__init__()
        self.gru = nn.GRU(
            input_dim,
            hidden_dim,
            num_layers=num_layers,
            batch_first=True,
        )
        self.output = nn.Linear(hidden_dim, output_dim)

    def forward(self, padded: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        packed = pack_padded_sequence(
            padded,
            lengths.cpu(),
            batch_first=True,
            enforce_sorted=False,
        )
        _, hidden = self.gru(packed)
        return self.output(hidden[-1]).squeeze(-1)


class TransformerSequenceModel(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 64,
        num_heads: int = 4,
        num_layers: int = 2,
        max_length: int = 2048,
        output_dim: int = 1,
    ):
        super().__init__()
        self.projection = nn.Linear(input_dim, hidden_dim)
        self.position = nn.Embedding(max_length, hidden_dim)
        layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 2,
            dropout=0.1,
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(
            layer, num_layers=num_layers, enable_nested_tensor=False
        )
        self.output = nn.Linear(hidden_dim, output_dim)

    def forward(
        self,
        padded: torch.Tensor,
        lengths: torch.Tensor,
        padding_mask: torch.Tensor,
    ) -> torch.Tensor:
        positions = torch.arange(padded.shape[1], device=padded.device)
        encoded = self.projection(padded) + self.position(positions).unsqueeze(0)
        encoded = self.encoder(encoded, src_key_padding_mask=padding_mask)
        last = encoded[
            torch.arange(encoded.shape[0], device=encoded.device),
            lengths.to(encoded.device) - 1,
        ]
        return self.output(last).squeeze(-1)


@dataclass
class TrainedSequenceModel:
    model: nn.Module
    name: str
    task: str
    mean: torch.Tensor
    std: torch.Tensor
    target_mean: float
    target_std: float


def _normalize_sequences(
    sequences: list[torch.Tensor],
    mean: torch.Tensor,
    std: torch.Tensor,
) -> list[torch.Tensor]:
    return [(sequence - mean) / std for sequence in sequences]


def train_sequence_model(
    sequences: list[torch.Tensor],
    targets: np.ndarray,
    name: str,
    task: str,
    epochs: int = 10,
    batch_size: int = 64,
    learning_rate: float = 1e-3,
    seed: int = 42,
    num_classes: int = 2,
) -> TrainedSequenceModel:
    if not sequences or any(len(sequence) == 0 for sequence in sequences):
        raise ValueError("Non-empty sequences are required")
    if len(sequences) != len(targets):
        raise ValueError("Sequences and targets must have the same length")
    feature_dimensions = {sequence.shape[1] for sequence in sequences}
    if len(feature_dimensions) != 1:
        raise ValueError("All sequences must have the same feature dimension")
    if task not in {"classification", "multiclass", "regression"}:
        raise ValueError(f"Unknown sequence task: {task}")

    torch.manual_seed(seed)
    feature_count = sequences[0].shape[1]
    total = torch.zeros(feature_count, dtype=torch.float64)
    squares = torch.zeros_like(total)
    observation_count = 0
    for sequence in sequences:
        values = sequence.double()
        total += values.sum(dim=0)
        squares += (values * values).sum(dim=0)
        observation_count += len(sequence)
    mean = (total / observation_count).float()
    variance = squares / observation_count - (total / observation_count) ** 2
    std = torch.sqrt(variance.clamp_min(0)).float().clamp_min(1e-6)
    input_dim = feature_count

    output_dim = num_classes if task == "multiclass" else 1
    if name == "gru":
        model: nn.Module = GRUSequenceModel(input_dim, output_dim=output_dim)
    elif name == "transformer":
        model = TransformerSequenceModel(input_dim, output_dim=output_dim)
    else:
        raise ValueError(f"Unknown sequence model: {name}")

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    if task == "classification":
        loss_function: nn.Module = nn.BCEWithLogitsLoss()
        target_tensor = torch.as_tensor(targets, dtype=torch.float32)
    elif task == "multiclass":
        loss_function = nn.CrossEntropyLoss()
        target_tensor = torch.as_tensor(targets, dtype=torch.long)
    else:
        loss_function = nn.MSELoss()
        target_mean = float(np.mean(targets))
        target_std = max(float(np.std(targets)), 1e-6)
        target_tensor = torch.as_tensor(
            (np.asarray(targets) - target_mean) / target_std,
            dtype=torch.float32,
        )
    if task != "regression":
        target_mean = 0.0
        target_std = 1.0
    generator = torch.Generator().manual_seed(seed)

    for _ in range(epochs):
        order = torch.randperm(len(sequences), generator=generator)
        model.train()
        for start in range(0, len(order), batch_size):
            indices = order[start : start + batch_size].tolist()
            padded, lengths, mask = padded_batch(
                [(sequences[index] - mean) / std for index in indices]
            )
            if name == "gru":
                predictions = model(padded, lengths)
            else:
                predictions = model(padded, lengths, mask)
            loss = loss_function(predictions, target_tensor[indices])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    return TrainedSequenceModel(
        model=model,
        name=name,
        task=task,
        mean=mean,
        std=std,
        target_mean=target_mean,
        target_std=target_std,
    )


def predict_sequence_model(
    trained: TrainedSequenceModel,
    sequences: list[torch.Tensor],
    batch_size: int = 256,
) -> np.ndarray:
    if not sequences:
        return np.empty(0, dtype=float)
    predictions = []
    trained.model.eval()

    with torch.no_grad():
        for start in range(0, len(sequences), batch_size):
            normalized = _normalize_sequences(
                sequences[start : start + batch_size], trained.mean, trained.std
            )
            padded, lengths, mask = padded_batch(normalized)
            if trained.name == "gru":
                output = trained.model(padded, lengths)
            else:
                output = trained.model(padded, lengths, mask)
            if trained.task == "classification":
                output = torch.sigmoid(output)
            elif trained.task == "multiclass":
                output = torch.argmax(output, dim=1)
            elif trained.task == "regression":
                output = output * trained.target_std + trained.target_mean
            predictions.append(output.cpu())

    return torch.cat(predictions).numpy()


def _sequence_dataset_collate(
    batch: list[tuple[torch.Tensor, float]],
) -> tuple[list[torch.Tensor], torch.Tensor]:
    return [item[0] for item in batch], torch.tensor([item[1] for item in batch])


def train_sequence_dataset(
    dataset: SequenceDataset,
    name: str,
    task: str,
    epochs: int = 10,
    batch_size: int = 64,
    learning_rate: float = 1e-3,
    seed: int = 42,
    num_classes: int = 2,
) -> TrainedSequenceModel:
    dataset_size = dataset.__len__()
    if dataset_size == 0:
        raise ValueError("Non-empty sequence dataset is required")
    first, _ = dataset[0]
    feature_count = first.shape[1]
    total = torch.zeros(feature_count, dtype=torch.float64)
    squares = torch.zeros_like(total)
    observation_count = 0
    targets = []
    for index in range(dataset_size):
        sequence, target = dataset[index]
        if len(sequence) == 0 or sequence.shape[1] != feature_count:
            raise ValueError("Sequence shapes are inconsistent")
        values = sequence.double()
        total += values.sum(0)
        squares += (values * values).sum(0)
        observation_count += len(values)
        targets.append(target)
    mean = (total / observation_count).float()
    variance = squares / observation_count - (total / observation_count) ** 2
    std = torch.sqrt(variance.clamp_min(0)).float().clamp_min(1e-6)
    target_values = np.asarray(targets)
    if task == "regression":
        target_mean = float(target_values.mean())
        target_std = max(float(target_values.std()), 1e-6)
    else:
        target_mean = 0.0
        target_std = 1.0
    output_dim = num_classes if task == "multiclass" else 1
    if name == "gru":
        model: nn.Module = GRUSequenceModel(feature_count, output_dim=output_dim)
    elif name == "transformer":
        model = TransformerSequenceModel(feature_count, output_dim=output_dim)
    else:
        raise ValueError(f"Unknown sequence model: {name}")
    if task == "classification":
        loss_function: nn.Module = nn.BCEWithLogitsLoss()
    elif task == "multiclass":
        loss_function = nn.CrossEntropyLoss()
    elif task == "regression":
        loss_function = nn.MSELoss()
    else:
        raise ValueError(f"Unknown sequence task: {task}")
    torch.manual_seed(seed)
    generator = torch.Generator().manual_seed(seed)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    for _ in range(epochs):
        model.train()
        order = torch.randperm(dataset_size, generator=generator)
        for start in range(0, dataset_size, batch_size):
            batch = [dataset[int(index)] for index in order[start : start + batch_size]]
            sequences, batch_targets = _sequence_dataset_collate(batch)
            normalized = [(sequence - mean) / std for sequence in sequences]
            padded, lengths, mask = padded_batch(normalized)
            predictions = (
                model(padded, lengths)
                if name == "gru"
                else model(padded, lengths, mask)
            )
            if task == "classification":
                expected = batch_targets.float()
            elif task == "multiclass":
                expected = batch_targets.long()
            else:
                expected = (batch_targets.float() - target_mean) / target_std
            loss = loss_function(predictions, expected)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    return TrainedSequenceModel(
        model=model,
        name=name,
        task=task,
        mean=mean,
        std=std,
        target_mean=target_mean,
        target_std=target_std,
    )


def predict_sequence_dataset(
    trained: TrainedSequenceModel,
    dataset: SequenceDataset,
    batch_size: int = 256,
) -> tuple[np.ndarray, np.ndarray]:
    predictions = []
    targets = []
    trained.model.eval()
    with torch.no_grad():
        for start in range(0, len(dataset), batch_size):
            batch = [
                dataset[index]
                for index in range(start, min(start + batch_size, len(dataset)))
            ]
            sequences, batch_targets = _sequence_dataset_collate(batch)
            normalized = [
                (sequence - trained.mean) / trained.std for sequence in sequences
            ]
            padded, lengths, mask = padded_batch(normalized)
            output = (
                trained.model(padded, lengths)
                if trained.name == "gru"
                else trained.model(padded, lengths, mask)
            )
            if trained.task == "classification":
                output = torch.sigmoid(output)
            elif trained.task == "multiclass":
                output = torch.argmax(output, dim=1)
            else:
                output = output * trained.target_std + trained.target_mean
            predictions.append(output.cpu())
            targets.append(batch_targets)
    return torch.cat(predictions).numpy(), torch.cat(targets).numpy()
