from dataclasses import dataclass

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


def make_classifier(
    name: str,
    seed: int = 42,
    calibrated: bool = False,
    historical: bool = False,
) -> BaseEstimator:
    if name == "logistic":
        estimator = LogisticRegression(C=10, random_state=seed)
    elif name == "svm":
        estimator = SVC(
            kernel="rbf",
            C=10,
            gamma="auto",
            probability=True,
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
        estimator = CalibratedClassifierCV(estimator, method="sigmoid", cv=3)
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

        return XGBClassifier(
            objective="multi:softprob",
            num_class=class_count,
            learning_rate=0.2,
            max_depth=5,
            n_estimators=100,
            eval_metric="mlogloss",
            n_jobs=1,
            random_state=seed,
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
    return 1.0 / (1.0 + np.exp(-scores))


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
    if task not in {"classification", "multiclass", "regression"}:
        raise ValueError(f"Unknown sequence task: {task}")

    torch.manual_seed(seed)
    combined = torch.cat(sequences, dim=0).float()
    mean = combined.mean(dim=0)
    std = combined.std(dim=0).clamp_min(1e-6)
    normalized = _normalize_sequences(sequences, mean, std)
    input_dim = normalized[0].shape[1]

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
        order = torch.randperm(len(normalized), generator=generator)
        model.train()
        for start in range(0, len(order), batch_size):
            indices = order[start : start + batch_size].tolist()
            padded, lengths, mask = padded_batch(
                [normalized[index] for index in indices]
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
    normalized = _normalize_sequences(sequences, trained.mean, trained.std)
    predictions = []
    trained.model.eval()

    with torch.no_grad():
        for start in range(0, len(normalized), batch_size):
            padded, lengths, mask = padded_batch(normalized[start : start + batch_size])
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
