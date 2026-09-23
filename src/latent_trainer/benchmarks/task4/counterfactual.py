from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch

from latent_trainer.benchmarks.cache.dataset import ShardReplayDataset
from latent_trainer.benchmarks.data.schema import canonical_feature_names
from latent_trainer.benchmarks.task4.checkpoint import load_compatible_model
from latent_trainer.benchmarks.task4.data import (
    StreamingNormalization,
    load_compatible_split,
)
from latent_trainer.paths.feedback import compute_feedback
from latent_trainer.paths.strategies.linear import path_linear


def constrain_to_observed_range(
    decoded: torch.Tensor,
    minimum: torch.Tensor,
    maximum: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    invalid = (decoded < minimum) | (decoded > maximum) | ~torch.isfinite(decoded)
    finite = torch.nan_to_num(decoded, nan=0.0, posinf=0.0, neginf=0.0)
    constrained = torch.maximum(torch.minimum(finite, maximum), minimum)
    return constrained, invalid


def generate_counterfactual(
    cache_manifest_path: Path,
    split_path: Path,
    checkpoint_path: Path,
    normalization_path: Path,
    dataset_index: int,
    n_waypoints: int = 20,
    top_k: int = 10,
) -> dict[str, Any]:
    dataset = ShardReplayDataset(cache_manifest_path)
    split = load_compatible_split(dataset, split_path)
    normalization = StreamingNormalization.load(normalization_path)
    model = load_compatible_model(
        checkpoint_path, dataset, split, normalization.fingerprint()
    )
    sample = dataset[dataset_index]
    normalized = (sample["average"] - normalization.mean) / normalization.std
    with torch.no_grad():
        start_mu, _ = model.model.encode(normalized.unsqueeze(0))
    winner_latents = []
    with torch.no_grad():
        for index in split.splits["train"]["dataset_indices"]:
            candidate = dataset[index]
            candidate_normalized = (
                candidate["average"] - normalization.mean
            ) / normalization.std
            mu, _ = model.model.encode(candidate_normalized.unsqueeze(0))
            for slot in range(2):
                if int(candidate["outcomes"][slot]) == 1:
                    winner_latents.append(mu[0, slot].numpy())
    if not winner_latents:
        raise ValueError("Training split has no winning latent representations")
    target = np.mean(winner_latents, axis=0)
    start = start_mu[0, 0].numpy()
    opponent = start_mu[0, 1].detach()
    path = path_linear(start, target, n_waypoints=n_waypoints)
    normalized_min = (normalization.minimum - normalization.mean) / normalization.std
    normalized_max = (normalization.maximum - normalization.mean) / normalization.std

    def raw_decode(latent: torch.Tensor) -> torch.Tensor:
        return model.model.decode(latent)

    def constrained_decode(latent: torch.Tensor) -> torch.Tensor:
        decoded = raw_decode(latent)
        return constrain_to_observed_range(decoded, normalized_min, normalized_max)[0]

    def score(latent: torch.Tensor) -> torch.Tensor:
        repeated_opponent = opponent.unsqueeze(0).expand(len(latent), -1)
        pair = torch.stack([latent, repeated_opponent], dim=1)
        return model.model.cls(pair).squeeze(1)

    path_tensor = torch.tensor(path, dtype=torch.float32)
    with torch.no_grad():
        unconstrained = raw_decode(path_tensor)
        _, invalid = constrain_to_observed_range(
            unconstrained, normalized_min, normalized_max
        )
    feedback = compute_feedback(
        path,
        decode_fn=constrained_decode,
        score_fn=score,
        norm_mean=normalization.mean,
        norm_std=normalization.std,
        feature_names=list(canonical_feature_names()),
        top_k=top_k,
        method_name="linear_winner_centroid",
    )
    return {
        "source_replay": {
            "dataset_index": dataset_index,
            "local_record_index": sample["local_record_index"],
            "source_index": sample["source_index"],
            "replay_id": sample["replay_id"],
        },
        "method": "model-associated path to the training winner centroid",
        "causal_claim": False,
        "observed_features": dict(
            zip(canonical_feature_names(), sample["average"][0].tolist(), strict=True)
        ),
        "derived_feedback": {
            "full_path": feedback["raw"],
            "minimum_viable": feedback["minimum_viable"],
            "gain_weighted": feedback["gain_weighted"],
        },
        "plausibility": {
            "range_source": "training split observed minima and maxima",
            "invalid_decoded_values_before_constraint": int(invalid.sum()),
            "decoded_values": int(invalid.numel()),
            "constraints_applied": True,
        },
        "probabilities": np.asarray(feedback["_p_vals"]).reshape(-1).tolist(),
    }
