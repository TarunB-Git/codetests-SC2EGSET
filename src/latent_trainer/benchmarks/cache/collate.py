from __future__ import annotations

import random
from typing import Any

import numpy as np
import torch
from torch.nn.utils.rnn import pack_padded_sequence, pad_sequence


def seed_worker(worker_id: int) -> None:
    seed = torch.initial_seed() % (2**32)
    np.random.seed(seed)
    random.seed(seed)


def collate_replays(samples: list[dict[str, Any]]) -> dict[str, Any]:
    player_sequences = [
        sequence for sample in samples for sequence in sample["sequences"]
    ]
    if any(len(sequence) == 0 for sequence in player_sequences):
        raise ValueError("Sequence batch contains an empty prefix")
    lengths = torch.tensor([len(sequence) for sequence in player_sequences])
    padded = pad_sequence(player_sequences, batch_first=True)
    steps = torch.arange(padded.shape[1]).unsqueeze(0)
    padding_mask = steps >= lengths.unsqueeze(1)
    return {
        "average": torch.stack([sample["average"] for sample in samples]),
        "sequence": padded,
        "sequence_lengths": lengths,
        "padding_mask": padding_mask,
        "outcomes": torch.stack([sample["outcomes"] for sample in samples]),
        "raw_mmr": torch.stack([sample["raw_mmr"] for sample in samples]),
        "mmr_valid": torch.stack([sample["mmr_valid"] for sample in samples]),
        "replay_ids": [sample["replay_id"] for sample in samples],
        "source_indices": [sample["source_index"] for sample in samples],
        "metadata": samples,
    }


def packed_sequences(batch: dict[str, Any]) -> torch.nn.utils.rnn.PackedSequence:
    return pack_padded_sequence(
        batch["sequence"],
        batch["sequence_lengths"].cpu(),
        batch_first=True,
        enforce_sorted=False,
    )
