from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any

import lightning as pl
import numpy as np
import torch
from lightning.pytorch.callbacks import ModelCheckpoint
from torch.utils.data import DataLoader

from latent_trainer.benchmarks.cache.collate import seed_worker
from latent_trainer.benchmarks.cache.dataset import ShardReplayDataset
from latent_trainer.benchmarks.data.schema import canonical_feature_names
from latent_trainer.benchmarks.task4.data import (
    GuidedVAEShardAdapter,
    fit_streaming_normalization,
    load_compatible_split,
)
from latent_trainer.models.lightning.lit_guided_vae import LitGuidedVAE


def train_guided_vae(
    cache_manifest_path: Path,
    split_path: Path,
    output_dir: Path,
    epochs: int = 20,
    batch_size: int = 128,
    latent_dim: int = 16,
    supervised_dim: int = 4,
    hidden_dims: tuple[int, ...] = (64, 128, 256),
    learning_rate: float = 1e-4,
    seed: int = 42,
    workers: int = 0,
    accelerator: str = "auto",
) -> dict[str, Any]:
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError("Task 4 output directory is not empty")
    output_dir.mkdir(parents=True, exist_ok=True)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    pl.seed_everything(seed, workers=True)
    dataset = ShardReplayDataset(cache_manifest_path)
    split = load_compatible_split(dataset, split_path)
    train_indices = split.splits["train"]["dataset_indices"]
    validation_indices = split.splits["validation"]["dataset_indices"]
    normalization = fit_streaming_normalization(dataset, train_indices)
    normalization_path = output_dir / "normalization.json"
    normalization.save(normalization_path)
    train_dataset = GuidedVAEShardAdapter(dataset, train_indices, normalization)
    validation_dataset = GuidedVAEShardAdapter(
        dataset, validation_indices, normalization
    )
    generator = torch.Generator().manual_seed(seed)
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=workers,
        worker_init_fn=seed_worker,
        generator=generator,
    )
    validation_loader = DataLoader(
        validation_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=workers,
        worker_init_fn=seed_worker,
    )
    compatibility: dict[str, str | int] = {
        "cache_fingerprint": dataset.manifest.fingerprint(),
        "split_fingerprint": split.fingerprint(),
        "source_fingerprint": str(dataset.manifest.source_identity["fingerprint"]),
        "cache_schema_version": dataset.manifest.cache_schema_version,
        "feature_count": len(canonical_feature_names()),
        "normalization_fingerprint": normalization.fingerprint(),
    }
    model = LitGuidedVAE(
        input_dim=len(canonical_feature_names()),
        encoder_hidden_dims=list(hidden_dims),
        supervised_dim=supervised_dim,
        latent_dim=latent_dim,
        learning_rate=learning_rate,
        mean=normalization.mean,
        std=normalization.std,
        compatibility_metadata=compatibility,
    )
    checkpoint = ModelCheckpoint(
        dirpath=output_dir,
        filename="guided-vae",
        monitor="val_loss",
        mode="min",
        save_top_k=1,
        auto_insert_metric_name=False,
    )
    trainer = pl.Trainer(
        max_epochs=epochs,
        deterministic=True,
        logger=False,
        callbacks=[checkpoint],
        enable_checkpointing=True,
        accelerator=accelerator,
    )
    trainer.fit(model, train_loader, validation_loader)
    result = {
        "checkpoint": checkpoint.best_model_path,
        "normalization": str(normalization_path),
        "compatibility": compatibility,
        "seed": seed,
        "epochs": epochs,
        "scientific_results_claimed": False,
    }
    (output_dir / "training.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return result
