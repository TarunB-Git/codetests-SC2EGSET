from __future__ import annotations

from pathlib import Path

import torch

from latent_trainer.benchmarks.cache.dataset import ShardReplayDataset
from latent_trainer.benchmarks.splits.manifest import SplitManifest
from latent_trainer.models.lightning.lit_guided_vae import LitGuidedVAE


def load_compatible_model(
    checkpoint_path: Path,
    dataset: ShardReplayDataset,
    split: SplitManifest,
    normalization_fingerprint: str,
) -> LitGuidedVAE:
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    compatibility = checkpoint.get("benchmark_compatibility", {})
    expected = {
        "cache_fingerprint": dataset.manifest.fingerprint(),
        "split_fingerprint": split.fingerprint(),
        "source_fingerprint": str(dataset.manifest.source_identity["fingerprint"]),
        "cache_schema_version": dataset.manifest.cache_schema_version,
        "feature_count": len(dataset.manifest.feature_names),
        "normalization_fingerprint": normalization_fingerprint,
    }
    if compatibility != expected:
        raise ValueError("Checkpoint cache or split compatibility metadata differs")
    model = LitGuidedVAE.load_from_checkpoint(checkpoint_path, map_location="cpu")
    model.eval()
    return model
