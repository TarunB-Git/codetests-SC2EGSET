from __future__ import annotations

import random
import time
from pathlib import Path
from typing import Any, Callable

from latent_trainer.benchmarks.cache.dataset import ShardReplayDataset
from latent_trainer.benchmarks.common.data import player_samples_from_replay
from latent_trainer.benchmarks.data.source import ReplaySource


def _timed(name: str, count: int, operation: Callable[[], None]) -> dict[str, Any]:
    started = time.perf_counter()
    operation()
    elapsed = time.perf_counter() - started
    return {
        "name": name,
        "records": count,
        "seconds": elapsed,
        "records_per_second": count / elapsed if elapsed > 0 else None,
    }


def benchmark_loading(
    source: ReplaySource,
    count: int = 100,
    seed: int = 42,
    cache_manifest: Path | None = None,
) -> dict[str, Any]:
    bounded_count = min(count, len(source))
    rng = random.Random(seed)
    random_indices = [rng.randrange(len(source)) for _ in range(bounded_count)]

    def sequential() -> None:
        for index in range(bounded_count):
            source[index]

    def indexed() -> None:
        for index in random_indices:
            source[index]

    def transformed() -> None:
        for index in range(bounded_count):
            player_samples_from_replay(source[index].replay)

    results = [
        _timed("sequential_indexed_reads", bounded_count, sequential),
        _timed("random_indexed_reads", bounded_count, indexed),
        _timed("transform_on_access", bounded_count, transformed),
    ]
    if cache_manifest is not None:
        dataset = ShardReplayDataset(cache_manifest)
        cache_count = min(count, len(dataset))

        def cached() -> None:
            for index in range(cache_count):
                dataset[index]

        results.append(_timed("cached_shard_reads", cache_count, cached))
    return {
        "seed": seed,
        "source_fingerprint": source.identity.fingerprint,
        "benchmarks": results,
    }
