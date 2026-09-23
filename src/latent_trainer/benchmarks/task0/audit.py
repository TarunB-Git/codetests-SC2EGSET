from __future__ import annotations

import math
from collections import Counter
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np

from latent_trainer.benchmarks.common.data import player_samples_from_replay
from latent_trainer.benchmarks.common.provenance import software_provenance
from latent_trainer.benchmarks.data.source import ReplayReadError, ReplaySource


def _distribution(values: list[float]) -> dict[str, float | int | None]:
    if not values:
        return {
            "count": 0,
            "min": None,
            "max": None,
            "mean": None,
            "median": None,
            "p05": None,
            "p95": None,
        }
    array = np.asarray(values, dtype=float)
    return {
        "count": len(values),
        "min": float(array.min()),
        "max": float(array.max()),
        "mean": float(array.mean()),
        "median": float(np.median(array)),
        "p05": float(np.quantile(array, 0.05)),
        "p95": float(np.quantile(array, 0.95)),
    }


def _counter(counter: Counter[Any]) -> dict[str, int]:
    return {str(key): value for key, value in sorted(counter.items(), key=str)}


def audit_source(
    source: ReplaySource,
    output_path: Path | None = None,
    invalid_mmr_sentinels: tuple[float, ...] = (-36400.0,),
    nonpositive_mmr_invalid: bool = True,
    max_replays: int = 0,
) -> dict[str, Any]:
    limit = len(source) if max_replays <= 0 else min(max_replays, len(source))
    rejections: Counter[str] = Counter()
    races: Counter[str] = Counter()
    outcomes: Counter[str] = Counter()
    versions: Counter[str] = Counter()
    maps: Counter[str] = Counter()
    regions: Counter[str] = Counter()
    games_per_player: Counter[str] = Counter()
    source_indices: list[int] = []
    durations: list[float] = []
    ratings: list[float] = []
    timestamps: list[str] = []
    player_observations = 0
    two_player_games = 0
    ordinary_outcome_games = 0
    invalid_outcome_games = 0
    parser_flags: Counter[str] = Counter()
    mmr_zero = 0
    mmr_negative = 0
    mmr_missing = 0
    mmr_sentinel: Counter[str] = Counter()
    mmr_valid = 0
    transform_compatible = 0

    for index in range(limit):
        try:
            record = source[index]
        except ReplayReadError as error:
            rejections[error.code] += 1
            continue
        replay = record.replay
        source_indices.append(record.source_index)
        timestamps.append(str(replay.details.timeUTC))
        versions[str(replay.metadata.gameVersion)] += 1
        maps[str(replay.metadata.mapName)] += 1
        durations.append(float(replay.header.elapsedGameLoops))
        parser_flags["game_events"] += int(bool(replay.gameEventsErr))
        parser_flags["message_events"] += int(bool(replay.messageEventsErr))
        parser_flags["tracker_events"] += int(bool(replay.trackerEventsErr))
        if len(replay.toonPlayerDescMap) == 2:
            two_player_games += 1
        else:
            rejections["not_two_players"] += 1
        game_outcomes = []
        for description in replay.toonPlayerDescMap:
            player_observations += 1
            info = description.toon_player_info
            toon = str(description.toon)
            if toon:
                games_per_player[toon] += 1
            races[str(info.race)] += 1
            outcomes[str(info.result)] += 1
            game_outcomes.append(str(info.result))
            regions[str(info.region)] += 1
            raw_mmr = getattr(info, "MMR", None)
            if raw_mmr is None or (isinstance(raw_mmr, float) and math.isnan(raw_mmr)):
                mmr_missing += 1
                continue
            rating = float(raw_mmr)
            ratings.append(rating)
            mmr_zero += int(rating == 0)
            mmr_negative += int(rating < 0)
            if rating in invalid_mmr_sentinels:
                mmr_sentinel[str(rating)] += 1
            invalid = rating in invalid_mmr_sentinels or (
                nonpositive_mmr_invalid and rating <= 0
            )
            mmr_valid += int(not invalid)
        if sorted(game_outcomes) == ["Loss", "Win"]:
            ordinary_outcome_games += 1
        else:
            invalid_outcome_games += 1
            rejections["invalid_outcome"] += 1
        try:
            compatible = len(player_samples_from_replay(replay)) == 2
        except (KeyError, TypeError, ValueError, RuntimeError):
            compatible = False
            rejections["transform_failure"] += 1
        transform_compatible += int(compatible)

    game_counts = list(games_per_player.values())
    recurrence = Counter(game_counts)
    gaps = np.diff(source_indices).tolist() if len(source_indices) > 1 else []
    audit = {
        "source": asdict(source.identity),
        "software": software_provenance(),
        "scope": {
            "requested_records": limit,
            "complete_source": limit == len(source),
            "representativeness_claimed": False,
        },
        "replay_count": limit,
        "successfully_parsed_replays": len(source_indices),
        "player_observation_count": player_observations,
        "unique_persistent_player_count": len(games_per_player),
        "games_per_player": _distribution([float(value) for value in game_counts]),
        "recurrence_distribution": _counter(recurrence),
        "two_player_games": two_player_games,
        "ordinary_win_loss_games": ordinary_outcome_games,
        "invalid_outcome_games": invalid_outcome_games,
        "parser_error_flags": _counter(parser_flags),
        "transform_compatible_games": transform_compatible,
        "rejection_counts": _counter(rejections),
        "race_distribution": _counter(races),
        "result_distribution": _counter(outcomes),
        "mmr": {
            "raw_distribution": _distribution(ratings),
            "valid_for_modelling": mmr_valid,
            "missing": mmr_missing,
            "zero": mmr_zero,
            "negative": mmr_negative,
            "sentinels": _counter(mmr_sentinel),
            "configured_sentinels": list(invalid_mmr_sentinels),
            "nonpositive_invalid": nonpositive_mmr_invalid,
        },
        "duration_loops": _distribution(durations),
        "short_game_counts": {
            "under_60_seconds": sum(value < 22.4 * 60 for value in durations),
            "at_most_12000_loops": sum(value <= 12000 for value in durations),
        },
        "timestamp_range": {
            "min": min(timestamps) if timestamps else None,
            "max": max(timestamps) if timestamps else None,
        },
        "game_versions": _counter(versions),
        "maps": _counter(maps),
        "regions": _counter(regions),
        "source_index_coverage": {
            "count": len(source_indices),
            "min": min(source_indices) if source_indices else None,
            "max": max(source_indices) if source_indices else None,
            "gap_distribution": _counter(Counter(gaps)),
        },
    }
    if output_path is not None:
        if output_path.exists():
            raise FileExistsError(output_path)
        import json

        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(
            json.dumps(audit, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
    return audit
