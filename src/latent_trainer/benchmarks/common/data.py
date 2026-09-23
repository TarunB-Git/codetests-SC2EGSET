import json
from dataclasses import dataclass, fields
from pathlib import Path

import torch
from sc2_datasets.replay_data.sc2_replay_data import SC2ReplayData
from sc2_datasets.replay_parser.tracker_events.events.player_stats.stats import Stats
from sc2_datasets.torch.datasets.sc2_dataset_single_json import SC2DatasetSingleJSON
from sc2_datasets.transforms.utils import (
    average_player_stats,
    filter_player_stats,
    select_outcome_1v1,
)

from latent_trainer.benchmarks.data.source import (
    InputFormat,
    ReplaySource,
    detect_input_format,
)

STAT_NAMES = tuple(field.name for field in fields(Stats))
GAME_LOOPS_PER_SECOND = 22.4


@dataclass(frozen=True)
class PlayerReplaySample:
    replay_id: str
    player_id: str
    player_toon: str
    opponent_id: str
    opponent_toon: str
    timestamp: str
    game_version: str
    map_name: str
    duration_loops: int
    race: str
    opponent_race: str
    mmr: float
    opponent_mmr: float
    outcome: int
    average: torch.Tensor
    sequence: torch.Tensor
    sequence_loops: torch.Tensor
    opponent_sequence: torch.Tensor
    opponent_sequence_loops: torch.Tensor


def _stats_tensor(events: list) -> tuple[torch.Tensor, torch.Tensor]:
    by_loop = {int(event.loop): event for event in events}
    ordered = [by_loop[loop] for loop in sorted(by_loop)]
    values = [
        [float(getattr(event.stats, name)) for name in STAT_NAMES] for event in ordered
    ]
    return (
        torch.tensor(values, dtype=torch.float32),
        torch.tensor([event.loop for event in ordered], dtype=torch.long),
    )


def player_samples_from_replay(
    replay: SC2ReplayData,
    min_duration_loops: int = 0,
) -> list[PlayerReplaySample]:
    if replay.trackerEventsErr:
        return []
    if replay.header.elapsedGameLoops <= min_duration_loops:
        return []

    averaged = average_player_stats(replay)
    outcomes = select_outcome_1v1(replay)
    events = filter_player_stats(replay)
    descriptions = {
        str(description.toon_player_info.playerID): description
        for description in replay.toonPlayerDescMap
    }
    player_ids = sorted(averaged, key=int)

    if len(player_ids) != 2:
        return []
    if any(player_id not in descriptions for player_id in player_ids):
        return []
    if any(player_id not in outcomes for player_id in player_ids):
        return []
    if any(not events.get(player_id) for player_id in player_ids):
        return []

    sequence_by_player = {
        player_id: _stats_tensor(events[player_id]) for player_id in player_ids
    }
    samples = []

    for player_id in player_ids:
        opponent_id = next(value for value in player_ids if value != player_id)
        description = descriptions[player_id]
        opponent_description = descriptions[opponent_id]
        info = description.toon_player_info
        opponent_info = opponent_description.toon_player_info
        outcome = int(outcomes[player_id])

        if outcome not in (0, 1):
            continue

        sequence, sequence_loops = sequence_by_player[player_id]
        opponent_sequence, opponent_sequence_loops = sequence_by_player[opponent_id]
        average = torch.tensor(averaged[player_id], dtype=torch.float32)

        if average.shape != (len(STAT_NAMES),):
            continue

        samples.append(
            PlayerReplaySample(
                replay_id=str(replay.filepath),
                player_id=player_id,
                player_toon=description.toon,
                opponent_id=opponent_id,
                opponent_toon=opponent_description.toon,
                timestamp=replay.details.timeUTC,
                game_version=replay.metadata.gameVersion,
                map_name=replay.metadata.mapName,
                duration_loops=int(replay.header.elapsedGameLoops),
                race=info.race,
                opponent_race=opponent_info.race,
                mmr=float(info.MMR or 0),
                opponent_mmr=float(opponent_info.MMR or 0),
                outcome=outcome,
                average=average,
                sequence=sequence,
                sequence_loops=sequence_loops,
                opponent_sequence=opponent_sequence,
                opponent_sequence_loops=opponent_sequence_loops,
            )
        )

    return samples


def load_player_samples(
    json_path: Path,
    max_replays: int = 0,
    min_duration_loops: int = 0,
    offsets_path: Path | None = None,
    input_format: InputFormat = "auto",
    source_indices_path: Path | None = None,
) -> list[PlayerReplaySample]:
    if offsets_path is not None:
        samples: list[PlayerReplaySample] = []
        with ReplaySource(
            json_path,
            offsets_path,
            input_format=input_format,
            source_indices_path=source_indices_path,
        ) as source:
            limit = len(source) if max_replays <= 0 else min(max_replays, len(source))
            for index in range(limit):
                samples.extend(
                    player_samples_from_replay(
                        source[index].replay,
                        min_duration_loops=min_duration_loops,
                    )
                )
        return samples
    detected_format = detect_input_format(json_path)
    selected_format = detected_format if input_format == "auto" else input_format
    if selected_format != detected_format:
        raise ValueError(
            f"Requested format {selected_format} does not match detected "
            f"{detected_format}"
        )
    if selected_format == "jsonl":
        samples = []
        replay_index = 0
        with json_path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                stripped = line.strip()
                if not stripped:
                    continue
                if max_replays > 0 and replay_index >= max_replays:
                    break
                loaded = json.loads(stripped)
                if not isinstance(loaded, dict):
                    raise ValueError(
                        f"JSONL record on line {line_number} is not an object"
                    )
                replay = SC2ReplayData.from_dict(
                    loaded,
                    replay_filepath=f"record-{replay_index}",
                )
                samples.extend(
                    player_samples_from_replay(
                        replay,
                        min_duration_loops=min_duration_loops,
                    )
                )
                replay_index += 1
        return samples
    dataset = SC2DatasetSingleJSON.from_json_path(json_path)
    limit = len(dataset) if max_replays <= 0 else min(max_replays, len(dataset))
    samples = []

    for index in range(limit):
        samples.extend(
            player_samples_from_replay(
                dataset[index], min_duration_loops=min_duration_loops
            )
        )

    return samples


def load_cached_player_samples(
    manifest_path: Path,
    max_replays: int = 0,
) -> list[PlayerReplaySample]:
    from latent_trainer.benchmarks.cache.dataset import ShardReplayDataset

    dataset = ShardReplayDataset(manifest_path)
    limit = len(dataset) if max_replays <= 0 else min(max_replays, len(dataset))
    result = []
    for index in range(limit):
        replay = dataset[index]
        for slot in range(2):
            opponent = 1 - slot
            result.append(
                PlayerReplaySample(
                    replay_id=replay["replay_id"],
                    player_id=replay["player_ids"][slot],
                    player_toon=replay["toon_ids"][slot],
                    opponent_id=replay["player_ids"][opponent],
                    opponent_toon=replay["toon_ids"][opponent],
                    timestamp=replay["timestamp"],
                    game_version=replay["game_version"],
                    map_name=replay["map_name"],
                    duration_loops=replay["duration_loops"],
                    race=replay["races"][slot],
                    opponent_race=replay["races"][opponent],
                    mmr=float(replay["raw_mmr"][slot]),
                    opponent_mmr=float(replay["raw_mmr"][opponent]),
                    outcome=int(replay["outcomes"][slot]),
                    average=replay["average"][slot],
                    sequence=replay["sequences"][slot],
                    sequence_loops=replay["sequence_loops"][slot],
                    opponent_sequence=replay["sequences"][opponent],
                    opponent_sequence_loops=replay["sequence_loops"][opponent],
                )
            )
    return result
