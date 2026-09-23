from dataclasses import fields

import torch
from sc2_datasets.replay_data.sc2_replay_data import SC2ReplayData
from sc2_datasets.replay_parser.tracker_events.events.player_stats.stats import Stats
from sc2_datasets.transforms.utils import average_player_stats, select_outcome_1v1

HISTORICAL_MIN_DURATION_LOOPS = 12000


def economy_average_players_vs_outcomes(
    sc2_replay: SC2ReplayData,
) -> tuple[torch.Tensor, torch.Tensor]:
    if sc2_replay.trackerEventsErr:
        raise ValueError("PlayerStats are unavailable after a tracker-event error")
    averaged = average_player_stats(sc2_replay)
    outcomes = select_outcome_1v1(sc2_replay)
    player_ids = sorted(averaged, key=int)

    if len(player_ids) != 2:
        raise ValueError(f"Expected two players, got {player_ids!r}")
    if set(player_ids) != set(outcomes):
        raise ValueError(
            f"Feature player IDs {player_ids!r} do not match outcome player IDs "
            f"{sorted(outcomes, key=int)!r}"
        )

    features = torch.tensor(
        [averaged[player_id] for player_id in player_ids], dtype=torch.float32
    )
    targets = torch.tensor(
        [outcomes[player_id] for player_id in player_ids], dtype=torch.long
    )
    expected_shape = (len(player_ids), len(fields(Stats)))

    if tuple(features.shape) != expected_shape:
        raise ValueError(
            f"Expected feature shape {expected_shape}, got {tuple(features.shape)}"
        )
    if tuple(targets.shape) != (len(player_ids),):
        raise ValueError(
            f"Expected target shape {(len(player_ids),)}, got {tuple(targets.shape)}"
        )
    if not torch.all((targets == 0) | (targets == 1)):
        raise ValueError(f"Expected binary outcomes, got {targets.tolist()!r}")

    return features, targets


def historical_economy_average_players_vs_outcomes(
    sc2_replay: SC2ReplayData,
) -> tuple[torch.Tensor, torch.Tensor] | None:
    if sc2_replay.header.elapsedGameLoops <= HISTORICAL_MIN_DURATION_LOOPS:
        return None
    return economy_average_players_vs_outcomes(sc2_replay)
