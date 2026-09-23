from typing import Optional, Tuple

import numpy as np
import torch
from sc2_datasets.replay_data.sc2_replay_data import SC2ReplayData, ToonPlayerDesc
from sc2_datasets.replay_parser.tracker_events.events.player_stats.player_stats import (
    PlayerStats,
    Stats,
)
from sc2_datasets.replay_parser.tracker_events.events.unit_died import UnitDied
from sc2_datasets.replay_parser.tracker_events.events.upgrade import Upgrade

# Race encoding: map race name to float
RACE_MAP = {"Zerg": 0.0, "Protoss": 1.0, "Terran": 2.0}

SORTED_PLAYERSTATS_KEYS = sorted(Stats.__dataclass_fields__.keys())
META_FEATURE_NAMES = ["supplyCappedPercent"]


def _get_stats_values(stats_obj) -> list:
    """Extract float values from a Stats object."""

    stats_dict = stats_obj.__dict__
    return_value = []
    for key in SORTED_PLAYERSTATS_KEYS:
        value = stats_dict[key]
        return_value.append(float(value))

    return return_value


def _get_player_stats_timeseries(
    sc2_replay: SC2ReplayData,
    player_id: int,
) -> list[PlayerStats]:
    """
    Collect all PlayerStats events for a given player, sorted by loop.

    Parameters
    ----------
    sc2_replay : SC2ReplayData
        Parsed SC2 replay data containing tracker events.
    player_id : int
        ID of the player to extract stats for.

    Returns
    -------
    list[PlayerStats]
        List of PlayerStats events for the specified player, sorted by game loop.
    """
    events = []
    for event in sc2_replay.trackerEvents:
        if type(event).__name__ == "PlayerStats" and event.playerId == player_id:
            events.append(event)
    events.sort(key=lambda e: e.loop)
    return events


def _temporal_snapshot(
    events: list[PlayerStats],
    start_frac: float,
    end_frac: float,
) -> np.ndarray:
    """
    Extract a temporal snapshot of player stats averaged over a fractional time window of the game.

    Parameters
    ----------
    events : list[PlayerStats]
        List of PlayerStats events.
    start_frac : float
        Fractional start time of the window.
    end_frac : float
        Fractional end time of the window.

    Returns
    -------
    np.ndarray
        Averaged stats values over the specified time window, or zeros if no events in window.
    """

    n = len(events)
    start_idx = int(start_frac * n)
    end_idx = max(int(end_frac * n), start_idx + 1)

    window = events[start_idx:end_idx]

    values = [_get_stats_values(stats_obj=e.stats) for e in window]
    return np.mean(values, axis=0)


def _count_units_born(sc2_replay: SC2ReplayData, player_id: int) -> int:
    """
    Counts the UnitBorn events for a given player.

    Parameters
    ----------
    sc2_replay : SC2ReplayData
        Parsed SC2 replay data containing tracker events.
    player_id : int
        ID of the player to count units for.

    Returns
    -------
    int
        Number of units born for the specified player.
    """

    count = 0
    for event in sc2_replay.trackerEvents:
        if type(event).__name__ == "UnitBorn":
            if hasattr(event, "controlPlayerId") and event.controlPlayerId == player_id:
                count += 1
    return count


def _count_units_died_by_opponent(sc2_replay: SC2ReplayData, player_id: int) -> int:
    """
    Counts the UnitDied events where the opponent killed this player's units.

    Parameters
    ----------
    sc2_replay : SC2ReplayData
        Parsed SC2 replay data containing tracker events.
    player_id : int
        ID of the player to count units for.

    Returns
    -------
    int
        Number of units killed by the opponent for the specified player.
    """
    count = 0
    for event in sc2_replay.trackerEvents:
        if isinstance(event, UnitDied):
            # if type(event).__name__ == "UnitDied":
            if hasattr(event, "killerPlayerId") and event.killerPlayerId == player_id:
                # This player KILLED an enemy unit (good for this player)
                count += 1
    return count


def _count_upgrades(sc2_replay: SC2ReplayData, player_id: int) -> int:
    """
    Counts the Upgrade events for a given player.

    Parameters
    ----------
    sc2_replay : SC2ReplayData
        Parsed SC2 replay data containing tracker events.
    player_id : int
        ID of the player to count upgrades for.

    Returns
    -------
    int
        Number of upgrades for the specified player.
    """

    count = 0
    for event in sc2_replay.trackerEvents:
        if isinstance(event, Upgrade) and event.playerId == player_id:
            # if type(event).__name__ == "Upgrade" and event.playerId == player_id:
            count += 1
    return count


def _get_player_info(
    sc2_replay: SC2ReplayData,
    player_id: int,
) -> ToonPlayerDesc | None:
    """
    Get the ToonPlayerInfo for a specific player from the replay data.

    Parameters
    ----------
    sc2_replay : SC2ReplayData
        Parsed SC2 replay data containing player information.
    player_id : int
        ID of the player to retrieve info for.

    Returns
    -------
    ToonPlayerDesc | None
        ToonPlayerDesc object containing player information, or None if not found.
    """

    for toon_desc in sc2_replay.toonPlayerDescMap:
        if toon_desc.toon_player_info.playerID == player_id:
            return toon_desc.toon_player_info
    return None


def _get_outcome(sc2_replay: SC2ReplayData) -> int | None:
    """
    Get the game outcome for player 1 (0=loss, 1=win), or None to skip if undecided/draw.

    Parameters
    ----------
    sc2_replay : SC2ReplayData
        Parsed SC2 replay data containing player information.

    Returns
    -------
    Optional[int]
        Game outcome for player 1 (0=loss, 1=win), or None to skip if undecided/draw.
    """

    result_map = {"Loss": 0, "Win": 1, "Victory": 1, "Defeat": 0}
    skip_results = {"Undecided", "Draw", "Tie"}

    for toon_desc in sc2_replay.toonPlayerDescMap:
        info = toon_desc.toon_player_info
        if info.result in skip_results:
            return None

    # Get player 1's result
    for toon_desc in sc2_replay.toonPlayerDescMap:
        info = toon_desc.toon_player_info
        if str(info.playerID) == "1":
            return result_map.get(info.result)

    return None


def prepare_player_features(
    sc2_replay: SC2ReplayData,
    player_id: int,
    game_duration: float,
) -> Optional[np.ndarray]:

    # 1. Temporal economy snapshots
    events = _get_player_stats_timeseries(
        sc2_replay=sc2_replay,
        player_id=player_id,
    )

    # Skip replays without economy data
    if not events:
        return None

    early_stats = _temporal_snapshot(
        events=events,
        start_frac=0.0,
        end_frac=0.33,
    )
    mid_stats = _temporal_snapshot(
        events=events,
        start_frac=0.33,
        end_frac=0.67,
    )
    late_stats = _temporal_snapshot(
        events=events,
        start_frac=0.67,
        end_frac=1.0,
    )

    # 2. Final economy state:
    final_stats = _get_stats_values(stats_obj=events[-1].stats)
    final_stats = np.array(final_stats, dtype=np.float32)

    # 3. Economy rate of change (late - early):
    econ_delta = late_stats - early_stats

    # 4. Player meta stats:
    player_info = _get_player_info(sc2_replay=sc2_replay, player_id=player_id)
    if player_info is None:
        return None

    meta_features = np.array(
        [
            # float(player_info.APM),
            # float(player_info.MMR) if player_info.MMR else 0.0,
            # float(player_info.SQ) if player_info.SQ else 0.0,
            float(player_info.supplyCappedPercent)
            if player_info.supplyCappedPercent
            else 0.0,
        ],
        dtype=np.float32,
    )

    # 5. Unit activity:
    # units_born = float(
    #     _count_units_born(
    #         sc2_replay=sc2_replay,
    #         player_id=player_id,
    #     )
    # )
    # units_killed = float(
    #     _count_units_died_by_opponent(
    #         sc2_replay=sc2_replay,
    #         player_id=player_id,
    #     )
    # )

    # # 6. Upgrades:
    # upgrade_count = float(
    #     _count_upgrades(
    #         sc2_replay=sc2_replay,
    #         player_id=player_id,
    #     )
    # )

    # Concatenate all features for this player
    player_feat = np.concatenate(
        [
            early_stats,  # 39
            mid_stats,  # 39
            late_stats,  # 39
            final_stats,  # 39
            econ_delta,  # 39
            meta_features,  # 1
            # [units_born],  # 1
            # [units_killed],  # 1
            # [upgrade_count],  # 1
            # duration,  # 1
        ]
    )

    return player_feat


def rich_transform(sc2_replay: SC2ReplayData) -> Tuple[torch.Tensor, int] | None:
    """Extract rich features from an SC2 replay.

    Returns:
        Tuple of (features_tensor [2, N_features], label) or None to skip.
    """
    # Get outcome
    label = _get_outcome(sc2_replay=sc2_replay)
    if label is None:
        return None

    # Game duration in loops
    try:
        game_duration = float(sc2_replay.header.elapsedGameLoops)
        # Games shorter than 3 minutes (4032 loops):
        if game_duration < 4032:
            return None
    except (AttributeError, ValueError, TypeError):
        return None

    player_features = []

    for player_id in [1, 2]:
        player_feat = prepare_player_features(
            sc2_replay=sc2_replay,
            player_id=player_id,
            game_duration=game_duration,
        )
        if player_feat is None:
            return None

        player_features.append(player_feat)

    features = torch.tensor(np.stack(player_features), dtype=torch.float32)

    return features, label
