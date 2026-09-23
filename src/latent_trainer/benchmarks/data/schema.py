from dataclasses import dataclass, fields
from enum import StrEnum

from sc2_datasets.replay_parser.tracker_events.events.player_stats.stats import Stats


class FeatureRole(StrEnum):
    MODEL_INPUT = "model_input"
    TARGET = "target"
    SPLIT_ONLY = "split_only_metadata"
    EVALUATION_ONLY = "evaluation_only_metadata"
    PROVENANCE = "provenance"
    REJECTED_PRIVATE = "rejected_private_field"


@dataclass(frozen=True)
class FeatureSpec:
    name: str
    role: FeatureRole
    group: str


def canonical_feature_names() -> tuple[str, ...]:
    names = tuple(field.name for field in fields(Stats))
    if len(names) != 39 or len(set(names)) != len(names):
        raise RuntimeError("Stats must define 39 unique canonical features")
    return names


FEATURE_REGISTRY = (
    *(
        FeatureSpec(name, FeatureRole.MODEL_INPUT, "gameplay")
        for name in canonical_feature_names()
    ),
    FeatureSpec("sequence_loops", FeatureRole.EVALUATION_ONLY, "sequence_timing"),
    FeatureSpec("race", FeatureRole.MODEL_INPUT, "controlled_race_ablation"),
    FeatureSpec("opponent_race", FeatureRole.MODEL_INPUT, "controlled_race_ablation"),
    FeatureSpec("mmr", FeatureRole.TARGET, "skill"),
    FeatureSpec("opponent_mmr", FeatureRole.EVALUATION_ONLY, "skill"),
    FeatureSpec("outcome", FeatureRole.TARGET, "outcome"),
    FeatureSpec("league", FeatureRole.REJECTED_PRIVATE, "skill_metadata"),
    FeatureSpec("rank", FeatureRole.REJECTED_PRIVATE, "skill_metadata"),
    FeatureSpec("player_toon", FeatureRole.SPLIT_ONLY, "identity"),
    FeatureSpec("opponent_toon", FeatureRole.SPLIT_ONLY, "identity"),
    FeatureSpec("nickname", FeatureRole.REJECTED_PRIVATE, "identity"),
    FeatureSpec("clan_tag", FeatureRole.REJECTED_PRIVATE, "identity"),
    FeatureSpec("local_record_index", FeatureRole.PROVENANCE, "source"),
    FeatureSpec("source_index", FeatureRole.PROVENANCE, "source"),
    FeatureSpec("replay_id", FeatureRole.PROVENANCE, "source"),
    FeatureSpec("timestamp", FeatureRole.SPLIT_ONLY, "time"),
    FeatureSpec("game_version", FeatureRole.SPLIT_ONLY, "domain"),
    FeatureSpec("map_name", FeatureRole.EVALUATION_ONLY, "domain"),
    FeatureSpec("duration_loops", FeatureRole.EVALUATION_ONLY, "duration"),
)


def fields_for_role(role: FeatureRole) -> tuple[str, ...]:
    return tuple(spec.name for spec in FEATURE_REGISTRY if spec.role == role)


def metadata_blind_feature_names(include_race: bool = False) -> tuple[str, ...]:
    names = canonical_feature_names()
    return names + (("race", "opponent_race") if include_race else ())


def validate_encoder_fields(names: tuple[str, ...] | list[str]) -> None:
    allowed = set(canonical_feature_names())
    unexpected = set(names) - allowed
    if unexpected:
        raise ValueError(
            f"Task 4 encoder fields are not gameplay inputs: {sorted(unexpected)}"
        )
