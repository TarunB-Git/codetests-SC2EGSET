from dataclasses import dataclass
from enum import StrEnum

from latent_trainer.benchmarks.data.schema import canonical_feature_names

CACHE_SCHEMA_VERSION = 1
EXTRACTOR_VERSION = 1


class RejectionCode(StrEnum):
    MALFORMED_JSON = "malformed_json"
    SCHEMA_FAILURE = "schema_failure"
    PARSER_FAILURE = "parser_failure"
    TRACKER_ERROR = "tracker_error"
    NOT_TWO_PLAYERS = "not_two_players"
    INVALID_OUTCOME = "invalid_outcome"
    MISSING_PLAYER_STATS = "missing_player_stats"
    TRANSFORM_FAILURE = "transform_failure"


@dataclass(frozen=True)
class CacheCompatibility:
    schema_version: int
    feature_names: tuple[str, ...]
    source_fingerprint: str


def expected_compatibility(source_fingerprint: str) -> CacheCompatibility:
    return CacheCompatibility(
        schema_version=CACHE_SCHEMA_VERSION,
        feature_names=canonical_feature_names(),
        source_fingerprint=source_fingerprint,
    )
