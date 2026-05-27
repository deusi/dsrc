from __future__ import annotations

from typing import Any, Iterable, Mapping

from src.envs.base_ctde_env import AVAction, AVActionMap, HeadwayBin, LanePreference, MergeMode, SpeedBin


SPEED_BINS: tuple[SpeedBin, ...] = (
    "slow",
    "nominal",
    "fast",
)
HEADWAY_BINS: tuple[HeadwayBin, ...] = (
    "normal",
    "larger",
    "largest",
)
LANE_PREFERENCES: tuple[LanePreference, ...] = (
    "keep",
    "prefer_left_if_safe",
    "prefer_right_if_safe",
)
MERGE_MODES: tuple[MergeMode, ...] = (
    "normal",
    "create_gap",
    "hold_lane",
)


def default_agent_ids(count: int) -> list[str]:
    """Create canonical AV identifiers for wrappers and tests."""
    return [f"av_{index}" for index in range(count)]


def tuple_observation_to_mapping(
    agent_ids: Iterable[str],
    observations: Iterable[Any],
) -> dict[str, Any]:
    """Convert tuple-based simulator observations into public AV-indexed mappings."""
    return {agent_id: observation for agent_id, observation in zip(agent_ids, observations, strict=True)}


def validate_action(action: Mapping[str, Any]) -> AVAction:
    """Validate one public AV action object."""
    required_fields = ("desired_speed_bin", "desired_headway_bin", "lane_preference", "merge_mode")
    for field in required_fields:
        if field not in action:
            raise ValueError(f"action is missing required field '{field}'")

    desired_speed_bin = action["desired_speed_bin"]
    desired_headway_bin = action["desired_headway_bin"]
    lane_preference = action["lane_preference"]
    merge_mode = action["merge_mode"]
    if desired_speed_bin not in SPEED_BINS:
        raise ValueError(f"unsupported desired_speed_bin '{desired_speed_bin}'")
    if desired_headway_bin not in HEADWAY_BINS:
        raise ValueError(f"unsupported desired_headway_bin '{desired_headway_bin}'")
    if lane_preference not in LANE_PREFERENCES:
        raise ValueError(f"unsupported lane_preference '{lane_preference}'")
    if merge_mode not in MERGE_MODES:
        raise ValueError(f"unsupported merge_mode '{merge_mode}'")

    return {
        "desired_speed_bin": desired_speed_bin,
        "desired_headway_bin": desired_headway_bin,
        "lane_preference": lane_preference,
        "merge_mode": merge_mode,
    }


def validate_action_mapping(
    av_actions: Mapping[str, Mapping[str, Any]],
    expected_agent_ids: Iterable[str] | None = None,
) -> dict[str, AVAction]:
    """Validate and normalize a public AV action mapping."""
    normalized = {agent_id: validate_action(action) for agent_id, action in av_actions.items()}

    if expected_agent_ids is not None:
        expected = set(expected_agent_ids)
        actual = set(normalized)
        if actual != expected:
            raise ValueError(f"action mapping keys {sorted(actual)} do not match expected AV ids {sorted(expected)}")

    return normalized


def decode_speed_bin(
    speed_bin: SpeedBin,
    free_flow_speed_mps: float = 30.0,
    min_contextual_speed_mps: float = 12.0,
) -> float:
    """Decode a discrete speed target into a contextual speed in m/s."""
    offsets = {
        "slow": -10.0,
        "nominal": -3.0,
        "fast": 0.0,
    }
    return max(min_contextual_speed_mps, free_flow_speed_mps + offsets[speed_bin])


def decode_headway_bin(headway_bin: HeadwayBin) -> float:
    """Decode a discrete headway target into seconds."""
    return {
        "normal": 1.6,
        "larger": 2.2,
        "largest": 3.0,
    }[headway_bin]


def lane_preference_to_action(lane_preference: LanePreference) -> str | None:
    """Map conservative lane preference to a simulator lane action candidate."""
    return {
        "keep": None,
        "prefer_left_if_safe": "LANE_LEFT",
        "prefer_right_if_safe": "LANE_RIGHT",
    }[lane_preference]
