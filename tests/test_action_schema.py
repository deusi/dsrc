from __future__ import annotations

import pytest

from src.envs.wrappers import validate_action


def safe_action(**overrides: str) -> dict[str, str]:
    action = {
        "desired_speed_bin": "nominal",
        "desired_headway_bin": "normal",
        "lane_preference": "keep",
        "merge_mode": "normal",
    }
    action.update(overrides)
    return action


def test_v2_action_schema_accepts_allowed_bins() -> None:
    assert validate_action(safe_action()) == safe_action()
    assert validate_action(safe_action(desired_speed_bin="slow"))["desired_speed_bin"] == "slow"
    assert validate_action(safe_action(desired_headway_bin="largest"))["desired_headway_bin"] == "largest"
    assert validate_action(safe_action(lane_preference="prefer_right_if_safe"))["lane_preference"] == "prefer_right_if_safe"
    assert validate_action(safe_action(merge_mode="create_gap"))["merge_mode"] == "create_gap"


@pytest.mark.parametrize(
    "bad_action",
    [
        {"desired_speed": 22.0, "desired_lane": "keep"},
        safe_action(lane_preference="left"),
        safe_action(lane_preference="right"),
        safe_action(merge_mode="block_lanes"),
        {"desired_speed_bin": "nominal", "lane_preference": "keep", "merge_mode": "normal"},
    ],
)
def test_v2_action_schema_rejects_legacy_or_unsafe_actions(bad_action: dict) -> None:
    with pytest.raises(ValueError):
        validate_action(bad_action)

