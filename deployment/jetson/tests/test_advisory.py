from __future__ import annotations

import pytest

from policy.actor_runtime import PolicyOutput
from policy.advisory import MPS_TO_MPH, AdvisoryDecoder


def policy_out(speed_bin: str = "nominal", headway: str = "normal", lane: str = "keep",
               merge: str = "normal", confidence: float = 0.8) -> PolicyOutput:
    return PolicyOutput(
        action={
            "desired_speed_bin": speed_bin,
            "desired_headway_bin": headway,
            "lane_preference": lane,
            "merge_mode": merge,
        },
        head_probs={}, chosen_prob={}, confidence=confidence, latency_ms=0.1,
    )


def obs_with(target_speed: float = 30.0, ego_speed: float = 25.0, density_bin: int = 1) -> dict:
    return {
        "ego_speed": ego_speed,
        "local_density_bin": density_bin,
        "cooperation": {"segment_target_speed": target_speed},
    }


def test_nominal_decode_matches_sim_wrapper() -> None:
    adv = AdvisoryDecoder(units="mph").decode(policy_out("nominal"), obs_with(30.0))
    assert adv.recommended_speed_mps == pytest.approx(27.0)  # 30 - 3
    assert adv.recommended_speed_display == pytest.approx(27.0 * MPS_TO_MPH)


def test_slow_decode_floors_at_contextual_minimum() -> None:
    adv = AdvisoryDecoder().decode(policy_out("slow"), obs_with(target_speed=20.0))
    assert adv.recommended_speed_mps == 12.0  # max(12, 20 - 10)


def test_headway_and_texts() -> None:
    adv = AdvisoryDecoder().decode(
        policy_out(headway="largest", lane="prefer_left_if_safe", merge="create_gap"),
        obs_with(),
    )
    assert adv.headway_target_s == 3.0
    assert "left" in adv.lane_text.lower()
    assert "gap" in adv.merge_text.lower()
    assert adv.traffic_text == "Moderate"


@pytest.mark.parametrize(
    "confidence,label", [(0.30, "low"), (0.55, "medium"), (0.69, "medium"), (0.95, "high")]
)
def test_confidence_labels(confidence: float, label: str) -> None:
    adv = AdvisoryDecoder().decode(policy_out(confidence=confidence), obs_with())
    assert adv.confidence_label == label


def test_units_kmh() -> None:
    adv = AdvisoryDecoder(units="kmh").decode(policy_out(), obs_with(30.0))
    assert adv.recommended_speed_display == pytest.approx(27.0 * 3.6)
