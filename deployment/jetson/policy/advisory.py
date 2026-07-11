"""Turn the actor's discrete action into driver-facing advisory text.

The speed decode mirrors src/envs/wrappers.py decode_speed_bin: the base
speed is the cooperation segment_target_speed (which equals the
configured free-flow speed when no cooperating AVs are heard), shifted
by the bin offset and floored at the contextual minimum - exactly what
the sim's safety layer would do with the same action.

This module also returns the decoded headway target so the run loop can
feed it back into the next observation's target_headway_s (matching the
sim loop, where the previous action shapes the next observation).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from policy import sim_contract
from policy.actor_runtime import PolicyOutput

MPS_TO_MPH = 2.236936
MPS_TO_KMH = 3.6

LANE_TEXT = {
    "keep": "Keep lane",
    "prefer_left_if_safe": "Prepare left (if safe)",
    "prefer_right_if_safe": "Prepare right (if safe)",
}
MERGE_TEXT = {
    "normal": "Normal driving",
    "create_gap": "Creating merge gap",
    "hold_lane": "Hold lane (merge zone)",
}
TRAFFIC_TEXT = {0: "Light", 1: "Moderate", 2: "Heavy"}


@dataclass
class Advisory:
    recommended_speed_mps: float
    recommended_speed_display: float
    current_speed_display: float
    units: str
    headway_target_s: float
    lane_text: str
    merge_text: str
    traffic_text: str
    confidence_label: str
    confidence: float
    action: dict[str, str]

    def one_line(self) -> str:
        return (
            f"rec {self.recommended_speed_display:5.1f} {self.units} | "
            f"cur {self.current_speed_display:5.1f} {self.units} | "
            f"{self.lane_text} | headway {self.headway_target_s:.1f}s | "
            f"traffic {self.traffic_text} | conf {self.confidence_label}"
        )


class AdvisoryDecoder:
    def __init__(
        self,
        units: str = "mph",
        min_contextual_speed_mps: float = 12.0,
        confidence_low_below: float = 0.45,
        confidence_high_at: float = 0.70,
    ) -> None:
        if units not in ("mph", "kmh", "mps"):
            raise ValueError(f"unknown units '{units}'")
        self.units = units
        self.min_contextual_speed_mps = min_contextual_speed_mps
        self.confidence_low_below = confidence_low_below
        self.confidence_high_at = confidence_high_at

    def _display(self, mps: float) -> float:
        if self.units == "mph":
            return mps * MPS_TO_MPH
        if self.units == "kmh":
            return mps * MPS_TO_KMH
        return mps

    def decode(self, policy_out: PolicyOutput, obs: dict[str, Any]) -> Advisory:
        action = policy_out.action
        base_speed = float(obs.get("cooperation", {}).get("segment_target_speed", 30.0))
        recommended = sim_contract.decode_speed_bin(
            action["desired_speed_bin"], base_speed, self.min_contextual_speed_mps
        )
        headway = sim_contract.decode_headway_bin(action["desired_headway_bin"])
        if policy_out.confidence < self.confidence_low_below:
            confidence_label = "low"
        elif policy_out.confidence >= self.confidence_high_at:
            confidence_label = "high"
        else:
            confidence_label = "medium"
        return Advisory(
            recommended_speed_mps=recommended,
            recommended_speed_display=self._display(recommended),
            current_speed_display=self._display(float(obs.get("ego_speed", 0.0))),
            units=self.units,
            headway_target_s=headway,
            lane_text=LANE_TEXT[action["lane_preference"]],
            merge_text=MERGE_TEXT[action["merge_mode"]],
            traffic_text=TRAFFIC_TEXT.get(int(obs.get("local_density_bin", 0)), "?"),
            confidence_label=confidence_label,
            confidence=policy_out.confidence,
            action=action,
        )
