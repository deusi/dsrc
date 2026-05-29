from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import numpy as np


PROFILE_IDS = ("cautious", "normal", "aggressive")

DEFAULT_PROFILE_PARAMS: dict[str, dict[str, float]] = {
    "normal": {
        "speed_multiplier": 1.0,
        "TIME_WANTED": 1.5,
        "POLITENESS": 0.0,
        "LANE_CHANGE_DELAY": 1.0,
        "LANE_CHANGE_MIN_ACC_GAIN": 0.2,
        "LANE_CHANGE_MAX_BRAKING_IMPOSED": 2.0,
        "COMFORT_ACC_MAX": 3.0,
        "COMFORT_ACC_MIN": -5.0,
        "ACC_MAX": 6.0,
    },
    "cautious": {
        "speed_multiplier": 0.9,
        "TIME_WANTED": 2.0,
        "POLITENESS": 0.5,
        "LANE_CHANGE_DELAY": 2.0,
        "LANE_CHANGE_MIN_ACC_GAIN": 0.35,
        "LANE_CHANGE_MAX_BRAKING_IMPOSED": 1.2,
        "COMFORT_ACC_MAX": 2.0,
        "COMFORT_ACC_MIN": -3.0,
        "ACC_MAX": 4.0,
    },
    "aggressive": {
        "speed_multiplier": 1.1,
        "TIME_WANTED": 0.9,
        "POLITENESS": 0.0,
        "LANE_CHANGE_DELAY": 0.5,
        "LANE_CHANGE_MIN_ACC_GAIN": 0.05,
        "LANE_CHANGE_MAX_BRAKING_IMPOSED": 3.0,
        "COMFORT_ACC_MAX": 4.0,
        "COMFORT_ACC_MIN": -6.0,
        "ACC_MAX": 7.0,
    },
}


@dataclass(frozen=True)
class HumanBehaviorProfile:
    profile_id: str
    speed_multiplier: float
    params: Mapping[str, float]


@dataclass(frozen=True)
class HumanBehaviorModel:
    model_id: str
    profiles: Mapping[str, HumanBehaviorProfile]
    weights: Mapping[str, float]

    def sample_profile_id(self, rng: np.random.RandomState) -> str:
        profile_ids = tuple(self.weights)
        probabilities = tuple(self.weights[profile_id] for profile_id in profile_ids)
        return str(rng.choice(profile_ids, p=probabilities))

    def profile_for(self, profile_id: str) -> HumanBehaviorProfile:
        if profile_id not in self.profiles:
            raise ValueError(f"unknown human behavior profile '{profile_id}'")
        return self.profiles[profile_id]


def load_human_behavior_model(config: Mapping[str, Any] | None) -> HumanBehaviorModel:
    cfg = dict(config or {"id": "normal"})
    model_id = str(cfg.get("id", "normal"))

    profile_overrides = _profile_overrides(cfg)
    profiles = {profile_id: _build_profile(profile_id, profile_overrides.get(profile_id, {})) for profile_id in PROFILE_IDS}

    raw_weights = _raw_weights(cfg, model_id)
    weights = _normalize_weights(raw_weights)
    return HumanBehaviorModel(model_id=model_id, profiles=profiles, weights=weights)


def apply_human_behavior_profile(
    vehicle: Any,
    profile: HumanBehaviorProfile,
    *,
    base_target_speed_mps: float,
    min_speed_mps: float,
    max_speed_mps: float,
    lane_speed_limit_mps: float | None = None,
) -> float:
    for name, value in profile.params.items():
        setattr(vehicle, name, float(value))

    upper_speed = max_speed_mps
    if lane_speed_limit_mps is not None:
        upper_speed = min(upper_speed, float(lane_speed_limit_mps))
    target_speed = float(np.clip(base_target_speed_mps * profile.speed_multiplier, min_speed_mps, upper_speed))
    vehicle.target_speed = target_speed
    vehicle.behavior_profile = profile.profile_id
    return target_speed


def _raw_weights(cfg: Mapping[str, Any], model_id: str) -> Mapping[str, float]:
    if "mix" in cfg:
        mix = cfg["mix"]
        if not isinstance(mix, Mapping):
            raise ValueError("human_model mix must be a mapping")
        return {str(profile_id): float(weight) for profile_id, weight in mix.items()}
    if "profiles" in cfg and isinstance(cfg["profiles"], Mapping):
        weights: dict[str, float] = {}
        for profile_id, profile_cfg in cfg["profiles"].items():
            if isinstance(profile_cfg, Mapping):
                weights[str(profile_id)] = float(profile_cfg.get("weight", 1.0))
            else:
                weights[str(profile_id)] = 1.0
        return weights
    if model_id in PROFILE_IDS:
        return {model_id: 1.0}
    raise ValueError(f"unknown human behavior model '{model_id}'")


def _profile_overrides(cfg: Mapping[str, Any]) -> dict[str, Mapping[str, Any]]:
    overrides: dict[str, Mapping[str, Any]] = {}
    model_id = str(cfg.get("id", "normal"))
    profile_cfg = cfg.get("profile")
    if profile_cfg is not None:
        if not isinstance(profile_cfg, Mapping):
            raise ValueError("human_model profile must be a mapping")
        if model_id not in PROFILE_IDS:
            raise ValueError(f"profile overrides require a known profile id, got '{model_id}'")
        overrides[model_id] = profile_cfg

    profiles_cfg = cfg.get("profiles")
    if isinstance(profiles_cfg, Mapping):
        for profile_id, entry in profiles_cfg.items():
            if str(profile_id) not in PROFILE_IDS:
                raise ValueError(f"unknown human behavior profile '{profile_id}'")
            if isinstance(entry, Mapping) and isinstance(entry.get("profile"), Mapping):
                overrides[str(profile_id)] = entry["profile"]
    return overrides


def _build_profile(profile_id: str, overrides: Mapping[str, Any]) -> HumanBehaviorProfile:
    if profile_id not in DEFAULT_PROFILE_PARAMS:
        raise ValueError(f"unknown human behavior profile '{profile_id}'")
    params = dict(DEFAULT_PROFILE_PARAMS[profile_id])
    params.update({str(key): float(value) for key, value in overrides.items()})
    speed_multiplier = float(params.pop("speed_multiplier"))
    if speed_multiplier <= 0:
        raise ValueError("human behavior speed_multiplier must be positive")
    return HumanBehaviorProfile(profile_id=profile_id, speed_multiplier=speed_multiplier, params=params)


def _normalize_weights(raw_weights: Mapping[str, float]) -> dict[str, float]:
    if not raw_weights:
        raise ValueError("human behavior model must include at least one profile weight")
    weights: dict[str, float] = {}
    for profile_id, weight in raw_weights.items():
        if profile_id not in PROFILE_IDS:
            raise ValueError(f"unknown human behavior profile '{profile_id}'")
        if weight < 0:
            raise ValueError("human behavior mixture weights must be non-negative")
        weights[profile_id] = float(weight)
    total = sum(weights.values())
    if total <= 0:
        raise ValueError("human behavior mixture weights must have positive total")
    return {profile_id: weight / total for profile_id, weight in weights.items() if weight > 0}
