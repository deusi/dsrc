from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from src.controllers import BaseController
from src.baselines.controllers import (
    AVMediatedSpeedHarmonizationController,
    BackpressureController,
    CooperativeSmoothingController,
    DensityLookupController,
    DynamicSpeedLimitController,
    NoAVController,
    RandomAVController,
    SelfishAVController,
)


BASELINE_NAMES = (
    "no_av",
    "random_av",
    "selfish_av",
    "density_lookup",
    "dynamic_speed_limit",
    "av_mediated_speed_harmonization",
    "backpressure",
    "cooperative_smoothing",
)


def make_baseline(name: str, config: Mapping[str, Any] | None = None) -> BaseController:
    normalized = name.strip().lower()
    factories = {
        "no_av": NoAVController,
        "random_av": RandomAVController,
        "selfish_av": SelfishAVController,
        "density_lookup": DensityLookupController,
        "dynamic_speed_limit": DynamicSpeedLimitController,
        "av_mediated_speed_harmonization": AVMediatedSpeedHarmonizationController,
        "speed_harmonization": AVMediatedSpeedHarmonizationController,
        "backpressure": BackpressureController,
        "cooperative_smoothing": CooperativeSmoothingController,
    }
    try:
        return factories[normalized](config)
    except KeyError as exc:
        raise ValueError(f"unknown baseline '{name}'") from exc
