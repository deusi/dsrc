from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Literal, Mapping, TypeAlias

from src.envs.base_ctde_env import AVActionMap, AVObservationMap, GlobalState


CooperationMode: TypeAlias = Literal["none", "local_aggregate", "global_state"]
SafetyMode: TypeAlias = Literal["external_filter", "integrated_rl", "simulator_default"]

COOPERATION_MODES: tuple[CooperationMode, ...] = ("none", "local_aggregate", "global_state")
SAFETY_MODES: tuple[SafetyMode, ...] = ("external_filter", "integrated_rl", "simulator_default")


@dataclass(frozen=True)
class ControllerMetadata:
    name: str
    family: str
    version: str = "v1"
    requires_global_state: bool = False
    cooperation_mode: CooperationMode = "none"
    safety_mode: SafetyMode = "external_filter"
    supports_fallback_individual: bool = True

    def __post_init__(self) -> None:
        if self.cooperation_mode not in COOPERATION_MODES:
            raise ValueError(f"unsupported cooperation_mode '{self.cooperation_mode}'")
        if self.safety_mode not in SAFETY_MODES:
            raise ValueError(f"unsupported safety_mode '{self.safety_mode}'")
        if self.requires_global_state and self.cooperation_mode == "none":
            object.__setattr__(self, "cooperation_mode", "global_state")


class BaseController(ABC):
    """Stable controller contract shared by baselines and learned policies."""

    metadata: ControllerMetadata

    def __init__(self, metadata: ControllerMetadata) -> None:
        self.metadata = metadata

    @property
    def name(self) -> str:
        return self.metadata.name

    def reset(
        self,
        env_metadata: Mapping[str, Any] | None = None,
        seed: int | None = None,
    ) -> None:
        """Reset controller state at episode start."""

    @abstractmethod
    def act(
        self,
        local_obs: AVObservationMap,
        global_state: GlobalState | None = None,
    ) -> AVActionMap:
        """Produce one public action per AV identifier."""
