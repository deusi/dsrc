from __future__ import annotations

from typing import Any, Callable, Mapping

from src.road.inverted_tree import build_inverted_tree_topology
from src.road.merge import build_merge_topology
from src.road.ring import build_ring_topology
from src.road.segment_graph import TopologySpec
from src.road.straight import build_straight_topology


TOPOLOGY_IDS = ("ring", "straight_single_lane", "straight_multilane", "merge", "inverted_tree")


def build_topology(topology_id: str, config: Mapping[str, Any] | None = None) -> TopologySpec:
    road_config = config.get("road", config) if config else None
    builders: dict[str, Callable[[Mapping | None], TopologySpec]] = {
        "ring": build_ring_topology,
        "straight_single_lane": lambda cfg: build_straight_topology("straight_single_lane", cfg),
        "straight_multilane": lambda cfg: build_straight_topology("straight_multilane", cfg),
        "merge": build_merge_topology,
        "inverted_tree": build_inverted_tree_topology,
    }
    try:
        return builders[topology_id](road_config)
    except KeyError as exc:
        raise ValueError(f"unknown topology_id '{topology_id}'") from exc
