from __future__ import annotations

from dataclasses import dataclass

from src.road.highway_imports import ensure_highway_env_importable
from src.road.segment_graph import TopologySpec

ensure_highway_env_importable()

from highway_env.road.road import LaneIndex


@dataclass(frozen=True)
class BranchRoute:
    branch_id: str
    entry_edge: tuple[str, str]
    entry_segment: str
    destination: str
    lane_count: int

    def lane_index(self, lane_id: int) -> LaneIndex:
        if lane_id < 0 or lane_id >= self.lane_count:
            raise ValueError(f"lane_id {lane_id} outside branch {self.branch_id}")
        return (self.entry_edge[0], self.entry_edge[1], lane_id)


@dataclass(frozen=True)
class RoutePlan:
    enabled: bool
    destination: str | None
    branches: tuple[BranchRoute, ...]


def build_route_plan(topology: TopologySpec) -> RoutePlan:
    if topology.topology_id == "ring":
        return RoutePlan(enabled=False, destination=None, branches=())
    destination_by_topology = {
        "straight_single_lane": "s3",
        "straight_multilane": "s3",
        "merge": "m3",
        "inverted_tree": "exit",
        "inverted_tree_bottleneck": "exit",
    }
    try:
        destination = destination_by_topology[topology.topology_id]
    except KeyError as exc:
        raise ValueError(f"unsupported topology for demand: {topology.topology_id}") from exc

    branches: list[BranchRoute] = []
    for segment_id in topology.entry_segments:
        edge_ids = topology.segment_edges[segment_id]
        if len(edge_ids) != 1:
            raise ValueError(f"entry segment {segment_id} must map to exactly one entry road edge")
        start, end = edge_ids[0].split("->", 1)
        branches.append(
            BranchRoute(
                branch_id=_branch_id(topology.topology_id, segment_id),
                entry_edge=(start, end),
                entry_segment=segment_id,
                destination=destination,
                lane_count=topology.lane_counts[segment_id],
            )
        )
    return RoutePlan(enabled=True, destination=destination, branches=tuple(branches))


def road_route_to_destination(lane_index: LaneIndex, destination: str, topology: TopologySpec) -> list[LaneIndex]:
    """Build a no-lane-pinning route from the current road to the common destination."""
    path = topology.road_network.shortest_path(lane_index[1], destination)
    road_index = (lane_index[0], lane_index[1], None)
    if not path:
        return [road_index]
    return [road_index] + [(path[index], path[index + 1], None) for index in range(len(path) - 1)]


def _branch_id(topology_id: str, segment_id: str) -> str:
    if topology_id.startswith("straight"):
        return "main"
    if topology_id == "merge":
        return "ramp" if segment_id == "merge_ramp" else "main"
    if topology_id in {"inverted_tree", "inverted_tree_bottleneck"}:
        return segment_id.removeprefix("tree_leaf_")
    return segment_id
