from __future__ import annotations

from typing import Mapping

from src.road.highway_imports import ensure_highway_env_importable
from src.road.segment_graph import TopologySpec, edge_id, lane_segment_map

ensure_highway_env_importable()

from highway_env.road.road import RoadNetwork


DEFAULT_LENGTH_M = 1500.0
DEFAULT_SPEED_LIMIT_MPS = 30.0


def build_straight_topology(topology_id: str, config: Mapping | None = None) -> TopologySpec:
    if topology_id not in {"straight_single_lane", "straight_multilane"}:
        raise ValueError(f"unsupported straight topology '{topology_id}'")

    cfg = dict(config or {})
    length = float(cfg.get("length_m", DEFAULT_LENGTH_M))
    lanes = int(cfg.get("lanes_count", 1 if topology_id == "straight_single_lane" else 3))
    speed_limit = float(cfg.get("speed_limit_mps", DEFAULT_SPEED_LIMIT_MPS))
    segment_length = length / 3.0
    downstream_detector = max(0.0, min(segment_length, 1450.0 - 2.0 * segment_length))

    net = RoadNetwork()
    RoadNetwork.straight_road_network(
        lanes=lanes,
        start=0.0,
        length=segment_length,
        speed_limit=speed_limit,
        nodes_str=("s0", "s1"),
        net=net,
    )
    RoadNetwork.straight_road_network(
        lanes=lanes,
        start=segment_length,
        length=segment_length,
        speed_limit=speed_limit,
        nodes_str=("s1", "s2"),
        net=net,
    )
    RoadNetwork.straight_road_network(
        lanes=lanes,
        start=2.0 * segment_length,
        length=segment_length,
        speed_limit=speed_limit,
        nodes_str=("s2", "s3"),
        net=net,
    )

    edge_segments = {
        ("s0", "s1"): "straight_upstream",
        ("s1", "s2"): "straight_mid",
        ("s2", "s3"): "straight_downstream",
    }
    spec = TopologySpec(
        topology_id=topology_id,
        road_network=net,
        segment_ids=("straight_upstream", "straight_mid", "straight_downstream"),
        segment_lengths={
            "straight_upstream": segment_length,
            "straight_mid": segment_length,
            "straight_downstream": segment_length,
        },
        segment_edges={segment: (edge_id(*edge),) for edge, segment in edge_segments.items()},
        entry_segments=("straight_upstream",),
        exit_segments=("straight_downstream",),
        merge_nodes=(),
        detector_locations={
            "straight_upstream": (segment_length,),
            "straight_mid": (segment_length,),
            "straight_downstream": (downstream_detector,),
        },
        lane_counts={segment: lanes for segment in ("straight_upstream", "straight_mid", "straight_downstream")},
        bottleneck_segments=(),
        lane_segments=lane_segment_map(net, edge_segments),
        supports_lane_change=lanes > 1,
        metadata={"length_m": length, "speed_limit_mps": speed_limit},
    )
    spec.validate()
    return spec
