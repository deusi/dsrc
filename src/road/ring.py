from __future__ import annotations

import math
from typing import Mapping

from src.road.highway_imports import ensure_highway_env_importable
from src.road.segment_graph import TopologySpec, edge_id, lane_segment_map

ensure_highway_env_importable()

from highway_env.road.lane import CircularLane, LineType
from highway_env.road.road import RoadNetwork


DEFAULT_CIRCUMFERENCE_M = 260.0
DEFAULT_SPEED_LIMIT_MPS = 25.0


def build_ring_topology(config: Mapping | None = None) -> TopologySpec:
    cfg = dict(config or {})
    circumference = float(cfg.get("circumference_m", DEFAULT_CIRCUMFERENCE_M))
    speed_limit = float(cfg.get("speed_limit_mps", DEFAULT_SPEED_LIMIT_MPS))
    radius = circumference / (2.0 * math.pi)

    net = RoadNetwork()
    nodes = ("r0", "r1", "r2", "r3")
    phases = (0.0, math.pi / 2.0, math.pi, 3.0 * math.pi / 2.0, 2.0 * math.pi)
    for index, start_node in enumerate(nodes):
        end_node = nodes[(index + 1) % len(nodes)]
        net.add_lane(
            start_node,
            end_node,
            CircularLane(
                center=[0.0, 0.0],
                radius=radius,
                start_phase=phases[index],
                end_phase=phases[index + 1],
                clockwise=True,
                line_types=[LineType.CONTINUOUS_LINE, LineType.CONTINUOUS_LINE],
                speed_limit=speed_limit,
            ),
        )

    edge_segments = {(start, nodes[(index + 1) % len(nodes)]): "ring_main" for index, start in enumerate(nodes)}
    spec = TopologySpec(
        topology_id="ring",
        road_network=net,
        segment_ids=("ring_main",),
        segment_lengths={"ring_main": circumference},
        segment_edges={"ring_main": tuple(edge_id(start, end) for (start, end) in edge_segments)},
        entry_segments=("ring_main",),
        exit_segments=("ring_main",),
        merge_nodes=(),
        detector_locations={"ring_main": (0.25 * circumference, 0.5 * circumference, 0.75 * circumference)},
        lane_counts={"ring_main": 1},
        bottleneck_segments=(),
        lane_segments=lane_segment_map(net, edge_segments),
        supports_lane_change=False,
        metadata={"circumference_m": circumference, "radius_m": radius, "speed_limit_mps": speed_limit},
    )
    spec.validate()
    return spec

