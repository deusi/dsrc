from __future__ import annotations

import math
from typing import Mapping

import numpy as np

from src.road.highway_imports import ensure_highway_env_importable
from src.road.segment_graph import TopologySpec, edge_id, lane_segment_map

ensure_highway_env_importable()

from highway_env.road.lane import LineType, SineLane, StraightLane
from highway_env.road.road import RoadNetwork


DEFAULT_MAINLINE_M = 800.0
DEFAULT_RAMP_M = 500.0
DEFAULT_WEAVE_M = 250.0
DEFAULT_TRUNK_M = 700.0
DEFAULT_SPEED_LIMIT_MPS = 30.0


def build_merge_topology(config: Mapping | None = None) -> TopologySpec:
    cfg = dict(config or {})
    mainline = float(cfg.get("mainline_upstream_m", DEFAULT_MAINLINE_M))
    ramp = float(cfg.get("ramp_m", DEFAULT_RAMP_M))
    weave = float(cfg.get("weave_m", DEFAULT_WEAVE_M))
    trunk = float(cfg.get("trunk_m", DEFAULT_TRUNK_M))
    speed_limit = float(cfg.get("speed_limit_mps", DEFAULT_SPEED_LIMIT_MPS))

    net = RoadNetwork()
    lane_width = StraightLane.DEFAULT_WIDTH
    c, s, n = LineType.CONTINUOUS_LINE, LineType.STRIPED, LineType.NONE
    for lane_id, y in enumerate((0.0, lane_width)):
        line_types = [c if lane_id == 0 else s, c if lane_id == 1 else n]
        net.add_lane("m0", "m1", StraightLane([0.0, y], [mainline, y], line_types=line_types, speed_limit=speed_limit))
        net.add_lane("m1", "m2", StraightLane([mainline, y], [mainline + weave, y], line_types=line_types, speed_limit=speed_limit))
        net.add_lane(
            "m2",
            "m3",
            StraightLane([mainline + weave, y], [mainline + weave + trunk, y], line_types=line_types, speed_limit=speed_limit),
        )

    ramp_start = [mainline - ramp, 4.0 * lane_width]
    ramp_end = [mainline, lane_width]
    net.add_lane(
        "r0",
        "m1",
        SineLane(
            ramp_start,
            ramp_end,
            amplitude=lane_width,
            pulsation=math.pi / ramp,
            phase=math.pi / 2.0,
            line_types=[c, c],
            speed_limit=0.8 * speed_limit,
        ),
    )

    edge_segments = {
        ("m0", "m1"): "merge_main_upstream",
        ("r0", "m1"): "merge_ramp",
        ("m1", "m2"): "merge_weave",
        ("m2", "m3"): "merge_trunk",
    }
    spec = TopologySpec(
        topology_id="merge",
        road_network=net,
        segment_ids=("merge_main_upstream", "merge_ramp", "merge_weave", "merge_trunk"),
        segment_lengths={
            "merge_main_upstream": mainline,
            "merge_ramp": float(np.linalg.norm(np.array(ramp_end) - np.array(ramp_start))),
            "merge_weave": weave,
            "merge_trunk": trunk,
        },
        segment_edges={segment: (edge_id(*edge),) for edge, segment in edge_segments.items()},
        entry_segments=("merge_main_upstream", "merge_ramp"),
        exit_segments=("merge_trunk",),
        merge_nodes=("merge_0",),
        detector_locations={
            "merge_main_upstream": (mainline - 50.0,),
            "merge_ramp": (max(0.0, ramp - 50.0),),
            "merge_weave": (weave / 2.0,),
            "merge_trunk": (trunk - 50.0,),
        },
        lane_counts={
            "merge_main_upstream": 2,
            "merge_ramp": 1,
            "merge_weave": 2,
            "merge_trunk": 2,
        },
        bottleneck_segments=("merge_weave",),
        lane_segments=lane_segment_map(net, edge_segments),
        supports_lane_change=True,
        metadata={
            "mainline_upstream_m": mainline,
            "ramp_m": ramp,
            "weave_m": weave,
            "trunk_m": trunk,
            "speed_limit_mps": speed_limit,
        },
    )
    spec.validate()
    return spec

