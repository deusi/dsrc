from __future__ import annotations

import math
from typing import Mapping

from src.road.highway_imports import ensure_highway_env_importable
from src.road.segment_graph import TopologySpec, edge_id, lane_segment_map

ensure_highway_env_importable()

from highway_env.road.lane import LineType, SineLane, StraightLane
from highway_env.road.road import RoadNetwork


DEFAULT_LEAF_M = 500.0
DEFAULT_MIDDLE_M = 600.0
DEFAULT_TRUNK_M = 600.0
DEFAULT_BOTTLENECK_M = 300.0
DEFAULT_SPEED_LIMIT_MPS = 30.0


def _add_sine(net: RoadNetwork, start: str, end: str, p0: list[float], p1: list[float], length_hint: float, speed_limit: float) -> None:
    net.add_lane(
        start,
        end,
        SineLane(
            p0,
            p1,
            amplitude=2.0,
            pulsation=math.pi / max(length_hint, 1.0),
            phase=math.pi / 2.0,
            line_types=[LineType.CONTINUOUS_LINE, LineType.CONTINUOUS_LINE],
            speed_limit=speed_limit,
        ),
    )


def build_inverted_tree_topology(config: Mapping | None = None) -> TopologySpec:
    cfg = dict(config or {})
    leaf = float(cfg.get("leaf_m", DEFAULT_LEAF_M))
    middle = float(cfg.get("middle_m", DEFAULT_MIDDLE_M))
    trunk = float(cfg.get("trunk_m", DEFAULT_TRUNK_M))
    bottleneck = float(cfg.get("bottleneck_m", DEFAULT_BOTTLENECK_M))
    speed_limit = float(cfg.get("speed_limit_mps", DEFAULT_SPEED_LIMIT_MPS))

    net = RoadNetwork()
    c, s, n = LineType.CONTINUOUS_LINE, LineType.STRIPED, LineType.NONE
    leaf_y = {
        "a1": 24.0,
        "a2": 18.0,
        "a3": 12.0,
        "a4": -12.0,
        "a5": -18.0,
        "a6": -24.0,
    }
    for index, (leaf_id, y) in enumerate(leaf_y.items(), start=1):
        merge_node = "b1" if index <= 3 else "b2"
        target_y = 12.0 if index <= 3 else -12.0
        _add_sine(net, f"{leaf_id}_entry", merge_node, [0.0, y], [leaf, target_y], leaf, speed_limit)

    lane_width = StraightLane.DEFAULT_WIDTH
    b1_starts = (12.0, 12.0 + lane_width)
    b1_ends = (4.0, 8.0)
    b2_starts = (-12.0, -12.0 - lane_width)
    b2_ends = (0.0, -4.0)
    for lane_id in range(2):
        net.add_lane(
            "b1",
            "c",
            SineLane(
                [leaf, b1_starts[lane_id]],
                [leaf + middle, b1_ends[lane_id]],
                amplitude=2.0,
                pulsation=math.pi / middle,
                phase=math.pi / 2.0,
                line_types=[c if lane_id == 0 else s, c if lane_id == 1 else n],
                speed_limit=speed_limit,
            ),
        )
        net.add_lane(
            "b2",
            "c",
            SineLane(
                [leaf, b2_starts[lane_id]],
                [leaf + middle, b2_ends[lane_id]],
                amplitude=2.0,
                pulsation=math.pi / middle,
                phase=math.pi / 2.0,
                line_types=[c if lane_id == 0 else s, c if lane_id == 1 else n],
                speed_limit=speed_limit,
            ),
        )

    for lane_id, y in enumerate((0.0, lane_width)):
        net.add_lane(
            "c",
            "d",
            StraightLane(
                [leaf + middle, y],
                [leaf + middle + trunk, y],
                line_types=[c if lane_id == 0 else s, c if lane_id == 1 else n],
                speed_limit=speed_limit,
            ),
        )
    net.add_lane(
        "d",
        "exit",
        StraightLane(
            [leaf + middle + trunk, 0.0],
            [leaf + middle + trunk + bottleneck, 0.0],
            line_types=[c, c],
            speed_limit=0.8 * speed_limit,
        ),
    )

    edge_segments: dict[tuple[str, str], str] = {}
    for leaf_id in leaf_y:
        edge_segments[(f"{leaf_id}_entry", "b1" if leaf_id in {"a1", "a2", "a3"} else "b2")] = f"tree_leaf_{leaf_id}"
    edge_segments.update(
        {
            ("b1", "c"): "tree_middle_b1",
            ("b2", "c"): "tree_middle_b2",
            ("c", "d"): "tree_trunk_c",
            ("d", "exit"): "tree_bottleneck_d",
        }
    )
    segment_ids = (
        "tree_leaf_a1",
        "tree_leaf_a2",
        "tree_leaf_a3",
        "tree_leaf_a4",
        "tree_leaf_a5",
        "tree_leaf_a6",
        "tree_middle_b1",
        "tree_middle_b2",
        "tree_trunk_c",
        "tree_bottleneck_d",
    )
    spec = TopologySpec(
        topology_id="inverted_tree",
        road_network=net,
        segment_ids=segment_ids,
        segment_lengths={
            **{f"tree_leaf_{leaf_id}": leaf for leaf_id in leaf_y},
            "tree_middle_b1": middle,
            "tree_middle_b2": middle,
            "tree_trunk_c": trunk,
            "tree_bottleneck_d": bottleneck,
        },
        segment_edges={
            segment: tuple(edge_id(*edge) for edge, edge_segment in edge_segments.items() if edge_segment == segment)
            for segment in segment_ids
        },
        entry_segments=tuple(f"tree_leaf_{leaf_id}" for leaf_id in leaf_y),
        exit_segments=("tree_bottleneck_d",),
        merge_nodes=("tree_merge_b1", "tree_merge_b2", "tree_merge_c"),
        detector_locations={
            **{f"tree_leaf_{leaf_id}": (leaf,) for leaf_id in leaf_y},
            "tree_middle_b1": (middle,),
            "tree_middle_b2": (middle,),
            "tree_trunk_c": (trunk / 2.0,),
            "tree_bottleneck_d": (bottleneck,),
        },
        lane_counts={
            **{f"tree_leaf_{leaf_id}": 1 for leaf_id in leaf_y},
            "tree_middle_b1": 2,
            "tree_middle_b2": 2,
            "tree_trunk_c": 2,
            "tree_bottleneck_d": 1,
        },
        bottleneck_segments=("tree_bottleneck_d",),
        lane_segments=lane_segment_map(net, edge_segments),
        supports_lane_change=True,
        metadata={
            "leaf_m": leaf,
            "middle_m": middle,
            "trunk_m": trunk,
            "bottleneck_m": bottleneck,
            "speed_limit_mps": speed_limit,
        },
    )
    spec.validate()
    return spec

