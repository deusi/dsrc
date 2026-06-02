from __future__ import annotations

import pytest

from src.envs.topology_env import HighwayTopologyEnv
from src.road.topology_factory import TOPOLOGY_IDS, build_topology


EXPECTED_SEGMENTS = {
    "ring": ("ring_main",),
    "straight_single_lane": ("straight_upstream", "straight_mid", "straight_downstream"),
    "straight_multilane": ("straight_upstream", "straight_mid", "straight_downstream"),
    "merge": ("merge_main_upstream", "merge_ramp", "merge_weave", "merge_trunk"),
    "inverted_tree": (
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
    ),
}


def safe_action() -> dict[str, str]:
    return {
        "desired_speed_bin": "nominal",
        "desired_headway_bin": "normal",
        "lane_preference": "keep",
        "merge_mode": "normal",
    }


@pytest.mark.parametrize("topology_id", TOPOLOGY_IDS)
def test_topology_factory_metadata(topology_id: str) -> None:
    spec = build_topology(topology_id)
    assert spec.topology_id == topology_id
    assert spec.segment_ids == EXPECTED_SEGMENTS[topology_id]
    assert set(spec.entry_segments).issubset(spec.segment_ids)
    assert set(spec.exit_segments).issubset(spec.segment_ids)
    assert set(spec.lane_counts) == set(spec.segment_ids)
    assert spec.lane_segments
    for segment_id in spec.segment_ids:
        assert spec.segment_lengths[segment_id] > 0
        assert spec.segment_edges[segment_id]
    for segment_id, locations in spec.detector_locations.items():
        assert segment_id in spec.segment_ids
        for location in locations:
            assert 0 <= location <= spec.segment_lengths[segment_id]


def test_expected_lane_counts_and_merge_nodes() -> None:
    assert build_topology("ring").lane_counts == {"ring_main": 1}
    assert build_topology("straight_single_lane").lane_counts["straight_upstream"] == 1
    assert build_topology("straight_multilane").lane_counts["straight_upstream"] == 3
    merge = build_topology("merge")
    assert merge.merge_nodes == ("merge_0",)
    assert merge.lane_counts["merge_main_upstream"] == 2
    assert merge.lane_counts["merge_ramp"] == 1
    tree = build_topology("inverted_tree")
    assert tree.merge_nodes == ("tree_merge_b1", "tree_merge_b2", "tree_merge_c")
    assert tree.lane_counts["tree_bottleneck_d"] == 1


@pytest.mark.parametrize("topology_id", TOPOLOGY_IDS)
def test_highway_topology_env_smoke(topology_id: str) -> None:
    env = HighwayTopologyEnv(topology_id, {"controlled_vehicles": 2, "duration_steps": 4})
    obs, info = env.reset(seed=7)
    assert info["topology_id"] == topology_id
    assert set(obs) == {"av_0", "av_1"}
    for local_obs in obs.values():
        assert "sensor" in local_obs
        assert "cooperation" in local_obs
        assert local_obs["is_active"] is True
    assert env.get_global_state()["topology_id"] == topology_id
    assert set(env.get_segment_metrics()) == set(EXPECTED_SEGMENTS[topology_id])

    actions = {agent_id: safe_action() for agent_id in obs}
    next_obs, rewards, terminated, truncated, step_info = env.step(actions)
    assert set(next_obs) == set(rewards)
    assert set(next_obs).issubset(set(env.agent_ids))
    assert isinstance(terminated, bool)
    assert isinstance(truncated, bool)
    assert step_info["topology_id"] == topology_id

    with pytest.raises(ValueError):
        env.step({"av_inactive": safe_action()})
