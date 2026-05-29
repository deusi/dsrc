from __future__ import annotations

import pytest

from src.demand import DemandSpawner, build_route_plan, load_demand_profile
from src.demand.route_sampler import road_route_to_destination
from src.envs.topology_env import HighwayTopologyEnv
from src.road.highway_imports import ensure_highway_env_importable
from src.road.topology_factory import build_topology

ensure_highway_env_importable()

from highway_env.road.road import Road


def action() -> dict[str, str]:
    return {
        "desired_speed_bin": "nominal",
        "desired_headway_bin": "normal",
        "lane_preference": "keep",
        "merge_mode": "normal",
    }


def demand_config(**overrides):
    cfg = {
        "id": "test",
        "total_vehicles_per_hour": 7200,
        "av_penetration": 0.5,
        "branch_split": {"main": 1.0},
        "burst": {"enabled": False},
        "speed_distribution": {"mean_mps": 24.0, "std_mps": 0.0, "min_mps": 12.0, "max_mps": 32.0},
        "spawn_min_gap_m": 0.0,
    }
    cfg.update(overrides)
    return cfg


def test_demand_profile_normalizes_branch_split_and_burst() -> None:
    profile = load_demand_profile(
        demand_config(
            total_vehicles_per_hour=1000,
            branch_split={"main": 2, "ramp": 1},
            burst={"enabled": True, "start_s": 10, "end_s": 20, "multiplier": 1.8},
        )
    )
    assert profile.branch_split == {"main": pytest.approx(2 / 3), "ramp": pytest.approx(1 / 3)}
    assert profile.vehicles_per_hour_at(9.9) == 1000
    assert profile.vehicles_per_hour_at(10.0) == 1800
    assert profile.vehicles_per_hour_at(20.0) == 1000


@pytest.mark.parametrize(
    "bad_cfg",
    [
        {"total_vehicles_per_hour": -1},
        {"av_penetration": 1.1},
        {"branch_split": {"main": 0}},
    ],
)
def test_demand_profile_rejects_invalid_values(bad_cfg) -> None:
    cfg = demand_config()
    cfg.update(bad_cfg)
    with pytest.raises(ValueError):
        load_demand_profile(cfg)


def test_route_plan_uses_one_way_shared_destinations_and_road_routes() -> None:
    straight = build_route_plan(build_topology("straight_multilane"))
    assert straight.enabled is True
    assert straight.destination == "s3"
    assert [branch.branch_id for branch in straight.branches] == ["main"]

    merge = build_route_plan(build_topology("merge"))
    assert merge.destination == "m3"
    assert {branch.branch_id for branch in merge.branches} == {"main", "ramp"}

    tree = build_route_plan(build_topology("inverted_tree"))
    assert tree.destination == "exit"
    assert {branch.branch_id for branch in tree.branches} == {"a1", "a2", "a3", "a4", "a5", "a6"}

    ring = build_route_plan(build_topology("ring"))
    assert ring.enabled is False
    assert ring.branches == ()

    route = road_route_to_destination(("s0", "s1", 2), "s3", build_topology("straight_multilane"))
    assert route == [("s0", "s1", None), ("s1", "s2", None), ("s2", "s3", None)]


def test_spawner_balances_lanes_within_selected_branch() -> None:
    topology = build_topology("straight_multilane")
    road = Road(network=topology.road_network, np_random=__import__("numpy").random.RandomState(7), record_history=False)
    route_plan = build_route_plan(topology)
    profile = load_demand_profile(demand_config(av_penetration=1.0), enabled=True)
    spawner = DemandSpawner(profile, route_plan, topology, road.np_random)
    branch = route_plan.branches[0]

    lanes = []
    for _ in range(6):
        vehicle = spawner._try_spawn_on_branch(road, branch, "av")
        assert vehicle is not None
        lanes.append(vehicle.lane_index[2])
        road.vehicles.clear()

    assert lanes == [0, 1, 2, 0, 1, 2]


def test_env_demand_is_reproducible_and_tracks_lifecycle() -> None:
    cfg = {
        "duration_steps": 8,
        "dt": 5.0,
        "demand": demand_config(total_vehicles_per_hour=36000, av_penetration=1.0),
    }
    env_a = HighwayTopologyEnv("straight_single_lane", cfg)
    env_b = HighwayTopologyEnv("straight_single_lane", cfg)
    obs_a, _ = env_a.reset(seed=11)
    obs_b, _ = env_b.reset(seed=11)

    states_a = []
    states_b = []
    for _ in range(8):
        obs_a, _, _, _, _ = env_a.step({agent_id: action() for agent_id in obs_a})
        obs_b, _, _, _, _ = env_b.step({agent_id: action() for agent_id in obs_b})
        states_a.append(env_a.get_global_state()["demand_state"])
        states_b.append(env_b.get_global_state()["demand_state"])

    assert states_a == states_b
    assert states_a[-1]["spawned_av_count"] == states_a[-1]["spawned_vehicle_count"]
    assert states_a[-1]["spawned_vehicle_count"] > 0
    assert env_a.get_segment_metrics()["straight_upstream"]["inflow"] >= 0


def test_zero_av_penetration_has_no_public_agents_but_counts_humans() -> None:
    env = HighwayTopologyEnv(
        "straight_single_lane",
        {"duration_steps": 4, "dt": 5.0, "demand": demand_config(total_vehicles_per_hour=36000, av_penetration=0.0)},
    )
    obs, _ = env.reset(seed=3)
    assert obs == {}
    for _ in range(4):
        obs, rewards, _, _, _ = env.step({})
        assert obs == {}
        assert rewards == {}

    demand_state = env.get_global_state()["demand_state"]
    assert demand_state["spawned_human_count"] == demand_state["spawned_vehicle_count"]
    assert demand_state["spawned_vehicle_count"] > 0


def test_stale_av_action_is_rejected_after_exit() -> None:
    env = HighwayTopologyEnv(
        "straight_single_lane",
        {
            "duration_steps": 30,
            "dt": 10.0,
            "demand": demand_config(total_vehicles_per_hour=72000, av_penetration=1.0),
            "max_speed_mps": 40.0,
        },
    )
    obs, _ = env.reset(seed=5)
    stale_id = None
    for _ in range(30):
        previous_ids = set(obs)
        obs, _, _, _, _ = env.step({agent_id: action() for agent_id in obs})
        exited = previous_ids - set(obs)
        if exited:
            stale_id = sorted(exited)[0]
            break
    if stale_id is None:
        pytest.skip("no AV exited during deterministic smoke horizon")

    with pytest.raises(ValueError):
        env.step({**{agent_id: action() for agent_id in obs}, stale_id: action()})
