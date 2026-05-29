from __future__ import annotations

import numpy as np
import pytest

from src.config.loaders import load_named_config
from src.envs.topology_env import HighwayTopologyEnv
from src.road.highway_imports import ensure_highway_env_importable
from src.road.topology_factory import build_topology
from src.vehicles import apply_human_behavior_profile, load_human_behavior_model

ensure_highway_env_importable()

from highway_env.road.road import Road
from highway_env.vehicle.behavior import IDMVehicle


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
        "total_vehicles_per_hour": 36000,
        "av_penetration": 0.0,
        "branch_split": {"main": 1.0},
        "burst": {"enabled": False},
        "speed_distribution": {"mean_mps": 24.0, "std_mps": 0.0, "min_mps": 12.0, "max_mps": 32.0},
        "spawn_min_gap_m": 0.0,
    }
    cfg.update(overrides)
    return cfg


def make_idm_vehicle() -> IDMVehicle:
    topology = build_topology("straight_multilane")
    road = Road(network=topology.road_network, np_random=np.random.RandomState(1), record_history=False)
    return IDMVehicle.make_on_lane(road, ("s0", "s1", 0), longitudinal=0.0, speed=24.0)


@pytest.mark.parametrize("profile_id", ["cautious", "normal", "aggressive", "heterogeneous"])
def test_human_model_configs_load(profile_id: str) -> None:
    model = load_human_behavior_model(load_named_config("human_model", profile_id))
    assert set(model.weights).issubset({"cautious", "normal", "aggressive"})
    assert sum(model.weights.values()) == pytest.approx(1.0)


def test_heterogeneous_weights_normalize_and_sample_known_profiles() -> None:
    model = load_human_behavior_model({"id": "heterogeneous", "mix": {"cautious": 3, "normal": 5, "aggressive": 2}})
    assert model.weights == {"cautious": pytest.approx(0.3), "normal": pytest.approx(0.5), "aggressive": pytest.approx(0.2)}
    rng = np.random.RandomState(10)
    assert {model.sample_profile_id(rng) for _ in range(50)} <= {"cautious", "normal", "aggressive"}


@pytest.mark.parametrize(
    "bad_cfg",
    [
        {"id": "heterogeneous", "mix": {"unknown": 1.0}},
        {"id": "heterogeneous", "mix": {"normal": -1.0}},
        {"id": "heterogeneous", "mix": {"normal": 0.0}},
    ],
)
def test_human_model_rejects_invalid_mixtures(bad_cfg) -> None:
    with pytest.raises(ValueError):
        load_human_behavior_model(bad_cfg)


def test_behavior_application_changes_idm_mobil_parameters() -> None:
    normal = load_human_behavior_model(load_named_config("human_model", "normal")).profile_for("normal")
    cautious = load_human_behavior_model(load_named_config("human_model", "cautious")).profile_for("cautious")
    aggressive = load_human_behavior_model(load_named_config("human_model", "aggressive")).profile_for("aggressive")

    cautious_vehicle = make_idm_vehicle()
    normal_vehicle = make_idm_vehicle()
    aggressive_vehicle = make_idm_vehicle()
    apply_human_behavior_profile(cautious_vehicle, cautious, base_target_speed_mps=24.0, min_speed_mps=12.0, max_speed_mps=32.0)
    apply_human_behavior_profile(normal_vehicle, normal, base_target_speed_mps=24.0, min_speed_mps=12.0, max_speed_mps=32.0)
    apply_human_behavior_profile(aggressive_vehicle, aggressive, base_target_speed_mps=24.0, min_speed_mps=12.0, max_speed_mps=32.0)

    assert cautious_vehicle.TIME_WANTED > normal_vehicle.TIME_WANTED
    assert cautious_vehicle.LANE_CHANGE_DELAY > normal_vehicle.LANE_CHANGE_DELAY
    assert aggressive_vehicle.TIME_WANTED < normal_vehicle.TIME_WANTED
    assert aggressive_vehicle.LANE_CHANGE_DELAY < normal_vehicle.LANE_CHANGE_DELAY
    assert cautious_vehicle.TIME_WANTED != aggressive_vehicle.TIME_WANTED
    assert cautious_vehicle.behavior_profile == "cautious"
    assert aggressive_vehicle.behavior_profile == "aggressive"


@pytest.mark.parametrize("profile_id", ["cautious", "normal", "aggressive"])
def test_env_assigns_single_human_profile(profile_id: str) -> None:
    env = HighwayTopologyEnv(
        "straight_single_lane",
        {
            "duration_steps": 4,
            "dt": 5.0,
            "demand": demand_config(av_penetration=0.0),
            "human_model": load_named_config("human_model", profile_id),
        },
    )
    obs, _ = env.reset(seed=3)
    assert obs == {}
    for _ in range(4):
        obs, rewards, _, _, info = env.step({})
        assert obs == {}
        assert rewards == {}
        assert {event["behavior_profile"] for event in info["demand"]["spawned"]} <= {profile_id}

    role_state = env.get_global_state()["vehicle_role_state"]
    assert role_state["spawned_human_by_profile"][profile_id] == env.get_global_state()["demand_state"]["spawned_human_count"]


def test_env_heterogeneous_profile_sampling_is_reproducible() -> None:
    cfg = {
        "duration_steps": 8,
        "dt": 5.0,
        "demand": demand_config(av_penetration=0.0),
        "human_model": load_named_config("human_model", "heterogeneous"),
    }
    env_a = HighwayTopologyEnv("straight_single_lane", cfg)
    env_b = HighwayTopologyEnv("straight_single_lane", cfg)
    env_a.reset(seed=23)
    env_b.reset(seed=23)

    states_a = []
    states_b = []
    for _ in range(8):
        env_a.step({})
        env_b.step({})
        states_a.append(env_a.get_global_state()["vehicle_role_state"])
        states_b.append(env_b.get_global_state()["vehicle_role_state"])

    assert states_a == states_b
    assert sum(states_a[-1]["spawned_human_by_profile"].values()) > 0


def test_av_only_demand_has_no_human_profile_counts_and_public_agents_work() -> None:
    env = HighwayTopologyEnv(
        "straight_single_lane",
        {
            "duration_steps": 4,
            "dt": 5.0,
            "demand": demand_config(av_penetration=1.0),
            "human_model": load_named_config("human_model", "aggressive"),
        },
    )
    obs, _ = env.reset(seed=5)
    for _ in range(4):
        obs, rewards, _, _, _ = env.step({agent_id: action() for agent_id in obs})
        assert set(rewards) == set(obs)

    role_state = env.get_global_state()["vehicle_role_state"]
    assert role_state["spawned_human_by_profile"] == {}
    assert env.get_global_state()["demand_state"]["spawned_human_count"] == 0
