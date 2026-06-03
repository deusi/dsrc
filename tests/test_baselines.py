from __future__ import annotations

import json

import pytest

from scripts import run_baseline
from src.baselines import BASELINE_NAMES, make_baseline
from src.envs.topology_env import HighwayTopologyEnv
from src.envs.wrappers import validate_action_mapping


def obs(**overrides):
    base = {
        "ego_speed": 20.0,
        "ego_headway_s": 2.0,
        "leader_gap": 50.0,
        "leader_relative_speed": 0.0,
        "left_lane_front_gap": 0.0,
        "left_lane_rear_gap": 0.0,
        "right_lane_front_gap": 0.0,
        "right_lane_rear_gap": 0.0,
        "local_density_bin": 0,
        "local_mean_speed_bin": 2,
        "local_queue_estimate": 0,
        "nearby_av_count": 0,
        "nearby_av_mean_speed": 30.0,
        "segment_target_speed": 30.0,
        "merge_pressure": 0.0,
        "downstream_congestion_estimate": 0.0,
        "distance_to_downstream_bottleneck": float("inf"),
        "cooperation": {
            "segment_target_speed": 30.0,
            "merge_pressure": 0.0,
            "downstream_congestion_estimate": 0.0,
        },
    }
    base.update(overrides)
    return base


def test_baseline_registry_exposes_canonical_names() -> None:
    assert BASELINE_NAMES == (
        "no_av",
        "random_av",
        "selfish_av",
        "density_lookup",
        "dynamic_speed_limit",
        "av_mediated_speed_harmonization",
        "backpressure",
        "cooperative_smoothing",
    )
    for name in BASELINE_NAMES:
        assert make_baseline(name).name == name


@pytest.mark.parametrize("name", BASELINE_NAMES)
def test_baselines_use_only_local_obs_and_return_valid_actions(name: str) -> None:
    controller = make_baseline(name)
    assert controller.metadata.requires_global_state is False
    local_obs = {"av_0": obs(), "av_1": obs(local_density_bin=2, local_queue_estimate=1)}
    actions = controller.act(local_obs)
    if name == "no_av":
        assert actions == {}
    else:
        assert set(actions) == set(local_obs)
        validate_action_mapping(actions, expected_agent_ids=local_obs.keys())
    with pytest.raises(ValueError):
        controller.act(local_obs, global_state={"segment_state": {}})


def test_random_baseline_is_seed_reproducible() -> None:
    a = make_baseline("random_av")
    b = make_baseline("random_av")
    a.reset(seed=11)
    b.reset(seed=11)
    local_obs = {"av_0": obs(), "av_1": obs()}
    assert a.act(local_obs) == b.act(local_obs)


def test_density_lookup_and_local_dynamic_speed_advisory_thresholds() -> None:
    density = make_baseline("density_lookup")
    dynamic = make_baseline("dynamic_speed_limit")
    low = {"av_0": obs(local_density_bin=0, local_queue_estimate=0, local_mean_speed_bin=2)}
    high = {"av_0": obs(local_density_bin=2, local_queue_estimate=1, local_mean_speed_bin=2)}

    assert density.act(low)["av_0"]["desired_speed_bin"] == "fast"
    assert density.act(high)["av_0"]["desired_speed_bin"] == "slow"
    assert dynamic.act(low)["av_0"]["desired_speed_bin"] == "fast"
    assert dynamic.act(high)["av_0"]["desired_speed_bin"] == "slow"


def test_speed_harmonization_reacts_to_speed_mismatch_not_density_only() -> None:
    controller = make_baseline("av_mediated_speed_harmonization")
    fast_ego = {"av_0": obs(ego_speed=30.0, nearby_av_count=1, nearby_av_mean_speed=20.0, local_density_bin=0)}
    slow_ego = {"av_0": obs(ego_speed=15.0, nearby_av_count=1, nearby_av_mean_speed=25.0, local_density_bin=0)}

    assert controller.act(fast_ego)["av_0"]["desired_speed_bin"] == "slow"
    assert controller.act(slow_ego)["av_0"]["desired_speed_bin"] == "fast"


def test_backpressure_and_cooperative_smoothing_local_pressure_behaviors() -> None:
    backpressure = make_baseline("backpressure")
    smoothing = make_baseline("cooperative_smoothing")
    pressure_obs = {
        "av_0": obs(
            merge_pressure=0.8,
            cooperation={"segment_target_speed": 30.0, "merge_pressure": 0.8, "downstream_congestion_estimate": 0.0},
        )
    }
    bottleneck_obs = {"av_0": obs(distance_to_downstream_bottleneck=50.0)}
    short_gap_obs = {"av_0": obs(leader_gap=10.0)}

    assert backpressure.act(pressure_obs)["av_0"]["merge_mode"] == "create_gap"
    assert backpressure.act(bottleneck_obs)["av_0"]["merge_mode"] == "hold_lane"
    assert smoothing.act(short_gap_obs)["av_0"]["desired_headway_bin"] == "largest"


def test_no_av_ring_initial_humans_have_no_public_agents() -> None:
    env = HighwayTopologyEnv(
        "ring",
        {"controlled_vehicles": 0, "initial_human_vehicles": 3, "duration_steps": 1},
    )
    observations, _ = env.reset(seed=5)
    assert observations == {}
    role_state = env.get_global_state()["vehicle_role_state"]
    assert role_state["active_av_count"] == 0
    assert role_state["active_human_count"] == 3


def test_run_baseline_writes_canonical_artifacts(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(
        run_baseline.sys,
        "argv",
        [
            "run_baseline.py",
            "--controller",
            "no_av",
            "--topology",
            "ring",
            "--duration-steps",
            "1",
            "--initial-human-vehicles",
            "2",
            "--output-root",
            str(tmp_path),
        ],
    )
    assert run_baseline.main() == 0
    output_dir = tmp_path / "no_av_ring_medium_seed7"
    assert (output_dir / "step_metrics.csv").exists()
    assert (output_dir / "segment_metrics.csv").exists()
    summary = json.loads((output_dir / "episode_summary.json").read_text())
    assert summary["controller"] == "no_av"
    assert summary["active_av_count"] == 0
