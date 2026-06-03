from __future__ import annotations

import argparse
import csv
import json

import torch

from scripts import validate_training_eval as validation


def args(**overrides):
    base = {
        "matrix": "smoke",
        "dry_run": False,
        "seed": 7,
        "total_updates": 1,
        "rollout_steps": 8,
        "duration_steps": 20,
        "eval_duration_steps": 12,
        "controlled_vehicles": 2,
        "initial_human_vehicles": 12,
        "device": "cpu",
        "checkpoint_root": "outputs/checkpoints",
        "learned_output_root": "outputs/metrics/learned",
        "baseline_output_root": "outputs/metrics/baselines",
        "validation_root": "outputs/validation/task11",
    }
    base.update(overrides)
    return argparse.Namespace(**base)


def write_csv(path, rows):
    fieldnames = sorted({field for row in rows for field in row})
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def test_build_training_matrix_smoke_covers_algorithms_and_topologies() -> None:
    matrix = validation.build_training_matrix("smoke", args())
    keys = {(spec.algorithm, spec.topology, spec.action_profile, spec.expected_critic_scope) for spec in matrix}
    assert ("shared_ppo", "ring", "speed_only", "local") in keys
    assert ("ippo", "straight_single_lane", "speed_headway", "local") in keys
    assert ("mappo", "merge", "full", "global") in keys
    assert ("mappo", "inverted_tree", "full", "global") in keys


def test_build_training_matrix_full_runs_every_algorithm_on_every_topology() -> None:
    matrix = validation.build_training_matrix("full", args())
    assert len(matrix) == len(validation.TRAINING_SETTINGS) * len(validation.TOPOLOGIES)
    assert {spec.topology for spec in matrix} == set(validation.TOPOLOGIES)
    assert {spec.algorithm for spec in matrix} == set(validation.TRAINING_SETTINGS)


def test_dry_run_rows_and_reports_are_written(tmp_path) -> None:
    rows = [
        validation.ValidationRow(
            category="training",
            name="shared_ppo_ring_speed_only_seed7",
            status="planned",
            command="python scripts/train_policy.py",
            output_dir=str(tmp_path / "checkpoints"),
        ),
        validation.ValidationRow(
            category="learned_eval",
            name="shared_ppo_ring_speed_only_seed7",
            status="planned",
            command="python scripts/evaluate_policy.py",
            output_dir=str(tmp_path / "learned"),
        ),
        validation.ValidationRow(
            category="baseline_eval",
            name="random_av_ring_medium_seed7",
            status="planned",
            command="python scripts/run_baseline.py",
            output_dir=str(tmp_path / "baselines"),
        ),
    ]
    validation.write_reports(tmp_path, rows)
    assert (tmp_path / "training_runs.csv").exists()
    assert (tmp_path / "learned_eval_runs.csv").exists()
    assert (tmp_path / "baseline_eval_runs.csv").exists()
    payload = json.loads((tmp_path / "training_eval_summary.json").read_text())
    assert payload["planned_count"] == 3
    assert (tmp_path / "validation_summary.md").exists()


def test_check_training_artifacts_validates_profile_scope_and_finite_metrics(tmp_path) -> None:
    spec = validation.TrainingSpec(
        algorithm="mappo",
        topology="merge",
        demand="high",
        seed=7,
        total_updates=1,
        rollout_steps=8,
        duration_steps=20,
        controlled_vehicles=2,
        initial_human_vehicles=12,
    )
    torch.save({"metadata": {"action_profile": "full"}, "state_dict": {}, "hidden_sizes": (16,)}, tmp_path / "actor.pt")
    torch.save({"scope": "global", "state_dict": {}, "input_dim": 1, "hidden_sizes": (16,)}, tmp_path / "critic.pt")
    (tmp_path / "config_resolved.yaml").write_text("training: {}\n")
    write_csv(tmp_path / "training_metrics.csv", [{"loss": 1.0, "score": 2.0}])
    assert validation.check_training_artifacts(tmp_path, spec) == []


def test_check_csv_finite_flags_nonfinite_numeric_values(tmp_path) -> None:
    path = tmp_path / "metrics.csv"
    write_csv(path, [{"loss": "nan", "label": "ok"}])
    failures = validation.check_csv_finite(path)
    assert failures
    assert "not finite" in failures[0]


def test_check_csv_finite_allows_known_metric_sentinels(tmp_path) -> None:
    path = tmp_path / "metrics.csv"
    write_csv(path, [{"min_lane_change_dwell_time": "inf", "rear_ttc_after_av_lane_change_min": "inf", "loss": 1.0}])
    assert validation.check_csv_finite(path) == []


def test_baseline_matrix_uses_separate_smoke_outputs() -> None:
    matrix = validation.build_baseline_matrix(args(seed=11, eval_duration_steps=5))
    assert {spec.controller for spec in matrix} == {
        "no_av",
        "random_av",
        "density_lookup",
        "backpressure",
        "cooperative_smoothing",
    }
    assert all(spec.seed == 11 for spec in matrix)
    assert all(spec.duration_steps == 5 for spec in matrix)
