#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
import shlex
import subprocess
import sys
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.baselines import make_baseline


TOPOLOGIES = ("ring", "straight_single_lane", "straight_multilane", "merge", "inverted_tree")
TRAINING_SETTINGS = {
    "shared_ppo": {"action_profile": "speed_only", "critic_scope": "local"},
    "ippo": {"action_profile": "speed_headway", "critic_scope": "local"},
    "mappo": {"action_profile": "full", "critic_scope": "global"},
}
SMOKE_TRAINING_SPECS = (
    ("shared_ppo", "ring"),
    ("ippo", "straight_single_lane"),
    ("mappo", "merge"),
    ("mappo", "inverted_tree"),
)
BASELINE_EVAL_SPECS = (
    ("no_av", "straight_single_lane", "high"),
    ("random_av", "ring", "medium"),
    ("density_lookup", "straight_single_lane", "high"),
    ("backpressure", "merge", "high"),
    ("cooperative_smoothing", "inverted_tree", "high"),
)
NONFINITE_SENTINEL_FIELDS = {
    "min_lane_change_dwell_time",
    "rear_ttc_after_av_lane_change_min",
}


@dataclass(frozen=True)
class TrainingSpec:
    algorithm: str
    topology: str
    demand: str
    seed: int
    total_updates: int
    rollout_steps: int
    duration_steps: int
    controlled_vehicles: int
    initial_human_vehicles: int

    @property
    def action_profile(self) -> str:
        return str(TRAINING_SETTINGS[self.algorithm]["action_profile"])

    @property
    def expected_critic_scope(self) -> str:
        return str(TRAINING_SETTINGS[self.algorithm]["critic_scope"])

    @property
    def experiment_id(self) -> str:
        return f"{self.algorithm}_{self.topology}_{self.action_profile}_seed{self.seed}"


@dataclass(frozen=True)
class BaselineEvalSpec:
    controller: str
    topology: str
    demand: str
    seed: int
    duration_steps: int
    controlled_vehicles: int
    initial_human_vehicles: int

    @property
    def experiment_id(self) -> str:
        return f"{self.controller}_{self.topology}_{self.demand}_seed{self.seed}"


@dataclass
class ValidationRow:
    category: str
    name: str
    status: str
    command: str
    output_dir: str
    detail: str = ""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate DSRC RL training and evaluation entrypoints.")
    parser.add_argument("--matrix", choices=("smoke", "full"), default="smoke")
    parser.add_argument("--dry-run", action="store_true", help="Write and print the planned commands without running them.")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--total-updates", type=int, default=1)
    parser.add_argument("--rollout-steps", type=int, default=8)
    parser.add_argument("--duration-steps", type=int, default=20)
    parser.add_argument("--eval-duration-steps", type=int, default=12)
    parser.add_argument("--controlled-vehicles", type=int, default=2)
    parser.add_argument("--initial-human-vehicles", type=int, default=12)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--checkpoint-root", default="outputs/checkpoints")
    parser.add_argument("--learned-output-root", default="outputs/metrics/learned")
    parser.add_argument("--baseline-output-root", default="outputs/metrics/baselines")
    parser.add_argument("--validation-root", default="outputs/validation/task11")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    validation_root = Path(args.validation_root)
    training_specs = build_training_matrix(args.matrix, args)
    baseline_specs = build_baseline_matrix(args)
    rows: list[ValidationRow] = []

    for spec in training_specs:
        train_row = run_training(spec, args)
        rows.append(train_row)
        if train_row.status == "pass" and not args.dry_run:
            rows.append(run_learned_evaluation(spec, args))
        elif args.dry_run:
            rows.append(planned_learned_evaluation(spec, args))

    for spec in baseline_specs:
        rows.append(run_baseline_evaluation(spec, args))

    write_reports(validation_root, rows)
    for row in rows:
        print(f"{row.status}: {row.category} {row.name} -> {row.output_dir}")
        if args.dry_run:
            print(f"  {row.command}")
    failures = [row for row in rows if row.status == "fail"]
    print(f"report: {validation_root / 'validation_summary.md'}")
    return 1 if failures else 0


def build_training_matrix(matrix: str, args: argparse.Namespace) -> list[TrainingSpec]:
    pairs = SMOKE_TRAINING_SPECS
    if matrix == "full":
        pairs = tuple((algorithm, topology) for algorithm in TRAINING_SETTINGS for topology in TOPOLOGIES)
    return [
        TrainingSpec(
            algorithm=algorithm,
            topology=topology,
            demand=demand_for_topology(topology),
            seed=args.seed,
            total_updates=args.total_updates,
            rollout_steps=args.rollout_steps,
            duration_steps=args.duration_steps,
            controlled_vehicles=args.controlled_vehicles,
            initial_human_vehicles=args.initial_human_vehicles,
        )
        for algorithm, topology in pairs
    ]


def build_baseline_matrix(args: argparse.Namespace) -> list[BaselineEvalSpec]:
    return [
        BaselineEvalSpec(
            controller=controller,
            topology=topology,
            demand=demand,
            seed=args.seed,
            duration_steps=args.eval_duration_steps,
            controlled_vehicles=args.controlled_vehicles,
            initial_human_vehicles=args.initial_human_vehicles,
        )
        for controller, topology, demand in BASELINE_EVAL_SPECS
    ]


def demand_for_topology(topology: str) -> str:
    if topology == "ring":
        return "medium"
    return "high"


def run_training(spec: TrainingSpec, args: argparse.Namespace) -> ValidationRow:
    output_dir = Path(args.checkpoint_root) / spec.experiment_id
    command = training_command(spec, args)
    if args.dry_run:
        return ValidationRow("training", spec.experiment_id, "planned", _command_string(command), str(output_dir))
    completed = run_command(command)
    failures = []
    if completed.returncode != 0:
        failures.append(f"command exited {completed.returncode}: {completed.stderr.strip() or completed.stdout.strip()}")
    failures.extend(check_training_artifacts(output_dir, spec))
    return ValidationRow(
        "training",
        spec.experiment_id,
        "fail" if failures else "pass",
        _command_string(command),
        str(output_dir),
        "; ".join(failures),
    )


def run_learned_evaluation(spec: TrainingSpec, args: argparse.Namespace) -> ValidationRow:
    output_root = Path(args.learned_output_root) / spec.experiment_id
    command = learned_eval_command(spec, args, output_root)
    completed = run_command(command)
    failures = []
    if completed.returncode != 0:
        failures.append(f"command exited {completed.returncode}: {completed.stderr.strip() or completed.stdout.strip()}")
    failures.extend(check_learned_eval_artifacts(output_root, spec))
    return ValidationRow(
        "learned_eval",
        spec.experiment_id,
        "fail" if failures else "pass",
        _command_string(command),
        str(output_root),
        "; ".join(failures),
    )


def planned_learned_evaluation(spec: TrainingSpec, args: argparse.Namespace) -> ValidationRow:
    output_root = Path(args.learned_output_root) / spec.experiment_id
    command = learned_eval_command(spec, args, output_root)
    return ValidationRow("learned_eval", spec.experiment_id, "planned", _command_string(command), str(output_root))


def run_baseline_evaluation(spec: BaselineEvalSpec, args: argparse.Namespace) -> ValidationRow:
    output_root = Path(args.baseline_output_root)
    command = baseline_eval_command(spec, args, output_root)
    if args.dry_run:
        return ValidationRow("baseline_eval", spec.experiment_id, "planned", _command_string(command), str(output_root / spec.experiment_id))
    completed = run_command(command)
    failures = []
    if completed.returncode != 0:
        failures.append(f"command exited {completed.returncode}: {completed.stderr.strip() or completed.stdout.strip()}")
    failures.extend(check_baseline_eval_artifacts(output_root / spec.experiment_id, spec))
    return ValidationRow(
        "baseline_eval",
        spec.experiment_id,
        "fail" if failures else "pass",
        _command_string(command),
        str(output_root / spec.experiment_id),
        "; ".join(failures),
    )


def training_command(spec: TrainingSpec, args: argparse.Namespace) -> list[str]:
    return [
        sys.executable,
        "scripts/train_policy.py",
        "--training",
        spec.algorithm,
        "--topology",
        spec.topology,
        "--demand",
        spec.demand,
        "--seed",
        str(spec.seed),
        "--total-updates",
        str(spec.total_updates),
        "--rollout-steps",
        str(spec.rollout_steps),
        "--duration-steps",
        str(spec.duration_steps),
        "--controlled-vehicles",
        str(spec.controlled_vehicles),
        "--initial-human-vehicles",
        str(spec.initial_human_vehicles),
        "--device",
        args.device,
        "--output-root",
        args.checkpoint_root,
    ]


def learned_eval_command(spec: TrainingSpec, args: argparse.Namespace, output_root: Path) -> list[str]:
    actor_path = Path(args.checkpoint_root) / spec.experiment_id / "actor.pt"
    return [
        sys.executable,
        "scripts/evaluate_policy.py",
        "--actor",
        str(actor_path),
        "--topology",
        spec.topology,
        "--demand",
        spec.demand,
        "--seed",
        str(spec.seed),
        "--duration-steps",
        str(args.eval_duration_steps),
        "--controlled-vehicles",
        str(spec.controlled_vehicles),
        "--initial-human-vehicles",
        str(spec.initial_human_vehicles),
        "--device",
        args.device,
        "--output-root",
        str(output_root),
    ]


def baseline_eval_command(spec: BaselineEvalSpec, args: argparse.Namespace, output_root: Path) -> list[str]:
    return [
        sys.executable,
        "scripts/run_baseline.py",
        "--controller",
        spec.controller,
        "--topology",
        spec.topology,
        "--demand",
        spec.demand,
        "--seed",
        str(spec.seed),
        "--duration-steps",
        str(spec.duration_steps),
        "--controlled-vehicles",
        str(spec.controlled_vehicles),
        "--initial-human-vehicles",
        str(spec.initial_human_vehicles),
        "--output-root",
        str(output_root),
    ]


def run_command(command: Sequence[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(command, cwd=REPO_ROOT, text=True, capture_output=True, check=False)


def check_training_artifacts(output_dir: Path, spec: TrainingSpec) -> list[str]:
    import torch

    failures = require_files(output_dir, ("actor.pt", "critic.pt", "config_resolved.yaml", "training_metrics.csv"))
    if failures:
        return failures
    actor_payload = torch.load(output_dir / "actor.pt", map_location="cpu")
    critic_payload = torch.load(output_dir / "critic.pt", map_location="cpu")
    metadata = actor_payload.get("metadata", {}) if isinstance(actor_payload, Mapping) else {}
    action_profile = metadata.get("action_profile") if isinstance(metadata, Mapping) else None
    if action_profile != spec.action_profile:
        failures.append(f"actor action_profile={action_profile!r}, expected {spec.action_profile!r}")
    critic_scope = critic_payload.get("scope") if isinstance(critic_payload, Mapping) else None
    if critic_scope != spec.expected_critic_scope:
        failures.append(f"critic scope={critic_scope!r}, expected {spec.expected_critic_scope!r}")
    failures.extend(check_csv_finite(output_dir / "training_metrics.csv"))
    return failures


def check_learned_eval_artifacts(output_root: Path, spec: TrainingSpec) -> list[str]:
    run_dir = output_root / f"learned_policy_{spec.topology}_{spec.demand}_seed{spec.seed}"
    failures = require_files(run_dir, ("episode_summary.json", "step_metrics.csv", "segment_metrics.csv"))
    if failures:
        return failures
    summary = json.loads((run_dir / "episode_summary.json").read_text())
    if summary.get("controller") != "learned_policy":
        failures.append(f"learned eval controller={summary.get('controller')!r}")
    actor_path = Path(summary.get("actor", ""))
    if actor_path.exists():
        from src.rl.controller import LearnedPolicyController

        controller = LearnedPolicyController.from_checkpoint(str(actor_path))
        if controller.metadata.requires_global_state:
            failures.append("learned eval controller unexpectedly requires global_state")
    else:
        failures.append(f"learned eval actor path missing: {actor_path}")
    failures.extend(check_csv_finite(run_dir / "step_metrics.csv"))
    failures.extend(check_csv_finite(run_dir / "segment_metrics.csv"))
    return failures


def check_baseline_eval_artifacts(run_dir: Path, spec: BaselineEvalSpec) -> list[str]:
    failures = require_files(run_dir, ("episode_summary.json", "step_metrics.csv", "segment_metrics.csv"))
    if failures:
        return failures
    summary = json.loads((run_dir / "episode_summary.json").read_text())
    if summary.get("controller") != spec.controller:
        failures.append(f"baseline summary controller={summary.get('controller')!r}, expected {spec.controller!r}")
    if make_baseline(spec.controller).metadata.requires_global_state:
        failures.append(f"{spec.controller} unexpectedly requires global_state")
    if spec.controller == "no_av":
        if int(summary.get("active_av_count", 0)) != 0:
            failures.append("no_av produced active AVs")
        if int(summary.get("spawned_vehicle_count", 0)) <= 0 and int(summary.get("active_vehicle_count", 0)) <= 0:
            failures.append("no_av produced no human traffic")
    failures.extend(check_csv_finite(run_dir / "step_metrics.csv"))
    failures.extend(check_csv_finite(run_dir / "segment_metrics.csv"))
    return failures


def require_files(directory: Path, filenames: Iterable[str]) -> list[str]:
    return [f"missing {directory / filename}" for filename in filenames if not (directory / filename).exists()]


def check_csv_finite(path: Path) -> list[str]:
    failures: list[str] = []
    with path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        for row_index, row in enumerate(reader, start=1):
            for field, value in row.items():
                if field in NONFINITE_SENTINEL_FIELDS:
                    continue
                if value in ("", None):
                    continue
                try:
                    numeric = float(value)
                except (TypeError, ValueError):
                    continue
                if not math.isfinite(numeric):
                    failures.append(f"{path} row {row_index} field {field} is not finite")
    return failures


def write_reports(output_root: Path, rows: Sequence[ValidationRow]) -> None:
    output_root.mkdir(parents=True, exist_ok=True)
    write_category_csv(output_root / "training_runs.csv", [row for row in rows if row.category == "training"])
    write_category_csv(output_root / "learned_eval_runs.csv", [row for row in rows if row.category == "learned_eval"])
    write_category_csv(output_root / "baseline_eval_runs.csv", [row for row in rows if row.category == "baseline_eval"])
    payload = {
        "run_count": len(rows),
        "failure_count": sum(1 for row in rows if row.status == "fail"),
        "planned_count": sum(1 for row in rows if row.status == "planned"),
        "rows": [row.__dict__ for row in rows],
    }
    (output_root / "training_eval_summary.json").write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    write_markdown_summary(output_root / "validation_summary.md", rows)


def write_category_csv(path: Path, rows: Sequence[ValidationRow]) -> None:
    with path.open("w", newline="") as handle:
        fieldnames = ["category", "name", "status", "output_dir", "command", "detail"]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: getattr(row, field) for field in fieldnames})


def write_markdown_summary(path: Path, rows: Sequence[ValidationRow]) -> None:
    failures = [row for row in rows if row.status == "fail"]
    lines = [
        "# Task 11 Training And Evaluation Validation",
        "",
        f"- Runs: {len(rows)}",
        f"- Passed: {sum(1 for row in rows if row.status == 'pass')}",
        f"- Planned: {sum(1 for row in rows if row.status == 'planned')}",
        f"- Failed: {len(failures)}",
        "",
        "## Failures",
        "",
    ]
    if not failures:
        lines.append("None.")
    else:
        for row in failures:
            lines.append(f"- {row.category} / {row.name}: {row.detail}")
    lines.extend(["", "## Runs", ""])
    for row in rows:
        lines.append(f"- {row.status}: {row.category} / {row.name} -> `{row.output_dir}`")
    path.write_text("\n".join(lines) + "\n")


def _command_string(command: Sequence[str]) -> str:
    return " ".join(shlex.quote(part) for part in command)


if __name__ == "__main__":
    raise SystemExit(main())
