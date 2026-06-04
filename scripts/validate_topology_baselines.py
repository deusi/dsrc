#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.run_baseline import build_config
from src.baselines import BASELINE_NAMES, make_baseline
from src.envs.topology_env import HighwayTopologyEnv
from src.metrics import MetricsLogger
from src.road.topology_factory import TOPOLOGY_IDS, build_topology


DEFAULT_BASELINES = tuple(BASELINE_NAMES)
DEFAULT_DEMANDS = ("low", "high", "burst")
DEFAULT_SEEDS = (7, 11, 13, 17, 19)
KNOWN_DIAGNOSTICS = {
    "safety_masked_action",
    "etiquette_blocked_action",
    "follower_disruption_blocked",
    "external_safety_override",
    "simulator_blocked_action",
}


@dataclass(frozen=True)
class RunSpec:
    topology: str
    baseline: str
    demand: str
    seed: int
    duration_steps: int
    dt: float
    av_penetration: float | None

    @property
    def experiment_id(self) -> str:
        return f"{self.baseline}_{self.topology}_{self.demand}_seed{self.seed}"


@dataclass
class RunResult:
    spec: RunSpec
    paths: dict[str, str]
    summary: dict[str, Any]
    step_rows: list[dict[str, Any]]
    segment_rows: list[dict[str, Any]]
    hard_failures: list[str]
    notes: list[str]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate DSRC topology/baseline behavior.")
    parser.add_argument("--topologies", default=",".join(TOPOLOGY_IDS))
    parser.add_argument("--controllers", default=",".join(DEFAULT_BASELINES))
    parser.add_argument("--demands", default=",".join(DEFAULT_DEMANDS))
    parser.add_argument("--seeds", default=",".join(str(seed) for seed in DEFAULT_SEEDS))
    parser.add_argument("--duration-steps", type=int, default=180)
    parser.add_argument("--burst-duration-steps", type=int, default=720)
    parser.add_argument("--dt", type=float, default=1.0)
    parser.add_argument("--av-penetration", type=float, default=None)
    parser.add_argument("--initial-human-vehicles", type=int, default=12)
    parser.add_argument("--controlled-vehicles", type=int, default=2)
    parser.add_argument("--output-root", default="outputs/validation/task10")
    parser.add_argument("--smoke", action="store_true", help="Run one short seed/demand pass for fast validation.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    topologies = _csv_values(args.topologies)
    controllers = _csv_values(args.controllers)
    demands = _csv_values(args.demands)
    seeds = tuple(int(value) for value in _csv_values(args.seeds))
    if args.smoke:
        seeds = seeds[:1]
        demands = tuple(demand for demand in demands if demand == "high") or demands[:1]
        args.duration_steps = min(args.duration_steps, 30)
        args.burst_duration_steps = min(args.burst_duration_steps, 60)

    output_root = Path(args.output_root)
    run_results: list[RunResult] = []
    for spec in build_matrix(topologies, controllers, demands, seeds, args):
        run_results.append(run_one(spec, output_root, args))

    directional_rows = directional_checks(run_results)
    write_reports(output_root, run_results, directional_rows)
    hard_failure_count = sum(len(result.hard_failures) for result in run_results)
    warn_count = sum(1 for row in directional_rows if row["status"] == "warn")
    insufficient_count = sum(1 for row in directional_rows if row["status"] == "insufficient_signal")
    print(f"task10 validation runs: {len(run_results)}")
    print(f"hard failures: {hard_failure_count}")
    print(f"directional warnings: {warn_count}")
    print(f"insufficient signal: {insufficient_count}")
    print(f"report: {output_root / 'validation_summary.md'}")
    return 1 if hard_failure_count else 0


def build_matrix(
    topologies: Sequence[str],
    controllers: Sequence[str],
    demands: Sequence[str],
    seeds: Sequence[int],
    args: argparse.Namespace,
) -> list[RunSpec]:
    matrix: list[RunSpec] = []
    for topology in topologies:
        for controller in controllers:
            for demand in demands:
                for seed in seeds:
                    duration = args.burst_duration_steps if demand == "burst" else args.duration_steps
                    matrix.append(
                        RunSpec(
                            topology=topology,
                            baseline=controller,
                            demand=demand,
                            seed=seed,
                            duration_steps=duration,
                            dt=args.dt,
                            av_penetration=args.av_penetration,
                        )
                    )
    return matrix


def run_one(spec: RunSpec, output_root: Path, args: argparse.Namespace) -> RunResult:
    controller = make_baseline(spec.baseline)
    controller.reset(
        env_metadata={"topology_id": spec.topology, "demand": spec.demand},
        seed=spec.seed,
    )
    config = validation_config(spec, args)
    env = HighwayTopologyEnv(spec.topology, config)
    observations, reset_info = env.reset(seed=spec.seed)
    logger = MetricsLogger(experiment_id=spec.experiment_id, output_root=output_root / "runs")
    hard_failures: list[str] = []
    notes: list[str] = []
    step_rows: list[dict[str, Any]] = []
    segment_rows: list[dict[str, Any]] = []
    previous_completed = 0
    terminated = False
    truncated = False
    steps = 0

    while not (terminated or truncated):
        try:
            if controller.metadata.requires_global_state:
                hard_failures.append("baseline unexpectedly requires global_state")
            actions = controller.act(observations, global_state=None)
            action_counts = _action_counts(actions)
            expected_ids = set(observations)
            actual_ids = set(actions)
            if spec.baseline == "no_av":
                if actions:
                    hard_failures.append("no_av returned public AV actions")
            elif actual_ids != expected_ids:
                hard_failures.append(f"action keys {sorted(actual_ids)} did not match active AV ids {sorted(expected_ids)}")
            observations, _, terminated, truncated, info = env.step(actions)
        except Exception as exc:  # noqa: BLE001
            hard_failures.append(f"run raised {type(exc).__name__}: {exc}")
            break

        steps += 1
        metrics = dict(info.get("metrics", {}))
        segment_metrics = env.get_segment_metrics()
        row = {"step": steps, **metrics, **action_counts}
        step_rows.append(row)
        logger.record_step(row)
        logger.record_segments(time_s=float(info.get("time", 0.0)), segment_metrics=segment_metrics)
        for segment_id, segment in segment_metrics.items():
            segment_rows.append({"step": steps, "time": float(info.get("time", 0.0)), "segment_id": segment_id, **segment})

        active_from_segments = sum(int(segment.get("vehicle_count", 0)) for segment in segment_metrics.values())
        active_from_global = int(env.get_global_state().get("active_vehicle_count", 0))
        if active_from_segments != active_from_global:
            hard_failures.append(f"active vehicle mismatch at step {steps}: segments={active_from_segments} global={active_from_global}")
        completed = int(env.get_global_state().get("completed_vehicle_count", 0))
        if completed < previous_completed:
            hard_failures.append(f"completed_vehicle_count decreased at step {steps}")
        previous_completed = completed
        hard_failures.extend(_segment_invariant_failures(steps, segment_metrics))
        diagnostics = info.get("diagnostics", {})
        if isinstance(diagnostics, Mapping):
            unknown = set(diagnostics) - KNOWN_DIAGNOSTICS
            if unknown:
                hard_failures.append(f"unknown diagnostics at step {steps}: {sorted(unknown)}")

    summary = {
        **env.get_episode_summary(),
        "controller": spec.baseline,
        "demand": spec.demand,
        "seed": spec.seed,
        "reset_info": reset_info,
        "validation": {
            "topology": spec.topology,
            "duration_steps": spec.duration_steps,
            "hard_failures": hard_failures,
            "notes": notes,
        },
    }
    hard_failures.extend(topology_hard_failures(spec, summary, step_rows, segment_rows, config))
    summary["validation"]["hard_failures"] = hard_failures
    paths = logger.write_episode(summary)
    return RunResult(spec, paths, summary, step_rows, segment_rows, hard_failures, notes)


def validation_config(spec: RunSpec, args: argparse.Namespace) -> dict[str, Any]:
    baseline_args = argparse.Namespace(
        controller=spec.baseline,
        topology=spec.topology,
        demand=spec.demand,
        human_model="normal",
        av_penetration=spec.av_penetration,
        seed=spec.seed,
        duration_steps=spec.duration_steps,
        dt=spec.dt,
        controlled_vehicles=args.controlled_vehicles,
        initial_human_vehicles=args.initial_human_vehicles,
        output_root=str(args.output_root),
    )
    config = build_config(baseline_args)
    if spec.topology == "ring":
        config["initial_human_vehicles"] = args.initial_human_vehicles
    if spec.topology in {"merge", "inverted_tree", "inverted_tree_bottleneck"}:
        demand = dict(config["demand"])
        branches = _branch_ids_for_topology(spec.topology)
        demand["branch_split"] = {branch_id: 1.0 for branch_id in branches}
        config["demand"] = demand
    return config


def topology_hard_failures(
    spec: RunSpec,
    summary: Mapping[str, Any],
    step_rows: Sequence[Mapping[str, Any]],
    segment_rows: Sequence[Mapping[str, Any]],
    config: Mapping[str, Any],
) -> list[str]:
    failures: list[str] = []
    demand_state = summary
    active_branches = _branch_ids_for_topology(spec.topology)
    if spec.topology == "ring":
        if int(summary.get("completed_vehicle_count", 0)) != 0:
            failures.append("ring validation completed/exited vehicles")
        role_state = summary.get("reset_info", {})
        if spec.baseline == "no_av" and int(summary.get("active_av_count", 0)) != 0:
            failures.append("no_av ring had active AVs")
        if spec.baseline == "no_av" and int(summary.get("active_vehicle_count", 0)) <= 0:
            failures.append("no_av ring had no initial human vehicles")
        return failures

    if spec.demand == "high" and int(demand_state.get("spawned_vehicle_count", 0)) <= 0:
        failures.append("high demand spawned no vehicles")
    if spec.demand == "high" and spec.duration_steps >= 120 and int(summary.get("completed_vehicle_count", 0)) <= 0:
        failures.append("high demand produced no completed/exited vehicles")
    if spec.topology in {"merge", "inverted_tree", "inverted_tree_bottleneck"} and spec.demand in {"high", "burst"}:
        spawned = demand_state.get("per_branch_spawned", {})
        if isinstance(spawned, Mapping):
            missing = [branch_id for branch_id in active_branches if int(spawned.get(branch_id, 0)) <= 0]
            if missing:
                failures.append(f"branches spawned no vehicles: {missing}")
            failures.extend(_branch_split_failures(spawned, active_branches))
        branch_throughput = _latest_mapping(step_rows, "branch_throughput")
        branch_travel = _latest_mapping(step_rows, "branch_travel_time_mean")
        missing_throughput = [branch_id for branch_id in active_branches if branch_id not in branch_throughput]
        missing_travel = [branch_id for branch_id in active_branches if branch_id not in branch_travel]
        if missing_throughput:
            failures.append(f"branch_throughput missing branches: {missing_throughput}")
        if missing_travel:
            failures.append(f"branch_travel_time_mean missing branches: {missing_travel}")
    fairness = _float(summary.get("fairness_jain"), 1.0)
    if fairness < 0.0 or fairness > 1.0:
        failures.append(f"fairness_jain outside [0, 1]: {fairness}")
    return failures


def _branch_split_failures(spawned: Mapping[str, Any], branch_ids: Sequence[str]) -> list[str]:
    counts = {branch_id: int(spawned.get(branch_id, 0)) for branch_id in branch_ids}
    total = sum(counts.values())
    if not branch_ids or total < 2 * len(branch_ids):
        return []
    expected = 1.0 / len(branch_ids)
    tolerance = 0.35
    failures = []
    for branch_id, count in counts.items():
        observed = count / total
        if abs(observed - expected) > tolerance:
            failures.append(
                f"branch split for {branch_id} outside loose tolerance: observed={observed:.2f} expected={expected:.2f}"
            )
    return failures


def directional_checks(results: Sequence[RunResult]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    rows.extend(universal_directional_checks(results))
    rows.extend(ring_directional_checks(results))
    rows.extend(straight_directional_checks(results))
    rows.extend(multilane_directional_checks(results))
    rows.extend(merge_directional_checks(results))
    rows.extend(tree_directional_checks(results))
    return rows


def universal_directional_checks(results: Sequence[RunResult]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for topology in sorted({result.spec.topology for result in results}):
        group = [result for result in results if result.spec.topology == topology and result.spec.demand == "low"]
        selfish = _median_window_metric(group, "selfish_av", "early", "mean_speed")
        density = _median_window_metric(group, "density_lookup", "early", "mean_speed")
        dynamic = _median_window_metric(group, "dynamic_speed_limit", "early", "mean_speed")
        rows.append(_comparison_row(topology, "low", "selfish early speed >= density/dynamic", selfish, max(density, dynamic), ">="))
        random_safety = _median_safety_events(group, "random_av")
        for baseline in ("density_lookup", "dynamic_speed_limit", "cooperative_smoothing"):
            structured = _median_safety_events(group, baseline)
            rows.append(_comparison_row(topology, "low", f"{baseline} safety events <= random", structured, random_safety, "<="))
    return rows


def ring_directional_checks(results: Sequence[RunResult]) -> list[dict[str, Any]]:
    group = [result for result in results if result.spec.topology == "ring" and result.spec.demand in {"high", "burst"}]
    rows: list[dict[str, Any]] = []
    random_std = _median_window_metric(group, "random_av", "saturated", "speed_std")
    random_jam = _median_window_metric(group, "random_av", "saturated", "jam_fraction")
    for baseline in ("density_lookup", "dynamic_speed_limit", "av_mediated_speed_harmonization"):
        std = _median_window_metric(group, baseline, "saturated", "speed_std")
        jam = _median_window_metric(group, baseline, "saturated", "jam_fraction")
        rows.append(_dual_metric_row("ring", "high/burst", f"{baseline} reduces speed_std or jam vs random", std, random_std, jam, random_jam))
    harmonization_std = _median_window_metric(group, "av_mediated_speed_harmonization", "saturated", "speed_std")
    selfish_std = _median_window_metric(group, "selfish_av", "saturated", "speed_std")
    rows.append(_comparison_row("ring", "high/burst", "selfish saturated speed_std not better than harmonization", selfish_std, harmonization_std, ">="))
    return rows


def straight_directional_checks(results: Sequence[RunResult]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    group_low = [result for result in results if result.spec.topology == "straight_single_lane" and result.spec.demand == "low"]
    selfish = _median_window_metric(group_low, "selfish_av", "early", "mean_speed")
    density = _median_window_metric(group_low, "density_lookup", "early", "mean_speed")
    rows.append(_comparison_row("straight_single_lane", "low", "selfish early speed >= density", selfish, density, ">="))
    group_stress = [result for result in results if result.spec.topology == "straight_single_lane" and result.spec.demand in {"high", "burst"}]
    random_throughput = _median_window_metric(group_stress, "random_av", "saturated", "throughput_recent")
    random_queue = _median_window_metric(group_stress, "random_av", "saturated", "queue_length_total")
    for baseline in ("density_lookup", "dynamic_speed_limit", "av_mediated_speed_harmonization"):
        throughput = _median_window_metric(group_stress, baseline, "saturated", "throughput_recent")
        queue = _median_window_metric(group_stress, baseline, "saturated", "queue_length_total")
        rows.append(
            _dual_metric_row(
                "straight_single_lane",
                "high/burst",
                f"{baseline} throughput higher or queue lower than random",
                random_throughput,
                throughput,
                queue,
                random_queue,
                first_relation="<=",
            )
        )
    return rows


def multilane_directional_checks(results: Sequence[RunResult]) -> list[dict[str, Any]]:
    group = [result for result in results if result.spec.topology == "straight_multilane" and result.spec.demand in {"high", "burst"}]
    rows: list[dict[str, Any]] = []
    selfish_lane = _median_window_metric(group, "selfish_av", "saturated", "lane_change_count") + _median_blocked_events(group, "selfish_av")
    density_lane = _median_window_metric(group, "density_lookup", "saturated", "lane_change_count") + _median_blocked_events(group, "density_lookup")
    rows.append(_comparison_row("straight_multilane", "high/burst", "selfish lane activity >= density", selfish_lane, density_lane, ">="))
    for baseline in ("density_lookup", "av_mediated_speed_harmonization"):
        roadblock = _median_window_metric(group, baseline, "saturated", "rolling_roadblock_score")
        rows.append(_threshold_row("straight_multilane", "high/burst", f"{baseline} roadblock score near zero", roadblock, 0.05, "<="))
    random_hard = _median_window_metric(group, "random_av", "saturated", "hard_brakes_caused_by_av")
    for baseline in ("density_lookup", "dynamic_speed_limit", "av_mediated_speed_harmonization", "cooperative_smoothing"):
        hard = _median_window_metric(group, baseline, "saturated", "hard_brakes_caused_by_av")
        rows.append(_comparison_row("straight_multilane", "high/burst", f"{baseline} hard brakes <= random", hard, random_hard, "<="))
    return rows


def merge_directional_checks(results: Sequence[RunResult]) -> list[dict[str, Any]]:
    group = [result for result in results if result.spec.topology == "merge" and result.spec.demand in {"high", "burst"}]
    rows: list[dict[str, Any]] = []
    random_queue = _median_window_metric(group, "random_av", "saturated", "queue_length_total")
    random_jam = _median_window_metric(group, "random_av", "saturated", "jam_fraction")
    for baseline in ("backpressure", "cooperative_smoothing"):
        queue = _median_window_metric(group, baseline, "saturated", "queue_length_total")
        jam = _median_window_metric(group, baseline, "saturated", "jam_fraction")
        rows.append(_dual_metric_row("merge", "high/burst", f"{baseline} queue or jam lower than random", queue, random_queue, jam, random_jam))
    backpressure_gap = _median_create_gap_events(group, "backpressure")
    density_gap = _median_create_gap_events(group, "density_lookup")
    rows.append(_comparison_row("merge", "high/burst", "backpressure create_gap >= density", backpressure_gap, density_gap, ">="))
    follower = _median_window_metric(group, "backpressure", "saturated", "follower_disruption_blocked_count")
    rows.append(_threshold_row("merge", "high/burst", "backpressure follower disruption low", follower, 2.0, "<="))
    return rows


def tree_directional_checks(results: Sequence[RunResult]) -> list[dict[str, Any]]:
    group = [result for result in results if result.spec.topology == "inverted_tree" and result.spec.demand in {"high", "burst"}]
    rows: list[dict[str, Any]] = []
    reference = max(
        _median_window_metric(group, "selfish_av", "saturated", "fairness_jain"),
        _median_window_metric(group, "random_av", "saturated", "fairness_jain"),
    )
    for baseline in ("backpressure", "cooperative_smoothing"):
        fairness = _median_window_metric(group, baseline, "saturated", "fairness_jain")
        rows.append(_comparison_row("inverted_tree", "high/burst", f"{baseline} fairness preserves random/selfish", fairness, reference, ">="))
    for baseline in sorted({result.spec.baseline for result in group}):
        starvation = _branch_starvation_rate(group, baseline)
        rows.append(_threshold_row("inverted_tree", "high/burst", f"{baseline} no branch starvation", starvation, 0.0, "<="))
    bottleneck_group = [
        result for result in results if result.spec.topology == "inverted_tree_bottleneck" and result.spec.demand in {"high", "burst"}
    ]
    spillback = max(
        _median_window_metric(bottleneck_group, baseline, "saturated", "queue_length_total")
        for baseline in ("random_av", "backpressure", "cooperative_smoothing")
    )
    rows.append(_threshold_row("inverted_tree_bottleneck", "high/burst", "stress scenario has nontrivial queues", spillback, 0.0, ">"))
    return rows


def write_reports(output_root: Path, results: Sequence[RunResult], directional_rows: Sequence[Mapping[str, Any]]) -> None:
    output_root.mkdir(parents=True, exist_ok=True)
    runs_csv = output_root / "run_summary.csv"
    with runs_csv.open("w", newline="") as handle:
        fieldnames = [
            "topology",
            "baseline",
            "demand",
            "seed",
            "hard_failure_count",
            "completed_vehicle_count",
            "active_vehicle_count",
            "travel_time_mean",
            "fairness_jain",
            "episode_summary",
            "step_metrics",
            "segment_metrics",
        ]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for result in results:
            writer.writerow(
                {
                    "topology": result.spec.topology,
                    "baseline": result.spec.baseline,
                    "demand": result.spec.demand,
                    "seed": result.spec.seed,
                    "hard_failure_count": len(result.hard_failures),
                    "completed_vehicle_count": result.summary.get("completed_vehicle_count", 0),
                    "active_vehicle_count": result.summary.get("active_vehicle_count", 0),
                    "travel_time_mean": result.summary.get("travel_time_mean", 0.0),
                    "fairness_jain": result.summary.get("fairness_jain", 1.0),
                    "episode_summary": result.paths.get("episode_summary", ""),
                    "step_metrics": result.paths.get("step_metrics", ""),
                    "segment_metrics": result.paths.get("segment_metrics", ""),
                }
            )

    directional_csv = output_root / "directional_checks.csv"
    with directional_csv.open("w", newline="") as handle:
        fieldnames = ["topology", "demand", "check", "status", "observed", "reference", "detail"]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in directional_rows:
            writer.writerow({field: row.get(field, "") for field in fieldnames})

    hard_failures = [
        {
            "topology": result.spec.topology,
            "baseline": result.spec.baseline,
            "demand": result.spec.demand,
            "seed": result.spec.seed,
            "failures": result.hard_failures,
        }
        for result in results
        if result.hard_failures
    ]
    (output_root / "validation_summary.json").write_text(
        json.dumps(
            {
                "run_count": len(results),
                "hard_failures": hard_failures,
                "directional_checks": list(directional_rows),
            },
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )
    write_markdown_summary(output_root / "validation_summary.md", results, directional_rows, hard_failures)


def write_markdown_summary(
    path: Path,
    results: Sequence[RunResult],
    directional_rows: Sequence[Mapping[str, Any]],
    hard_failures: Sequence[Mapping[str, Any]],
) -> None:
    lines = [
        "# Task 10 Topology Baseline Validation",
        "",
        f"- Runs: {len(results)}",
        f"- Hard failure runs: {len(hard_failures)}",
        f"- Directional warnings: {sum(1 for row in directional_rows if row.get('status') == 'warn')}",
        f"- Insufficient signal checks: {sum(1 for row in directional_rows if row.get('status') == 'insufficient_signal')}",
        "",
        "## Hard Failures",
        "",
    ]
    if not hard_failures:
        lines.append("None.")
    else:
        for failure in hard_failures:
            lines.append(
                f"- {failure['topology']} / {failure['baseline']} / {failure['demand']} / seed {failure['seed']}: "
                + "; ".join(failure["failures"])
            )
    lines.extend(["", "## Directional Checks", ""])
    for row in directional_rows:
        lines.append(
            f"- **{row['status']}** `{row['topology']}` `{row['demand']}`: {row['check']} "
            f"(observed={row.get('observed')}, reference={row.get('reference')}) {row.get('detail', '')}"
        )
    path.write_text("\n".join(lines) + "\n")


def _segment_invariant_failures(step: int, segment_metrics: Mapping[str, Mapping[str, Any]]) -> list[str]:
    failures: list[str] = []
    for segment_id, segment in segment_metrics.items():
        for key in ("vehicle_count", "av_count", "density", "queue_length", "mean_speed", "inflow", "outflow"):
            value = _float(segment.get(key), 0.0)
            if value < 0:
                failures.append(f"negative {key} for {segment_id} at step {step}: {value}")
        for key in ("inflow", "outflow"):
            value = segment.get(key, 0)
            if int(value) != value:
                failures.append(f"non-integer {key} for {segment_id} at step {step}: {value}")
    return failures


def _csv_values(raw: str) -> tuple[str, ...]:
    return tuple(value.strip() for value in raw.split(",") if value.strip())


def _branch_ids_for_topology(topology_id: str) -> tuple[str, ...]:
    topology = build_topology(topology_id)
    if topology_id == "ring":
        return ("initial",)
    if topology_id.startswith("straight"):
        return ("main",)
    if topology_id == "merge":
        return ("main", "ramp")
    if topology_id in {"inverted_tree", "inverted_tree_bottleneck"}:
        return tuple(segment.removeprefix("tree_leaf_") for segment in topology.entry_segments)
    return tuple(topology.entry_segments)


def _latest_mapping(rows: Sequence[Mapping[str, Any]], key: str) -> dict[str, Any]:
    for row in reversed(rows):
        value = row.get(key)
        if isinstance(value, Mapping):
            return dict(value)
    return {}


def _window_values(result: RunResult, window: str, key: str) -> list[float]:
    rows = result.step_rows
    if not rows:
        return []
    count = len(rows)
    if window == "early":
        selected = rows[: max(1, math.ceil(count * 0.2))]
    elif window == "saturated":
        selected = rows[max(0, math.floor(count * 0.6)) :]
    elif window == "burst":
        selected = rows[max(0, math.floor(count * 0.4)) :]
    else:
        selected = rows
    return [_float(row.get(key), 0.0) for row in selected if key in row]


def _median_window_metric(results: Sequence[RunResult], baseline: str, window: str, key: str) -> float:
    values = [_median(_window_values(result, window, key)) for result in results if result.spec.baseline == baseline]
    return _median([value for value in values if math.isfinite(value)])


def _median_safety_events(results: Sequence[RunResult], baseline: str) -> float:
    keys = (
        "safety_masked_action_count",
        "etiquette_blocked_action_count",
        "follower_disruption_blocked_count",
        "external_safety_override_count",
    )
    values = []
    for result in results:
        if result.spec.baseline != baseline:
            continue
        totals = [sum(_float(row.get(key), 0.0) for key in keys) for row in result.step_rows]
        values.append(_median(totals))
    return _median(values)


def _median_blocked_events(results: Sequence[RunResult], baseline: str) -> float:
    values = []
    for result in results:
        if result.spec.baseline != baseline:
            continue
        totals = [
            _float(row.get("simulator_blocked_action_count"), 0.0) + _float(row.get("safety_masked_action_count"), 0.0)
            for row in result.step_rows
        ]
        values.append(_median(totals))
    return _median(values)


def _median_create_gap_events(results: Sequence[RunResult], baseline: str) -> float:
    values = []
    for result in results:
        if result.spec.baseline != baseline:
            continue
        values.append(_median(_window_values(result, "saturated", "action_create_gap_count")))
    return _median(values)


def _branch_starvation_rate(results: Sequence[RunResult], baseline: str) -> float:
    rates = []
    for result in results:
        if result.spec.baseline != baseline:
            continue
        throughput = _latest_mapping(result.step_rows, "branch_throughput")
        if not throughput:
            continue
        counts = [int(value) for value in throughput.values()]
        if max(counts, default=0) > 0 and min(counts, default=0) == 0:
            rates.append(1.0)
        else:
            rates.append(0.0)
    return _median(rates)


def _comparison_row(
    topology: str,
    demand: str,
    check: str,
    observed: float,
    reference: float,
    relation: str,
) -> dict[str, Any]:
    if not math.isfinite(observed) or not math.isfinite(reference):
        return _row(topology, demand, check, "insufficient_signal", observed, reference, "missing metric samples")
    passed = observed >= reference if relation == ">=" else observed <= reference if relation == "<=" else observed > reference
    return _row(topology, demand, check, "pass" if passed else "warn", observed, reference, relation)


def _threshold_row(
    topology: str,
    demand: str,
    check: str,
    observed: float,
    threshold: float,
    relation: str,
) -> dict[str, Any]:
    if not math.isfinite(observed):
        return _row(topology, demand, check, "insufficient_signal", observed, threshold, "missing metric samples")
    passed = observed <= threshold if relation == "<=" else observed > threshold
    return _row(topology, demand, check, "pass" if passed else "warn", observed, threshold, relation)


def _dual_metric_row(
    topology: str,
    demand: str,
    check: str,
    observed_a: float,
    reference_a: float,
    observed_b: float,
    reference_b: float,
    first_relation: str = "<=",
) -> dict[str, Any]:
    if not all(math.isfinite(value) for value in (observed_a, reference_a, observed_b, reference_b)):
        return _row(
            topology,
            demand,
            check,
            "insufficient_signal",
            {"metric_a": observed_a, "metric_b": observed_b},
            {"metric_a": reference_a, "metric_b": reference_b},
            "missing metric samples",
        )
    first_pass = observed_a <= reference_a if first_relation == "<=" else observed_a >= reference_a
    second_pass = observed_b <= reference_b
    return _row(
        topology,
        demand,
        check,
        "pass" if first_pass or second_pass else "warn",
        {"metric_a": observed_a, "metric_b": observed_b},
        {"metric_a": reference_a, "metric_b": reference_b},
        "any metric passes",
    )


def _row(
    topology: str,
    demand: str,
    check: str,
    status: str,
    observed: Any,
    reference: Any,
    detail: str,
) -> dict[str, Any]:
    return {
        "topology": topology,
        "demand": demand,
        "check": check,
        "status": status,
        "observed": observed,
        "reference": reference,
        "detail": detail,
    }


def _median(values: Sequence[float]) -> float:
    clean = sorted(value for value in values if math.isfinite(value))
    if not clean:
        return float("nan")
    midpoint = len(clean) // 2
    if len(clean) % 2:
        return clean[midpoint]
    return (clean[midpoint - 1] + clean[midpoint]) / 2.0


def _float(value: Any, default: float = 0.0) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return default
    return result if math.isfinite(result) else default


def _action_counts(actions: Mapping[str, Mapping[str, Any]]) -> dict[str, int]:
    counts = {
        "action_count": len(actions),
        "action_create_gap_count": 0,
        "action_hold_lane_count": 0,
        "action_lane_preference_count": 0,
        "action_slow_count": 0,
        "action_fast_count": 0,
    }
    for action_map in actions.values():
        if action_map.get("merge_mode") == "create_gap":
            counts["action_create_gap_count"] += 1
        if action_map.get("merge_mode") == "hold_lane":
            counts["action_hold_lane_count"] += 1
        if action_map.get("lane_preference") != "keep":
            counts["action_lane_preference_count"] += 1
        if action_map.get("desired_speed_bin") == "slow":
            counts["action_slow_count"] += 1
        if action_map.get("desired_speed_bin") == "fast":
            counts["action_fast_count"] += 1
    return counts


if __name__ == "__main__":
    raise SystemExit(main())
