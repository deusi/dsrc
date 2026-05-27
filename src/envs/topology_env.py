from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np

from src.envs.base_ctde_env import (
    AVActionMap,
    AVObservationMap,
    EpisodeSummary,
    GlobalState,
    InfoDict,
    RewardMap,
    SegmentMetrics,
)
from src.envs.base_ctde_env import BaseCTDEEnv
from src.envs.wrappers import default_agent_ids, validate_action_mapping
from src.road.highway_imports import ensure_highway_env_importable
from src.road.segment_graph import TopologySpec
from src.road.topology_factory import build_topology
from src.safety import SafetyConstraints, SafetyContext, SafetyState, apply_safety_layer
from src.safety.etiquette import is_low_speed_uncongested

ensure_highway_env_importable()

from highway_env.road.road import LaneIndex, Road
from highway_env.vehicle.controller import ControlledVehicle


class HighwayTopologyEnv(BaseCTDEEnv):
    """Minimal DSRC wrapper around project-owned HighwayEnv topology builders."""

    def __init__(
        self,
        topology_id: str,
        config: Mapping[str, Any] | None = None,
    ) -> None:
        self.topology_id = topology_id
        self.config = dict(config or {})
        self.topology: TopologySpec = build_topology(topology_id, self.config.get("road"))
        self.agent_ids = default_agent_ids(int(self.config.get("controlled_vehicles", 2)))
        self.road: Road | None = None
        self._vehicles: dict[str, ControlledVehicle] = {}
        self._safety_states: dict[str, SafetyState] = {}
        self._target_headways: dict[str, float] = {}
        self._completed_vehicle_count = 0
        self._step_count = 0
        self._time = 0.0

    def reset(
        self,
        config: Mapping[str, Any] | None = None,
        seed: int | None = None,
        options: Mapping[str, Any] | None = None,
    ) -> tuple[AVObservationMap, InfoDict]:
        if config:
            self.config.update(config)
            self.topology = build_topology(self.topology_id, self.config.get("road"))
        if options and "config" in options and isinstance(options["config"], Mapping):
            self.config.update(options["config"])
            self.topology = build_topology(self.topology_id, self.config.get("road"))

        self.agent_ids = default_agent_ids(int(self.config.get("controlled_vehicles", len(self.agent_ids))))
        self.road = Road(
            network=self.topology.road_network,
            np_random=np.random.RandomState(seed),
            record_history=False,
        )
        self._vehicles = {}
        self._safety_states = {}
        self._target_headways = {}
        self._completed_vehicle_count = 0
        self._step_count = 0
        self._time = 0.0
        self._spawn_controlled_vehicles()
        return self.get_local_observations(), {"topology_id": self.topology_id, "segment_ids": self.topology.segment_ids}

    def step(
        self,
        av_actions: AVActionMap,
    ) -> tuple[AVObservationMap, RewardMap, bool, bool, InfoDict]:
        if self.road is None:
            raise RuntimeError("environment must be reset before step")

        active_agent_ids = list(self._vehicles)
        normalized_actions = validate_action_mapping(av_actions, expected_agent_ids=active_agent_ids)
        diagnostics: dict[str, list[dict[str, Any]]] = {
            "safety_masked_action": [],
            "etiquette_blocked_action": [],
            "follower_disruption_blocked": [],
            "simulator_blocked_action": [],
        }
        for agent_id, action in normalized_actions.items():
            vehicle = self._vehicles[agent_id]
            safety_decision = apply_safety_layer(
                action,
                self._safety_states[agent_id],
                self._safety_context_for_vehicle(vehicle),
                self._safety_constraints(),
                agent_id=agent_id,
            )
            for key, events in safety_decision.diagnostics.items():
                diagnostics.setdefault(key, []).extend(events)
            vehicle.target_speed = np.clip(
                safety_decision.target_speed_mps,
                float(self.config.get("min_speed_mps", 0.0)),
                float(self.config.get("max_speed_mps", 40.0)),
            )
            self._target_headways[agent_id] = safety_decision.target_headway_s
            if self.topology.supports_lane_change and safety_decision.lane_action is not None:
                before = vehicle.target_lane_index
                vehicle.act(safety_decision.lane_action)
                if vehicle.target_lane_index == before:
                    diagnostics["simulator_blocked_action"].append({"agent_id": agent_id, "lane_action": safety_decision.lane_action})
                else:
                    state = self._safety_states[agent_id]
                    state.last_lane_change_time_s = self._time
                    state.lane_changes_last_km += 1
                    state.last_lane_index = vehicle.target_lane_index

        self.road.act()
        self.road.step(float(self.config.get("dt", 1.0)))
        self._step_count += 1
        self._time += float(self.config.get("dt", 1.0))
        self._update_safety_distances()
        self._clear_exited_vehicles()

        observations = self.get_local_observations()
        rewards = {agent_id: self._reward_for_vehicle(vehicle) for agent_id, vehicle in self._vehicles.items()}
        terminated = any(vehicle.crashed for vehicle in self._vehicles.values())
        truncated = self._step_count >= int(self.config.get("duration_steps", 120))
        info: InfoDict = {
            "topology_id": self.topology_id,
            "time": self._time,
            "diagnostics": diagnostics,
        }
        return observations, rewards, terminated, truncated, info

    def get_local_observations(self) -> AVObservationMap:
        return {agent_id: self._observation_for_vehicle(vehicle) for agent_id, vehicle in self._vehicles.items()}

    def get_global_state(self) -> GlobalState:
        return {
            "time": self._time,
            "topology_id": self.topology_id,
            "active_vehicle_count": len(self._vehicles),
            "active_av_count": len(self._vehicles),
            "completed_vehicle_count": self._completed_vehicle_count,
            "segment_state": self.get_segment_metrics(),
            "branch_state": {},
            "demand_state": {},
        }

    def get_segment_metrics(self) -> SegmentMetrics:
        records: dict[str, dict[str, Any]] = {
            segment_id: {
                "vehicle_count": 0,
                "av_count": 0,
                "mean_speed": 0.0,
                "speed_std": 0.0,
                "density": 0.0,
                "queue_length": 0,
                "jam_fraction": 0.0,
                "inflow": 0,
                "outflow": 0,
                "lane_changes_per_av_km": 0.0,
                "rolling_roadblock_score": 0.0,
                "all_lane_av_low_speed_occupancy": 0.0,
            }
            for segment_id in self.topology.segment_ids
        }
        speeds: dict[str, list[float]] = {segment_id: [] for segment_id in self.topology.segment_ids}
        for vehicle in self._vehicles.values():
            segment_id = self.topology.segment_for_lane(vehicle.lane_index)
            if segment_id is None:
                continue
            records[segment_id]["vehicle_count"] += 1
            records[segment_id]["av_count"] += 1
            speeds[segment_id].append(float(vehicle.speed))
        for segment_id, segment_speeds in speeds.items():
            length_km = self.topology.segment_lengths[segment_id] / 1000.0
            records[segment_id]["density"] = records[segment_id]["vehicle_count"] / max(length_km, 1e-9)
            if segment_speeds:
                records[segment_id]["mean_speed"] = float(np.mean(segment_speeds))
                records[segment_id]["speed_std"] = float(np.std(segment_speeds))
                records[segment_id]["jam_fraction"] = float(np.mean([speed < 5.0 for speed in segment_speeds]))
                records[segment_id]["queue_length"] = int(sum(speed < 5.0 for speed in segment_speeds))
        return records

    def get_episode_summary(self) -> EpisodeSummary:
        return {
            "topology_id": self.topology_id,
            "steps": self._step_count,
            "time": self._time,
            "active_vehicle_count": len(self._vehicles),
            "completed_vehicle_count": self._completed_vehicle_count,
        }

    def _spawn_controlled_vehicles(self) -> None:
        if self.road is None:
            raise RuntimeError("road is not initialized")
        spawn_lanes, destination = self._spawn_lanes_and_destination()
        for index, agent_id in enumerate(self.agent_ids):
            lane_index = spawn_lanes[index % len(spawn_lanes)]
            lane = self.road.network.get_lane(lane_index)
            longitudinal = min(20.0 + 35.0 * index, max(5.0, lane.length - 10.0))
            vehicle = ControlledVehicle.make_on_lane(
                self.road,
                lane_index,
                longitudinal=longitudinal,
                speed=float(self.config.get("initial_speed_mps", min(lane.speed_limit or 24.0, 24.0))),
            )
            vehicle.plan_route_to(destination)
            self.road.vehicles.append(vehicle)
            self._vehicles[agent_id] = vehicle
            self._safety_states[agent_id] = SafetyState(last_lane_index=vehicle.lane_index)
            self._target_headways[agent_id] = 1.6

    def _spawn_lanes_and_destination(self) -> tuple[list[LaneIndex], str]:
        lanes_by_topology: dict[str, tuple[list[LaneIndex], str]] = {
            "ring": ([( "r0", "r1", 0)], "r0"),
            "straight_single_lane": ([("s0", "s1", 0)], "s3"),
            "straight_multilane": ([("s0", "s1", 0), ("s0", "s1", 1), ("s0", "s1", 2)], "s3"),
            "merge": ([("m0", "m1", 0), ("m0", "m1", 1), ("r0", "m1", 0)], "m3"),
            "inverted_tree": (
                [
                    ("a1_entry", "b1", 0),
                    ("a2_entry", "b1", 0),
                    ("a3_entry", "b1", 0),
                    ("a4_entry", "b2", 0),
                    ("a5_entry", "b2", 0),
                    ("a6_entry", "b2", 0),
                ],
                "exit",
            ),
        }
        return lanes_by_topology[self.topology_id]

    def _clear_exited_vehicles(self) -> None:
        if self.road is None or self.topology_id == "ring":
            return
        active: dict[str, ControlledVehicle] = {}
        for agent_id, vehicle in self._vehicles.items():
            if self._has_exited(vehicle):
                self._completed_vehicle_count += 1
            else:
                active[agent_id] = vehicle
        self._vehicles = active
        self._safety_states = {agent_id: state for agent_id, state in self._safety_states.items() if agent_id in active}
        self._target_headways = {agent_id: headway for agent_id, headway in self._target_headways.items() if agent_id in active}
        self.road.vehicles = [vehicle for vehicle in self.road.vehicles if vehicle in self._vehicles.values()]

    def _has_exited(self, vehicle: ControlledVehicle) -> bool:
        if vehicle.lane_index is None:
            return True
        segment_id = self.topology.segment_for_lane(vehicle.lane_index)
        if segment_id not in self.topology.exit_segments:
            return False
        lane = self.road.network.get_lane(vehicle.lane_index) if self.road is not None else None
        if lane is None:
            return True
        longitudinal, _ = lane.local_coordinates(vehicle.position)
        return longitudinal >= lane.length - vehicle.LENGTH

    def _observation_for_vehicle(self, vehicle: ControlledVehicle) -> dict[str, Any]:
        segment_id = self.topology.segment_for_lane(vehicle.lane_index)
        leader, follower = (None, None) if self.road is None else self.road.neighbour_vehicles(vehicle)
        lane_id = vehicle.lane_index[2] if vehicle.lane_index is not None else -1
        agent_id = self._agent_id_for_vehicle(vehicle)
        state = self._safety_states[agent_id] if agent_id is not None else SafetyState()
        target_headway = self._target_headways.get(agent_id or "", 1.6)
        ego_headway = self._headway_s(vehicle, leader)
        density = self.get_segment_metrics().get(segment_id, {}).get("density", 0.0) if segment_id else 0.0
        free_flow_speed = self._free_flow_speed_for_vehicle(vehicle)
        uncongested_low_speed = is_low_speed_uncongested(float(vehicle.speed), free_flow_speed, float(density), self._safety_constraints())
        return {
            "is_active": True,
            "ego_speed": float(vehicle.speed),
            "ego_acceleration": float(vehicle.action.get("acceleration", 0.0)),
            "ego_lane": int(lane_id),
            "ego_headway_s": ego_headway,
            "target_headway_s": target_headway,
            "time_since_last_lane_change": (
                float("inf") if state.last_lane_change_time_s is None else self._time - state.last_lane_change_time_s
            ),
            "lane_changes_last_km": state.lane_changes_last_km,
            "current_segment": segment_id,
            "distance_to_next_merge": 0.0,
            "distance_to_downstream_bottleneck": 0.0 if segment_id in self.topology.bottleneck_segments else float("inf"),
            "leader_gap": float(vehicle.lane_distance_to(leader)) if leader is not None else float("inf"),
            "leader_relative_speed": float(leader.speed - vehicle.speed) if leader is not None else 0.0,
            "follower_gap": float(vehicle.lane_distance_to(follower)) if follower is not None else float("inf"),
            "follower_relative_speed": float(follower.speed - vehicle.speed) if follower is not None else 0.0,
            "left_lane_front_gap": float("inf"),
            "left_lane_rear_gap": float("inf"),
            "right_lane_front_gap": float("inf"),
            "right_lane_rear_gap": float("inf"),
            "target_lane_front_gap": float("inf"),
            "target_lane_rear_gap": float("inf"),
            "target_lane_rear_required_decel": 0.0,
            "downstream_congestion_estimate": 0.0,
            "merge_pressure": 0.0,
            "segment_target_speed": free_flow_speed,
            "uncongested_low_speed_flag": uncongested_low_speed,
            "local_density_bin": 0,
            "local_mean_speed_bin": 0,
            "local_queue_estimate": 0,
            "active_vehicle_count_local": len(self._vehicles),
            "active_av_count_local": len(self._vehicles),
            "nearby_av_count": max(0, len(self._vehicles) - 1),
            "nearby_av_density": 0.0,
            "nearby_av_mean_speed": 0.0,
            "nearby_av_lane_distribution": {},
        }

    @staticmethod
    def _reward_for_vehicle(vehicle: ControlledVehicle) -> float:
        return 0.0 if vehicle.crashed else float(vehicle.speed)

    def _safety_constraints(self) -> SafetyConstraints:
        cfg = self.config.get("safety", {})
        if not isinstance(cfg, Mapping):
            cfg = {}
        return SafetyConstraints(
            lane_change_dwell_s=float(cfg.get("lane_change_dwell_s", 15.0)),
            max_lane_changes_per_km=float(cfg.get("max_lane_changes_per_km", 2.0)),
            max_follower_braking_mps2=float(cfg.get("max_follower_braking_mps2", 2.5)),
            comfortable_decel_mps2=float(cfg.get("comfortable_decel_mps2", 3.0)),
        )

    def _safety_context_for_vehicle(self, vehicle: ControlledVehicle) -> SafetyContext:
        segment_id = self.topology.segment_for_lane(vehicle.lane_index)
        metrics = self.get_segment_metrics().get(segment_id, {}) if segment_id else {}
        free_flow_speed = self._free_flow_speed_for_vehicle(vehicle)
        target_lane_exists = True
        if vehicle.target_lane_index is not None:
            target_lane_exists = bool(self.road and vehicle.target_lane_index in self.road.network.lanes_dict())
        return SafetyContext(
            time_s=self._time,
            free_flow_speed_mps=free_flow_speed,
            min_contextual_speed_mps=float(self.config.get("min_contextual_speed_mps", 12.0)),
            local_density_veh_per_km=float(metrics.get("density", 0.0)),
            downstream_congested=False,
            target_lane_exists=target_lane_exists,
            target_lane_front_gap=float("inf"),
            target_lane_rear_gap=float("inf"),
            target_lane_rear_required_decel_mps2=0.0,
            av_mean_speed_mps=float(metrics.get("mean_speed", free_flow_speed)),
            local_mean_speed_mps=float(metrics.get("mean_speed", free_flow_speed)),
            near_merge=bool(segment_id and ("merge" in segment_id or segment_id in self.topology.bottleneck_segments)),
        )

    def _free_flow_speed_for_vehicle(self, vehicle: ControlledVehicle) -> float:
        if self.road is None or vehicle.lane_index is None:
            return 30.0
        lane = self.road.network.get_lane(vehicle.lane_index)
        return float(lane.speed_limit or 30.0)

    @staticmethod
    def _headway_s(vehicle: ControlledVehicle, leader: Any | None) -> float:
        if leader is None or vehicle.speed <= 0:
            return float("inf")
        return max(0.0, float(vehicle.lane_distance_to(leader)) / max(float(vehicle.speed), 1e-6))

    def _agent_id_for_vehicle(self, vehicle: ControlledVehicle) -> str | None:
        for agent_id, candidate in self._vehicles.items():
            if candidate is vehicle:
                return agent_id
        return None

    def _update_safety_distances(self) -> None:
        dt = float(self.config.get("dt", 1.0))
        for agent_id, vehicle in self._vehicles.items():
            state = self._safety_states[agent_id]
            state.distance_since_window_start_m += max(0.0, float(vehicle.speed)) * dt
            if state.distance_since_window_start_m >= 1000.0:
                state.distance_since_window_start_m = 0.0
                state.lane_changes_last_km = 0
