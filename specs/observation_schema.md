# Observation Schema

This file defines the public observation contract for decentralized execution and centralized training.

## Local Observation

The public local observation format should be one mapping per AV:

```python
{
    "av_0": {...},
    "av_1": {...},
}
```

Each local AV observation should standardize the following keys:

- `is_active`
- `ego_speed`
- `ego_acceleration`
- `ego_lane`
- `ego_headway_s`
- `target_headway_s`
- `time_since_last_lane_change`
- `lane_changes_last_km`
- `current_segment`
- `distance_to_next_merge`
- `distance_to_downstream_bottleneck`
- `leader_gap`
- `leader_relative_speed`
- `follower_gap`
- `follower_relative_speed`
- `left_lane_front_gap`
- `left_lane_rear_gap`
- `right_lane_front_gap`
- `right_lane_rear_gap`
- `target_lane_front_gap`
- `target_lane_rear_gap`
- `target_lane_rear_required_decel`
- `downstream_congestion_estimate`
- `merge_pressure`
- `segment_target_speed`
- `uncongested_low_speed_flag`
- `local_density_bin`
- `local_mean_speed_bin`
- `local_queue_estimate`
- `active_vehicle_count_local`
- `active_av_count_local`
- `nearby_av_count`
- `nearby_av_density`
- `nearby_av_mean_speed`
- `nearby_av_lane_distribution`

Optional realism fields should stay under a nested `sensor` block rather than changing the top-level keys:

- `sensor.range_m`
- `sensor.latency_s`
- `sensor.position_noise_std`
- `sensor.speed_noise_std`

Default sensing configuration:

```yaml
sensing:
  range_m: 150.0
  latency_s: 0.0
  position_noise_std: 0.0
  speed_noise_std: 0.0
  density_bin_edges_veh_per_km: [12.0, 30.0]
  mean_speed_bin_edges_mps: [8.0, 18.0]
  queue_speed_mps: 5.0
```

Local counts and aggregates exclude the ego AV. `active_vehicle_count_local` counts all sensed non-ego vehicles within `range_m`; `active_av_count_local` counts only sensed non-ego AVs. Density bins are computed from `count / (2 * range_m / 1000)`. Mean speed bins use the mean speed of sensed vehicles, falling back to ego speed if no vehicles are sensed. `local_queue_estimate` counts sensed vehicles below `queue_speed_mps`.

When `latency_s > 0`, observations should be built from the newest stored sensing snapshot at least `latency_s` old. If the buffer is not warm at episode start, use the oldest available snapshot. Active AV identifiers still come from the current environment state; if an AV is no longer present in the delayed snapshot, omit it.

Optional cooperative aggregate fields should stay under a nested `cooperation` block:

- `cooperation.segment_target_speed`
- `cooperation.merge_pressure`
- `cooperation.downstream_congestion_estimate`

The v2 cooperation contract exposes only aggregate traffic-state information. It should not expose individual neighboring AV identities, direct V2V messages, AV-to-lane assignments, or coordinated lane-occupation plans.

If `nearby_av_count == 0`, aggregate cooperation fields should use neutral fallback values and the policy must still be able to operate as an individual local controller.

Neutral fallback values:

- `nearby_av_count: 0`
- `nearby_av_density: 0.0`
- `nearby_av_mean_speed: free_flow_speed`
- `nearby_av_lane_distribution: {}`
- `cooperation.segment_target_speed: free_flow_speed`
- `cooperation.merge_pressure: 0.0`
- `cooperation.downstream_congestion_estimate: 0.0`

Allowed communication/aggregation:

- local density
- mean speed
- queue estimate
- downstream congestion estimate
- segment-level target speed
- merge pressure

For non-learning baselines, these fields are available only as part of each AV's own public local observation. Baseline controllers must not reconstruct hidden segment state from environment internals, roadside detector/oracle sensing, global state, or runner-side aggregation across AVs.

Disallowed communication/aggregation:

- joint lane occupation plans
- instructions for which AV should occupy which lane
- coordinated roadblock formations

## Global Critic State

The public global state should be one dict with stable top-level sections:

- `time`
- `topology_id`
- `active_vehicle_count`
- `active_av_count`
- `completed_vehicle_count`
- `segment_state`
- `branch_state`
- `demand_state`

`active_vehicle_count` and `active_av_count` count only vehicles that are still on the topology. Vehicles that have exited should be excluded from AV/RV computation, local observations, rewards, and segment metrics; they should appear only in `completed_vehicle_count` or episode summaries.

`segment_state` should be keyed by canonical segment identifier.

## Segment Metrics

Every segment metric record should standardize:

- `vehicle_count`
- `av_count`
- `mean_speed`
- `speed_std`
- `density`
- `queue_length`
- `jam_fraction`
- `inflow`
- `outflow`

## Repo Ownership

The written contract lives here in `specs/`.

The executable interface should later be implemented under:

- `src/sensing/`
- `src/metrics/`
- `src/envs/base_ctde_env.py`
