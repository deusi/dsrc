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

Optional cooperative-intent fields should stay under a nested `cooperation` block:

- `cooperation.nearby_av_intent_summary`

The v1 cooperation contract exposes only local aggregate AV information. It should not expose individual neighboring AV identities or direct V2V messages.

If `nearby_av_count == 0`, aggregate cooperation fields should use neutral fallback values and the policy must still be able to operate as an individual local controller.

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
