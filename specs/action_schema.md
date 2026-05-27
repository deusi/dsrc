# Action Schema

This file defines the v2 public AV action format. The action space is intentionally safety-constrained: AVs request smooth speed/headway targets and conservative lane preferences, while a safety and etiquette layer decides whether the request can be realized.

## Canonical Action Object

Every AV action should use the same dict structure:

```python
{
    "desired_speed_bin": str,
    "desired_headway_bin": str,
    "lane_preference": str,
    "merge_mode": str,
}
```

This is a breaking replacement for the old `desired_speed` + `desired_lane` contract.

## Field Definitions

`desired_speed_bin`

- allowed values:
  - `slow`
  - `nominal`
  - `fast`
- meaning: a discrete contextual speed target decoded by the wrapper or safety layer
- the decoder must respect speed limits, minimum contextual speed, and acceleration/deceleration bounds

`desired_headway_bin`

- allowed values:
  - `normal`
  - `larger`
  - `largest`
- meaning: a discrete following-distance target used for adaptive headway control and merge gap creation

`lane_preference`

- allowed values:
  - `keep`
  - `prefer_left_if_safe`
  - `prefer_right_if_safe`
- meaning: a conservative preference, not a direct lane-change command
- the safety layer may ignore or block the preference for dwell-time, rear-gap, lane-availability, merge-zone, or etiquette reasons

`merge_mode`

- allowed values:
  - `normal`
  - `create_gap`
  - `hold_lane`
- meaning:
  - `normal`: default smooth driving
  - `create_gap`: increase headway and smooth speed near a merge
  - `hold_lane`: suppress lateral movement near a merge or bottleneck

## Disallowed Actions

The public interface must not expose:

- raw acceleration, braking, steering, or lateral motion
- arbitrary low speed targets
- direct `left` or `right` lane-change commands
- lane occupation plans such as “cover all lanes”
- coordinated roadblock or blocking maneuvers

Humans must remain able to pass AVs when safe. AV-mediated speed harmonization should come from partial compliance and car-following dynamics, not deliberate obstruction.

## Hard Safety and Etiquette Checks

The safety/control layer should enforce hard constraints, not only reward penalties:

- minimum lane-change dwell time, default `15 s`
- maximum lane changes per km, default `2 / km`
- target lane must exist
- target lane must have safe front and rear gaps
- target-lane rear vehicle must not need to brake harder than `2.5 m/s^2`
- acceleration and deceleration must be bounded
- speed target must not fall below a contextual legal/safety minimum
- low-speed driving in uncongested conditions should be blocked
- synchronized all-lane AV slowdown should be blocked unless downstream congestion justifies it

## Diagnostics

When an action is blocked, masked, or modified, the environment `info` payload should expose diagnostics with stable event names:

- `safety_masked_action`
- `etiquette_blocked_action`
- `follower_disruption_blocked`
- `simulator_blocked_action`

## Action Mapping

Controllers should return one mapping keyed by AV identifier:

```python
{
    "av_0": {
        "desired_speed_bin": "nominal",
        "desired_headway_bin": "normal",
        "lane_preference": "keep",
        "merge_mode": "normal",
    },
    "av_1": {
        "desired_speed_bin": "slow",
        "desired_headway_bin": "larger",
        "lane_preference": "prefer_left_if_safe",
        "merge_mode": "create_gap",
    },
}
```

## Repo Ownership

The executable contract should live in:

- `src/envs/base_ctde_env.py`
- `src/envs/wrappers.py`
- `src/controllers/base.py`

Safety and etiquette conversion should live under:

- `src/safety/`
