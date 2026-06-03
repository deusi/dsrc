# Controller Contract

This file defines the controller interface shared by baselines and learned policies.

## Purpose

Controllers should be interchangeable across topology sweeps, demand sweeps, and RL evaluation without changing caller code.

## Public Interface

Every controller should support:

```python
reset(env_metadata=None, seed=None) -> None
act(local_obs, global_state=None) -> dict[str, dict]
```

Optional metadata should be exposed through:

- `name`
- `metadata`

## `reset(...)`

- called at episode start
- may receive topology or experiment metadata
- should clear any controller state

## `act(local_obs, global_state=None)`

- `local_obs` is required and keyed by AV identifier
- `global_state` is optional so the same interface works for decentralized baselines and CTDE controllers
- return value is the canonical v2 AV action mapping:

```python
{
    "av_0": {
        "desired_speed_bin": "nominal",
        "desired_headway_bin": "normal",
        "lane_preference": "keep",
        "merge_mode": "normal",
    }
}
```

## Metadata

Metadata should stay lightweight and include:

- `name`
- `family`
- `version`
- `requires_global_state`
- `cooperation_mode`
- `safety_mode`
- `supports_fallback_individual`

Allowed `cooperation_mode` values:

- `none`
- `local_aggregate`
- `global_state`

Allowed `safety_mode` values:

- `external_filter`
- `integrated_rl`
- `simulator_default`

Baselines should declare `external_filter` or `simulator_default`.

CTDE learned AV controllers should declare `integrated_rl`.

`integrated_rl` is the only mode that should use the full DSRC safety, etiquette, action-mask, penalty, and bounded physical-control layer. Non-CTDE AV baselines can declare `simulator_default` to use simulator/human-like safety behavior while retaining AV role accounting in metrics.

Non-learning baselines must declare `requires_global_state: false`. Their control decisions must use only the public `local_obs` passed to `act`; they must ignore or reject non-`None` `global_state`, must not read environment internals, and must not use runner-aggregated fleet or segment state. Global state remains available for metrics and CTDE critic training, not non-learning baseline control.

Cooperative learned controllers should declare `local_aggregate` and `supports_fallback_individual: true` when they consume local aggregate AV fields and can operate without neighboring AVs.

Controllers must not return joint lane-blocking plans, lane-coverage assignments, raw acceleration, raw braking, or direct left/right lane-change commands. Lane control is secondary and conservative; primary control is smooth speed/headway damping.

## Repo Ownership

The executable base contract should live in:

- `src/controllers/base.py`

Baseline and RL implementations should later live under:

- `src/baselines/`
- `src/rl/`
