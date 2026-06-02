# Config Schema

This file standardizes config families, file locations, and the composed experiment bundle.

## Config Families

All configs should use YAML and live under `configs/`:

- `configs/topology/`
- `configs/demand/`
- `configs/human_models/`
- `configs/experiments/`
- `configs/training/`

## Family Rules

Each family file should include:

- `id`
- `kind`
- family-specific content

Expected `kind` values:

- `topology`
- `demand`
- `human_model`
- `training`
- `experiment`

## Experiment Config Contract

Each experiment config should reference the family entries it composes:

```yaml
id: exp_ring_wave_damping
kind: experiment
refs:
  topology: ring
  demand: medium
  human_model: heterogeneous
  training: mappo
experiment:
  id: exp_ring_wave_damping
  seed: 7
controller:
  family: rl
  name: mappo
metrics:
  primary:
    - mean_speed
outputs:
  episode_summary: outputs/metrics/exp_ring_wave_damping/episode_summary.json
overrides:
  demand:
    total_vehicles_per_hour: 1600
```

## Merge Precedence

The composed config bundle should load in this order:

1. referenced family configs from `refs`
2. experiment-local sections such as `experiment`, `controller`, `sensing`, `metrics`, and `outputs`
3. `overrides` applied last to the referenced family sections

## Composed Bundle Shape

The loader should return one dict with these top-level sections:

- `experiment`
- `topology`
- `demand`
- `human_model`
- `training`
- `controller`
- `metrics`
- `outputs`
- `sensing` (optional)
- `resolved_refs`

## Optional Sensing Config

The environment may include a top-level `sensing` section. If omitted, deterministic local sensing defaults are used:

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

## Repo Ownership

Executable config loading should live in:

- `src/config/loaders.py`

Human-readable config guidance should live in:

- `configs/README.md`
